# llama.cpp and llm.c on Apple Silicon: Metal Kernel Tricks

## Sources
- llama.cpp discussion (M-series perf): https://github.com/ggml-org/llama.cpp/discussions/4167
- llama.cpp repo: https://github.com/ggml-org/llama.cpp
- llama.cpp Metal vs Vulkan issue: https://github.com/ggml-org/llama.cpp/issues/10982
- IQ quant performance on Apple Silicon: https://github.com/ggml-org/llama.cpp/discussions/5617
- Comparative study: https://arxiv.org/pdf/2511.05502

---

## llama.cpp Metal Architecture

### Metal backend files
- `ggml-metal.metal`: All Metal compute kernels (MSL)
- `ggml-metal.m`: Objective-C bridge, kernel dispatch
- `ggml-metal-impl.h`: Kernel implementation templates

### Key design patterns

**Threadgroup layout**:
- `N_R0`: number of rows per SIMD group
- `N_SG`: number of SIMD groups per threadgroup
- Template parameters for compile-time specialization

**Kernel arguments**: Passed via packed C structs (minimizes argument setup overhead):
```c
struct ggml_metal_kargs_mul_mv_q4_0_f32 {
    int64_t ne00;  // embedding dim
    int64_t ne01;  // sequence length
    // ... etc
};
```

**Quantization format families**:
- Q4_0: Simple 4-bit with per-block scale (block=32 weights). Single float16 scale per 32 weights.
- Q4_1: 4-bit with per-block scale AND zero-point (better quality than Q4_0)
- Q4_K_M: K-quant — uses superblocks with embedded scale quantization. More accurate but slower on Metal.
- Q8_0: 8-bit with per-block scale

### Metal vs CUDA performance patterns

**Q4_0 on M2 Max**: ~63 t/s for 7B model
**Q4_0 on RTX 4080**: ~128 t/s
→ Apple GPU is 2x slower but uses 1/4 the power

**Q4_K_M problem**: K-quant formats use lookup tables (codebooks) for encoding. The LUT lookup pattern (many random small reads) is cache-unfriendly on Apple Silicon's unified memory architecture. Result: Q4_K_M is significantly slower than Q4_0 on Apple Silicon, even though Q4_K_M is faster on NVIDIA.

**Practical implication for mllm**: The MLX group-wise scale+bias format (similar to Q4_1) is better suited to Apple Silicon than K-quant LUT formats. MLX achieves ~230 t/s vs llama.cpp's ~150 t/s on similar models.

### Memory management (macOS 15+ / iOS 18+)
```objc
// Uses MTLResidencySet to keep model weights in GPU resident memory
id<MTLResidencySet> residencySet = [device newResidencySetWithDescriptor:...];
[residencySet addAllocation:buffer];
[residencySet requestResidency];
```

This prevents GPU memory eviction — critical for large models and repeated inference calls.

### Partial GPU offload
llama.cpp supports partial offload: some layers on GPU, some on CPU. Useful when model doesn't fit in GPU memory (unified memory for Apple Silicon means this is less of an issue).

---

## Key Metal Tricks Not Exposed by MLX

### 1. Residency Sets
MLX may not guarantee persistent GPU residency for weights. llama.cpp explicitly pins weights via residency sets. For mllm inference loop, explicitly requesting GPU residency for weight buffers could prevent latency spikes from eviction.

### 2. CPU+GPU Hybrid Inference
llama.cpp routes tensor operations to CPU when GPU is saturated. For very small models like 117M, GPU utilization may be low enough that CPU NEON ops are competitive. MLX's model offloading is less granular.

### 3. SIMD Group Matrix Extensions (Metal 3)
Metal 3 (M2+) introduced `simdgroup_matrix` for accelerated 8×8 matrix multiply in shaders. This is the Metal equivalent of tensor cores. Usage:
```msl
simdgroup_float8x8 A, B, C;
simdgroup_multiply_accumulate(C, A, B, C);  // C += A * B
```

This is not exposed via MLX's Python API — only through custom Metal kernels. For a tiled int4 matmul:
1. Load 8×8 block of dequantized weights into SIMD group matrix
2. Load 8×8 block of activations
3. Use `simdgroup_multiply_accumulate` for FP16 GEMM
4. Accumulate results

**Performance**: simdgroup_matrix ops run at M2's full FP16 rate: ~6.8 TFLOPS on M2 Pro, ~13.6 TFLOPS on M2 Max.

### 4. Async Copy (simdgroup_async_copy)
Used by Metal FlashAttention to overlap memory loads with computation:
```msl
// Load next tile while computing current tile
simdgroup_async_copy(destination, src, tile_size);
// ... compute on current tile ...
simdgroup_async_copy_wait();
// Process loaded tile
```

This requires careful double-buffering but can hide memory latency entirely.

---

## llm.c on Apple Silicon

llm.c (Karpathy, 2024) targets CUDA primarily but has CPU paths. Key techniques relevant to Apple Silicon:

- **FlashAttention-2 CPU port**: Tiled attention computation, avoids materializing full NxN attention matrix
- **BLAS integration**: Uses Apple's Accelerate framework for SGEMM on CPU
- **Memory mapping**: Model weights memory-mapped from disk, loaded on-demand

The CPU path in llm.c (via Accelerate) is competitive with MLX GPU for very small models like 117M because:
1. M4's CPU (4 performance cores) can sustain ~2.6 TFLOPS FP32 via NEON + AMX
2. Memory bandwidth to CPU is same as GPU (unified memory)
3. Cache effects favor CPU for very small weight matrices

### Practical comparison for 117M model
At 117M with batch=1 decode:
- MLX GPU: ~625 t/s (current, with int4 KV cache)
- MLX CPU: ~100-200 t/s (estimated, without GPU acceleration)
- llama.cpp Metal: likely 300-500 t/s (similar model size, better SIMD matrix usage)
- llama.cpp BLAS (CPU): ~50-100 t/s

The GPU advantage at 117M is real but not as large as for 7B+ models because the model weight loads are small enough for the CPU cache to handle.

---

## Takeaways for mllm

1. **Q4_1-style format** (scale + zero-point per 32 weights) > K-quant LUT format for Metal. Current MLX format is already in this family. Don't adopt K-quant.

2. **Residency sets**: For production inference loop, explicitly pin weight buffers in GPU memory via Metal residency API to avoid eviction latency spikes.

3. **SIMD Group Matrix**: For custom tiled matmul kernel, use `simdgroup_matrix` (Metal 3) to access hardware-accelerated 8×8 FP16 matmul. Not available via MLX Python API — requires custom MSL kernel.

4. **Async copy**: For prefill attention with long sequences, use `simdgroup_async_copy` to overlap Q/K/V loads with computation.

5. **MLX already beats llama.cpp** (~230 vs ~150 t/s for similar models) primarily because MLX's scale+bias format is better suited to Apple's memory subsystem than llama.cpp's K-quant LUT approach. The gap exists at the quantization format level, not kernel efficiency.
