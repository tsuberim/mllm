# Custom Metal Kernels: Flash Attention, Tiled Int4 Matmul, Metal-Specific Tricks

## Sources
- Metal Flash Attention: https://github.com/philipturner/metal-flash-attention
- Metal FA 2.0 blog: https://medium.com/engineering-draw-things/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c
- FlashMetal: https://github.com/at2005/FlashMetal
- MLX custom kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- MLX qmm PR: https://github.com/ml-explore/mlx/pull/2078
- Drawthings integration: https://engineering.drawthings.ai/p/integrating-metal-flashattention-accelerating-the-heart-of-image-generation-in-the-apple-ecosystem-16a86142eb18

---

## Metal FlashAttention (MFA)

**Repo**: github.com/philipturner/metal-flash-attention

### Algorithm
Port of FlashAttention to Metal Shading Language (MSL). Single-headed attention only (multi-head requires calling per head or extending).

### Tiling Strategy
Three-dimensional blocking:
- Block along N (sequence dimension)
- Block along D (head dimension) — novel vs CUDA FA
- D-blocking warps the aspect ratio of attention matrix blocks to minimize register spilling bandwidth cost
- Example block shapes: "16-32 along parallelization dimension, 80-128 along traversal dimension"
- Dynamic parameter file selects block sizes based on head dimension D

### Register Handling
At large head dimensions (D=256): intentional register spilling but optimized — better than naive spilling by careful ordering of loads/stores.

### Apple Silicon-Specific Constraints
- **No native FP32 atomics**: `metal::atomic<float>` is emulated. MFA avoids by splitting backward into two kernels (dQ and dK/dV), achieving "100% parallelization efficiency across both row and column dimensions."
- **simdgroup_async_copy**: Uses undocumented hardware feature (since A14) to overlap compute and load instructions. This is critical for hiding memory latency.
- **BFloat16 emulation**: For older devices without native BF16.

### Performance Numbers (M1 Max)
- Forward pass: **86% ALU utilization** (D=64-256)
- Forward + backward: 62-64% ALU utilization
- Peak: 4400 gigainstructions/second (83% ALU utilization)
- M3/M4: 82-94% ALU utilization (forward), 61-71% (forward+backward)
- Compared to CoreML GPU SD: **20-40% faster** on M1 Pro / M2 Pro and above
- vs ggml implementations: **up to 94% faster**
- vs DiffusionKit: **up to 163% faster**

### Computational Work (per call)
- Forward: (2D + 5) * N² operations
- Backward dQ: (3D + 5) * N²
- Backward dK/dV: (4D + 5) * N²

### Block-Sparse Support
Automatically detects attention mask sparsity. Single shader handles sparse, causal, or irregular masks dynamically. Key for causal masking without separate kernels.

---

## Metal FlashAttention 2.0 (Draw Things, Sep 2024)

**Improvements over MFA 1.0**:
- Both forward (inference) and backward (training) passes
- Runtime code generation for better compiler compatibility
- Better tuning for large head dimensions
- Reduced FP16 inference NaN errors via careful memory/register precision choices
- Up to 20% faster on M3/M4/A17 Pro hardware
- Up to 19% faster backward vs naive implementations

---

## MLX Custom Metal Kernels API

**Source**: ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html

### API
```python
kernel = mx.fast.metal_kernel(
    name="my_kernel",
    input_names=["a", "b"],
    output_names=["out"],
    source="""
    uint elem = thread_position_in_grid.x;
    out[elem] = a[elem] + b[elem];
    """,
    template=[("T", mx.float32)],
)
result = kernel(inputs=[a, b], grid=(N,), threadgroup=(256,),
                output_shapes=[(N,)], output_dtypes=[mx.float32])
```

### Auto-generated signature includes
- Shape/stride metadata for each input: `a_shape`, `a_strides`, `a_ndim`
- Metal attributes: `[[thread_position_in_grid]]`, `[[thread_index_in_simdgroup]]`, etc.
- Template instantiations

### Performance patterns
- SIMD reductions: `simd_sum()` much faster than atomics for aggregation
- 40x speedup reported for grid sample backward using simdgroup reductions
- 8x speedup for fused forward grid sample
- Strided access: use `elem_to_loc()` from `mlx/backend/metal/kernels/utils.h`

### Limitations
- No documentation on tiled matmul examples
- Output arrays always row-contiguous
- Each threadgroup dimension must be ≤ corresponding grid dimension

### Integration with computation graph
- `@mx.custom_function` + `.vjp` decorator for differentiable custom ops
- Kernels built once (JIT compiled), invoked multiple times

---

## MLX Int4 Matmul Kernels (qmm_t / qmm_n)

**Source**: github.com/ml-explore/mlx/pull/2078 and MLX source

### Implementation approach
Two kernel variants:
- `qmm_t`: Transposed B (weights) — typical for linear layers (B is weight matrix, stored transposed)
- `qmm_n`: Non-transposed
- `gather_qmm`: Indexed access for MoE-style routing

### Kernel operations (per group of 64 weights)
1. Load packed 4-bit values (8 weights per byte)
2. Load scale and bias for the group
3. Unpack: bit-shifting and masking
4. Dequantize: `w_hat = scale * quantized_val + bias`
5. Accumulate dot products in fp16/bf16

### Tiling approach
The `gather_qmm` batched kernel includes optimization: when `sorted_indices=True` and consecutive indices are identical, batches operations to reduce redundant weight loads.

---

## llama.cpp Metal Kernels

**Source**: github.com/ggml-org/llama.cpp, Metal backend (`ggml-metal.metal`)

### Approach vs MLX
- llama.cpp implements its own quantization formats (Q4_0, Q4_1, Q4_K_M, Q8_0, etc.)
- Metal kernels: `kernel_get_rows_q4_0`, `kernel_mul_mat_q4_0_f32`, etc.
- Template-based dequantization kernels; `N_R0` = rows per SIMD group, `N_SG` = SIMD groups per threadgroup
- Arguments via packed structs to minimize kernel call overhead
- macOS 15+ / iOS 18+: uses residency sets to keep GPU memory wired (prevents eviction)

### Performance (M2 Max, 7B models)
- Q4_0: **~63 t/s** (vs ~128 t/s on RTX-4080 — 2x slower)
- Q4_K_M: significantly slower than Q4_0 on Apple Silicon specifically because K-quants use lookup tables (codebooks) → "Apple Silicon doesn't like that" — many LUT loads pattern is cache-unfriendly

### Key difference from MLX
- llama.cpp Metal uses K-quant LUT-based formats (Q4_K, Q6_K) which are slower on Apple Silicon Metal than on CUDA
- MLX uses simple scale+bias per group (W4A16 style) which is better suited to Apple Silicon's memory architecture
- MLX achieves ~230 t/s vs llama.cpp ~150 t/s for similar models on the same hardware

---

## Prefill vs Decode Attention Kernels

**Source**: FlashInfer paper (arxiv:2501.01005), various

### Why separate kernels matter
- **Prefill**: All input tokens processed together. Compute-bound (GEMM over full sequence). Benefits from high arithmetic intensity, large tiles.
- **Decode**: Single token attending to full KV cache. Memory-bound (loading KV cache). Benefits from efficient memory access, vectorized loads.

### FlashInfer approach (CUDA, transferable concepts)
- Separate prefill kernel: large tiles, high occupancy
- Decode kernel: Split-K trick on sequence dimension to increase parallelism
- Result: 2.15x attention kernel efficiency improvement, 1.97x end-to-end throughput

### For Metal/MLX
- `mx.fast.scaled_dot_product_attention` (current mllm usage) handles both phases but may not be optimal for decode
- Custom decode kernel could: vectorize KV cache loading (batch 8 heads, load contiguously), use SIMD reductions for softmax per row, avoid materializing full attention score matrix
- Prefill: MFA-style tiled attention (already implemented by MFA)
- Decode: different kernel — loading single-row of attention matrix, attention to potentially thousands of cached KV pairs

---

## Recommendations for mllm Metal Kernel Strategy

### Priority 1: Tiled int4 matmul
Current MLX `quantized_matmul` is already well-optimized. Custom tiled int4 matmul only needed if specific layout or fusion opportunity exists (e.g., fuse dequant + matmul + activation in single pass). The fused SwiGLU kernel is an example of this approach.

### Priority 2: Decode attention kernel
For decode phase: custom Metal kernel that loads KV cache in vectorized chunks, does SIMD softmax across the Q*K products, then accumulates V. Can fuse causal mask check (always true for decode — all positions are valid). This avoids full attention matrix materialization.

### Priority 3: Prefill attention
Use MFA-style tiled attention for prefill. `mx.fast.scaled_dot_product_attention` already uses Metal under the hood — measure if MFA gives additional benefit for the 768-dim, 64-dim-per-head configuration.

### Key Metal-specific optimizations to use
- `simd_sum()` for reductions (not atomics)
- `simdgroup_async_copy` for overlapping compute and load (advanced)
- Residency sets (macOS 15+) to keep model weights pinned in GPU memory
- Template specialization on group_size, bits for compile-time optimization
