# LUT Dequantization vs Multiply-Add: Int4 Matmul Performance

## Sources
- LUT-GEMM paper: https://arxiv.org/abs/2206.09557 (ICLR 2024)
- FLUTE fast LUT matmul: https://aclanthology.org/2024.findings-emnlp.724/
- LUT Tensor Core ISCA 2025: https://dl.acm.org/doi/10.1145/3695053.3731057
- T-MAC (Microsoft, LUT-based, no multiply): https://microsoft.com
- MLX quantized matmul: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
- llama.cpp K-quant performance: https://github.com/ggml-org/llama.cpp/discussions/5617

---

## Two Approaches to Int4 Matrix Multiplication

### Approach 1: Dequantize-then-Multiply (W4A16)
1. Load packed 4-bit weights
2. Unpack and dequantize to fp16: `w_hat = scale * q + zero`
3. Multiply dequantized weights by fp16 activations
4. Accumulate in fp16/fp32

**Pros**: Hardware-friendly (FP16 multiply-add uses existing MAC units), simple, well-suited to GPU tensor cores
**Cons**: Dequantization step is overhead; doubles memory reads (need scale/bias too)

### Approach 2: LUT-based (no multiply)
1. Precompute lookup table: for each possible activation value and 4-bit weight value, compute their product
2. At runtime: index into LUT by (activation_quantized, weight_4bit) pair
3. Accumulate LUT results

**Pros**: Eliminates multiply — replaces with table lookup (faster on some architectures)
**Cons**: LUT must be loaded into cache; random access pattern; works best when activations are also quantized

---

## LUT-GEMM (ICLR 2024)

**Paper**: arxiv:2206.09557

Quantized matrix multiplication using lookup tables for efficient inference in large-scale generative language models.

**Key results**:
- **2.1x speedup** vs OPTQ (dequantization-based) on OPT-175B with 3-bit quantization
- Works with group-wise quantization for flexible compression/accuracy tradeoff
- Eliminates dequantization overhead
- Better at very low bits (3-bit, 2-bit) where multiply-add approach degrades more

**Limitation**: LUT-based approaches are efficient on CUDA but **underperform on Apple Silicon Metal** because:
- LUT lookups = scattered/random memory accesses
- Apple's unified memory has excellent sequential bandwidth but less cache for random access
- This is exactly why llama.cpp K-quant (LUT-based) is slower than Q4_0 (scale+bias) on Apple Silicon

---

## FLUTE Fast LUT Matmul (EMNLP 2024 Findings)

**Paper**: aclanthology.org/2024.findings-emnlp.724/

Structured LUT approach that avoids cache issues by organizing LUT for sequential access.

**Key results**:
- At batch sizes < 32 and group_size=128: **2-4x faster** than existing GEMM kernels
- Underperforms dequantization-based kernels on A100 at larger batch sizes
- Best for inference (batch=1), worse for training (large batch)

**Apple Silicon relevance**: FLUTE's advantages are measured on CUDA. The LUT organization optimized for CUDA's cache hierarchy may not transfer to Metal's different cache architecture.

---

## T-MAC (Microsoft, 2024)

**Source**: Microsoft Research blog on low-bit LLM inference on edge devices

Bit-wise table lookup approach:
- Decomposes weight matrix into bit planes
- Uses bitwise operations + small precomputed tables instead of multiply
- No floating-point multiply at all during matrix computation
- Designed for CPUs with SIMD (ARM NEON, x86 AVX)

**Apple Silicon (CPU) relevance**: T-MAC targets Apple M-series ARM NEON specifically. On the CPU:
- ARM NEON's `VTBL` instruction for table lookups is very fast
- Avoids fp16 multiply entirely
- Reports significant speedups on CPU paths

**vs GPU**: T-MAC is CPU-optimized. The Apple GPU (Metal) has fast fp16 MACs; T-MAC's advantage doesn't transfer to GPU.

---

## MLX Current Approach: Dequantize + FP16 GEMM

MLX uses the dequantize-then-multiply approach (W4A16) with per-group scale + bias. This is the right choice for Apple Silicon GPU because:

1. Metal GPU's FP16 units are fast (simdgroup_matrix operations)
2. Sequential memory access pattern (not random LUT access) is cache-friendly
3. Scale/bias are loaded sequentially per group (64 weights → 1 scale+bias)
4. The overhead of dequantization is absorbed into the matmul pipeline

**Current MLX kernel pipeline** (from source analysis):
```
Load packed 4-bit (8 weights per uint32)
→ Extract bits via shifting and masking
→ Load scale and bias for the group
→ dequant: w_fp16 = scale * (w_int4 - 8) + bias  [symmetric: w_fp16 = scale * w_int4]
→ w_fp16 × a_fp16 (FP16 MAC)
→ Accumulate in fp32 accumulator
```

---

## Tiled Int4 Matmul Design for Metal

For a custom tiled int4 matmul Metal kernel optimized for decode (batch=1):

### Key design constraints
- Batch size = 1 during decode (one token)
- Weight matrix: [hidden_dim, hidden_dim] or [hidden_dim, 4×hidden_dim] for SwiGLU
- Activation vector: [1, hidden_dim]
- This is a GEMV (matrix-vector), not GEMM

### GEMV optimization (decode phase)
For GEMV (batch=1), the optimal approach differs from GEMM:
1. Assign one threadgroup per output row
2. Each thread in threadgroup handles a slice of the input dot product
3. Use SIMD reductions within threadgroup for partial sum accumulation
4. Load weights row by row (sequential), dequantize, multiply with activation

```msl
kernel void qgemv(
    device const uint8_t* weights [[buffer(0)]],   // packed 4-bit
    device const half* scales [[buffer(1)]],
    device const half* activations [[buffer(2)]],
    device half* output [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    float acc = 0.0f;
    for (uint col = tid; col < N; col += tg_size) {
        // Load and dequantize 4-bit weight
        uint group_id = col / group_size;
        half scale = scales[row * (N / group_size) + group_id];
        uint byte = weights[(row * N + col) / 2];
        int nibble = (col % 2 == 0) ? (byte & 0xF) : (byte >> 4);
        half w = scale * half(nibble - 8);
        acc += float(w) * float(activations[col]);
    }
    // SIMD reduce
    acc = simd_sum(acc);
    if (tid == 0) output[row] = half(acc);
}
```

**Tiling for GEMM (prefill phase)**:
Use `simdgroup_matrix` (Metal 3+) for 8×8 FP16 matmul tiles. Requires dequantized weights in FP16 tile format. The overhead of dequantization amortizes over the tile.

---

## Recommendation: Don't Use LUT for Metal GPU

For MLX/Metal GPU path: **keep dequantize-then-multiply (W4A16)**.

The evidence is clear:
1. llama.cpp K-quant (LUT-based) is slower than Q4_0 (scale+bias) on Apple Silicon Metal
2. LUT's advantages are measured on CUDA where random access patterns are better cached
3. FLUTE only wins at batch<32 on CUDA; doesn't translate to Metal

**For CPU inference path**: T-MAC or ARM NEON VTBL-based LUT may be worth exploring, but MLX currently targets GPU.

**Priority**: Better to focus on:
1. GQA (reduces KV cache bandwidth)
2. Fused kernels (reduce memory round-trips)
3. Dedicated decode kernel (GEMV optimized vs GEMM-general)
