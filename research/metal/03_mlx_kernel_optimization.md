# MLX Metal Kernel Optimization for Transformer Inference

Research specific to this project: GPT-style transformer, 117M params, Apple Silicon inference via MLX.

---

## MLX `metal_kernel` API

```python
kernel = mx.fast.metal_kernel(
    name="swiglu",
    input_names=["gate", "up"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        T g = gate[elem];
        T u = up[elem];
        T sig = T(1) / (T(1) + metal::exp(-g));
        out[elem] = g * sig * u;
    """,
)
result = kernel(
    inputs=[gate, up],
    template=[("T", gate.dtype)],
    grid=(size, 1, 1),
    threadgroup=(min(256, size), 1, 1),
    output_shapes=[gate.shape],
    output_dtypes=[gate.dtype],
)[0]
```

**Key parameters:**
- `template`: `(name, value)` tuples — each unique combo JIT-compiles a distinct Metal library. `value` can be `mx.Dtype`, `int`, or `bool`.
- `grid`: total threads launched (not threadgroups). Maps to `dispatchThreads`.
- `threadgroup`: per-threadgroup size. MLX clamps each dim to corresponding grid dim.
- `init_value`: pre-fills all outputs before kernel runs (useful for reductions).
- `atomic_outputs=True`: wraps outputs as `device atomic<T>*`.
- `ensure_row_contiguous=True` (default): copies non-contiguous inputs automatically.
- `verbose=True`: prints full generated source at dispatch.

**Auto-injected into signature:** For input named `a`, if `a_shape`, `a_strides`, or `a_ndim` appear in source, they're added as `constant int*`, `constant int64_t*`, `constant uint&` respectively.

**Limitation:** No control over occupancy hints, `max_total_threads_per_threadgroup`, or threadgroup memory size from Python. These require raw `.metal` files via `[[kernel, max_total_threads_per_threadgroup(N)]]`.

**Build kernels once at module level.** Each `metal_kernel()` call creates a new MTLLibrary (slow). The returned callable can be called many times cheaply.

---

## Tiled GEMM on Metal (Steel GEMM Library)

MLX uses its Steel GEMM library (`mlx/backend/metal/kernels/steel/gemm/`) for large matrix multiplies (prefill).

**Template parameters:**
```
BM, BN, BK  — threadgroup tile dimensions in M, N, K
WM, WN      — SIMD-group tile dimensions (each 8×8 fragment)
threadgroup_size = WM × WN × 32
```

Typical values: `BM=BN=BK=32`, `WM=WN=2`. Threadgroup = 2×2×32 = 128 threads.

**Threadgroup memory (with bank-conflict padding):**
```
tgp_mem_a = BM × (BK + 16/sizeof(T))
tgp_mem_b = BK × (BN + 16/sizeof(T))
```

**Inner loop pattern:**
```metal
for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_unsafe();   // loads tile from device → threadgroup
    loader_b.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);       // simdgroup_multiply_accumulate on 8×8 tiles
}
```

**MMA:**
```metal
simdgroup_multiply_accumulate(D, A, B, C);  // D = A × B + C
```

Each SIMD-group holds 2 elements of an 8×8 matrix across 32 SIMD lanes. Accumulation uses serpentine N-ordering for cache efficiency.

**Quantized GEMM:** `QuantizedBlockLoader` uses the same `BM=BK=BN=32` geometry but handles int4 unpacking *during* the load phase into threadgroup memory, before the MMA.

For M3+ with `K%64==0`: MLX selects `qmm_t_nax` which uses NAX (Neural Acceleration eXtensions) optimizations.

---

## Attention Kernels: Decode vs Prefill Split

MLX selects different attention kernels based on query sequence length:

```cpp
// From ScaledDotProductAttention::use_fallback():
const bool supports_sdpa_vector = (query_sequence_length <= 8) &&
    (query_head_dim == 64 || query_head_dim == 80 || query_head_dim == 128) &&
    (query_sequence_length * gqa_factor) <= 32;

const bool supports_sdpa_full = query_sequence_length > 8 && ...;
```

**This project's dispatch:** head_dim=64, decode has T=1 → hits `sdpa_vector` path. Prefill (T>8) → uses Steel attention (tiled FlashAttention).

### Decode Path: `sdpa_vector` Kernel

```
BN=32 keys per iteration, BD=32 threads per dimension
qk_per_thread = head_dim / BD = 64 / 32 = 2 elements per thread
```

Each SIMD-group of 32 threads collectively holds one full query head (64 elements). Per iteration:
1. Load 32 keys
2. Compute 32 dot products using `simd_sum(score)`
3. Update online softmax: running `max_score` and `sum_exp`
4. Accumulate into output: `o[i] = o[i] * factor + exp_score * v[i]`

Cross-SIMD-group reduction via threadgroup memory + `simd_max` / `simd_sum`.

**2-pass split (`sdpa_vector_2pass`):** For long KV sequences (>1024–4096 tokens). Pass 1 partitions K into `blocks` groups (64/128/256 depending on chip suffix), emits partial `out/sums/maxs`. Pass 2 aggregates. The `blocks` count varies: `s` suffix = M1/M2, `d` suffix = M3+.

### Prefill Path: Steel Attention (`steel_attention`)

Tiles `BQ=64` (or 32) query tokens, `BK=32` key tokens. Loads Q into threadgroup memory once per block. Loops over K blocks with online softmax. Uses `simdgroup_matrix` MMA for `S = Q @ K^T` and `O += softmax(S) @ V`. `ExpSubOp` applies `exp2(x - max)` using log2(e) prescaling for `fast::exp2`.

**Key difference from CUDA FlashAttention:** Metal uses `simdgroup_matrix` instead of CUDA `wmma`. No persistent thread blocks or async memory prefetch — MLX uses the 2-pass split to compensate. `simdgroup_barrier(mem_flags::mem_none)` between MMA ops (weaker than threadgroup barrier, just ensures instruction ordering within SIMD-group).

### Current Code vs Optimal

**Current `infer.py` attention** computes `q @ k.swapaxes(-2, -1)` manually then softmax. This:
1. Does NOT call `mx.fast.scaled_dot_product_attention`
2. Allocates full `[B, H, T, T]` score matrix → falls through to fallback path
3. Loses the decode-optimized `sdpa_vector` kernel entirely

**Fix:** Replace manual `q @ k^T + softmax` with `mx.fast.scaled_dot_product_attention`. This is likely the single highest-impact optimization available.

---

## Fused SwiGLU: Theory vs Practice

### Theoretical Analysis

Current implementation (`infer.py` is already fused — correct). For completeness:

**Unfused path:** `silu(gate)` (read gate, write temp) + `temp * up` (read temp+up, write out) = 3 reads + 2 writes + 2 kernels.

**Fused path:** Read gate+up, write out = 2 reads + 1 write + 1 kernel.

**Bandwidth savings for base model (T=1 decode):** gate/up tensors are `[1, 3072]` FP16 = 6 KB each.
- Unfused: 24 KB memory traffic
- Fused: 18 KB memory traffic
- Theoretical ceiling: 1.33×

**Actual impact at T=1 decode:** Modest. The dominant bandwidth cost is weight matrices (~58.5 MB per token for 117M int4). The fused SwiGLU is correct and eliminates kernel dispatch overhead, but won't meaningfully change TPS.

### half2 Vectorized SwiGLU (improvement)

Current kernel uses scalar `T` per element. Can process 2 elements per instruction:

```metal
// Current:
uint elem = thread_position_in_grid.x;
T g = gate[elem]; T u = up[elem];
T sig = T(1) / (T(1) + metal::exp(-g));
out[elem] = g * sig * u;

// Vectorized half2 (when dtype == float16):
uint elem = thread_position_in_grid.x;
half2 g2 = ((device half2*)gate)[elem];
half2 u2 = ((device half2*)up)[elem];
half2 sig2 = half2(1) / (half2(1) + metal::exp(-g2));
((device half2*)out)[elem] = g2 * sig2 * u2;
// Grid: (size/2, 1, 1) instead of (size, 1, 1)
```

Halves the number of loads/stores and exp() calls.

---

## LUT Dequantization: Reality Check

**Theory:** Precompute 16-entry LUT `lut[i] = float(i) * scale + bias` per group. Unpack nibble → indexed load. Avoids multiply+add per weight.

**In practice:** Neither MLX nor llama.cpp uses LUT dequantization. Both use the bitmask + prescaled-activation approach (see architecture doc §10). Reasons:

1. MLX folds divisors into the prescaled activation vector so zero shifts are needed at dot-product time
2. A LUT would require: unpack nibble → register table load → multiply — same or more instructions
3. Accumulation happens in float32 regardless; the quantized extraction is not the bottleneck
4. Modern compilers vectorize the bitmask pattern effectively

**Verdict:** LUT dequantization is mentioned in llama.cpp issues but not the current approach. Don't implement it.

---

## Quantization-Aware Matmul: Full MSL Pattern

### MLX Quantization Format

- int4: 8 values per `uint32`. Layout: `v0 | (v1<<4) | ... | (v7<<28)`
- Per-group scale + bias (zero-point): one `float` each, per 64 weights
- `group_size=64` (MLX default for `mlx.nn.quantize`)

### Prescaled Dot Product (no shifts needed)

```metal
// load_vector: prescale x to fold in nibble-position divisors
for (int i = 0; i < values_per_thread; i += 4) {
    x_thread[i]   = x[i];
    x_thread[i+1] = x[i+1] / 16.0f;    // compensates for (w & 0x00f0) not shifting
    x_thread[i+2] = x[i+2] / 256.0f;
    x_thread[i+3] = x[i+3] / 4096.0f;
}

// Dot product (reinterpret packed bytes as uint16 — 4 nibbles per uint16)
const device uint16_t* ws = (const device uint16_t*)w;
for (int i = 0; i < (values_per_thread / 4); i++) {
    accum += (x_thread[4*i]   * (ws[i] & 0x000f) +
              x_thread[4*i+1] * (ws[i] & 0x00f0) +
              x_thread[4*i+2] * (ws[i] & 0x0f00) +
              x_thread[4*i+3] * (ws[i] & 0xf000));
}
return scale * accum + sum * bias;
```

`sum` is `simd_sum(x_thread)` — precomputed to efficiently apply zero-point: `bias * sum(x)`.

### `qmv_fast` (Decode Path, Batch=1)

Threadgroup organization:
- `num_simdgroups = 2`, `results_per_simdgroup = 4` → 8 output rows per threadgroup
- Each simdgroup lane: `values_per_thread = 8` (for 4-bit, `pack_factor=8`)
- `block_size = values_per_thread × SIMD_SIZE = 8 × 32 = 256` elements per iteration
- Scale advances every `group_size / values_per_thread = 64/8 = 8` threads

### `qmv_quad` (Quadgroup Variant)

Uses 4-thread quads instead of 32-thread SIMD-groups. `simd_sum()` → `quad_sum()`. Better for smaller K dimensions.

### Kernel Selection (MLX dispatch logic)

| Condition | Kernel |
|-----------|--------|
| M < 4 | `qmv` / `qmv_fast` (matrix-vector) |
| K < 1024 | `qvm` with 64 threads/block |
| K ≥ 1024 | `qvm_split_k` (8 or 32 threadgroups) |
| M ≥ threshold | `qmm` (32×32×32 tiled GEMM) |
| M3+ and K%64==0 | `qmm_t_nax` (NAX optimizations) |

---

## KV Cache Memory Patterns

### Current Layout: `[B, H, T, D]` = `[1, 12, T, 64]`

Strides: `[H×T×D, T×D, D, 1]` = `[12×T×64, T×64, 64, 1]`

`sdpa_vector` access:
```metal
keys += kv_head_idx * k_head_stride + simd_gid * k_seq_stride + simd_lid * qk_per_thread;
```
- `simd_gid` advances by token (stride=64 elems)
- 32 lanes load 2 elements each = 64 elements = one full token
- **Coalesced**: 32 lanes load consecutive 128-byte cache line

### The `mx.concatenate` Problem

Current code: `k = mx.concatenate([past_k, k], axis=2)` every decode step.

This **allocates a new tensor and copies the full KV cache** each step. At T=512 context, 12 layers × 2 tensors × `[1, 12, 512, 64]` FP16 = 18.87 MB copied per token.

At 625 TPS: 18.87 MB × 625 = **11.8 GB/s** on KV cache copies alone (~10% of the ~120 GB/s M4 budget).

**Fix:** Pre-allocate `[1, H, max_T, D]` and use in-place slice assignment. MLX's `mlx-lm` framework does this. Priority: high.

---

## half vs float in Metal Kernels

**MLX approach:** Accumulation always in `float` (`AccumType = float` in quantized kernels and attention). Storage and I/O in `T` (half/bfloat16). Matches CUDA best practice.

**Vectorized loads:** MLX uses scalar loops (vectorization comes from 32-thread SIMD group structure). `rms_norm.metal` uses `N_READS=4` scalar reads per loop iteration (compiler vectorizes to 4-element unroll).

**When to use `half2`:** For elementwise kernels (SwiGLU, activations) where you want explicit 2-wide packing without compiler reliance. Half loads 2 values per 32-bit load. See the half2 SwiGLU pattern above.

**`bfloat16_t` in Metal:** Supported natively on M-series. MLX uses it throughout. Defined in `mlx/backend/metal/kernels/bf16.h` (auto-included via `utils.h`).

---

## Current Benchmarks vs Theoretical Ceiling (M4, 120 GB/s)

| Setup | Measured TPS | Ceiling TPS | Utilization |
|-------|-------------|-------------|-------------|
| fp32, no KV cache | 38.7 | — | — |
| fp32 + KV cache | 242.6 | — | — |
| int4 + KV cache | 625.3 | ~2051 | ~30% |

~70% of theoretical bandwidth is unaccounted for. Main suspects:

1. **Manual attention (`q @ k^T`)** instead of `mx.fast.scaled_dot_product_attention` — falls through to generic fallback, not `sdpa_vector`
2. **`mx.concatenate` KV cache** — 10% of bandwidth budget on copies at T=512
3. **Kernel launch overhead** — many small ops at T=1 decode; launch latency not amortized
4. **Low occupancy** from register pressure on M1/M2 code paths

---

## Planned Optimizations (Priority Order)

### 1. Switch to `mx.fast.scaled_dot_product_attention` (highest impact)

Replace manual attention with the MLX fast path. At T=1 decode, this enables `sdpa_vector` kernel (online softmax, no `[B,H,T,T]` allocation).

```python
# Current:
scores = (q @ k.swapaxes(-2, -1)) * self.scale
if T > 1:
    mask = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    scores = mx.where(mask, scores, mx.full(scores.shape, float("-inf")))
y = (mx.softmax(scores, axis=-1) @ v)

# Replace with:
y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
# Returns [B, H, T, D] directly
```

### 2. Pre-allocated KV Cache

Eliminate per-step `mx.concatenate`. Allocate `[1, H, max_T, D]` buffer at model load time; use slice assignment each step.

### 3. Tiled int4 Matmul (Prefill)

The current `qmm` path uses 32×32×32 tiles. Larger tiles (64×64×64) improve GEMM efficiency for prefill. Profile first to confirm this is a bottleneck before implementing.

### 4. Separate Prefill/Decode Kernels

Prefill: compute-bound → needs tiled GEMM with `simdgroup_matrix`. Decode: memory-bound → needs `qmv_fast` with minimal overhead. Already partially handled by MLX's kernel selection; explicit user-side bifurcation might give additional gains.

### 5. half2 Vectorized SwiGLU

See pattern above. Free and clean to implement.

### 6. Fused RMSNorm + QKV Matmul

Avoids writing norm output to DRAM before reading it back for the matmul. One kernel launch, one device memory read for x. Non-trivial to implement correctly with quantized weights.

---

## llama.cpp Metal Kernel Patterns (Reference)

**Q4_0 dequantization** (`ggml-metal.metal`):
```metal
template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    // unpacks nibbles, applies scale/bias per nibble position
}
```

Same bitmask+prescaled-divisor pattern as MLX. Symmetric quantization (zero-point = -8).

**SwiGLU variants in ggml:**
```metal
kernel void kernel_swiglu_f32(...) {
    const float silu = x0 / (1.0f + exp(-x0));
    dst_row[i0] = silu * x1;
}
// Also: kernel_geglu_f32, kernel_reglu_f32, kernel_swiglu_oai_f32 (with clipping)
```

**Key difference from MLX:** llama.cpp uses scalar loops + simdgroup reductions for GEMV; no `simdgroup_matrix`. MLX uses Steel GEMM with `simdgroup_multiply_accumulate` for prefill — substantially faster for matrix-matrix ops.

---

## Sources

- [MLX Custom Metal Kernels Documentation](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [MLX fast.metal_kernel API](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html)
- [MLX source: mlx/backend/metal/kernels/steel/gemm/gemm.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/gemm.h)
- [MLX source: mlx/backend/metal/kernels/sdpa_vector.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h)
- [MLX source: mlx/backend/metal/kernels/quantized.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h)
- [MLX source: mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h](https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h)
- [MLX use_optimal_threadgroups PR #1833](https://github.com/ml-explore/mlx/pull/1833)
- [llama.cpp ggml-metal.metal](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-metal.metal)
- [ThunderMittens for Apple Metal (Hazy Research)](https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx)
- [Apple G13 GPU Architecture (dougallj)](https://dougallj.github.io/applegpu/docs.html)
- [Scale Compute Workloads Across Apple GPUs — WWDC22](https://developer.apple.com/videos/play/wwdc2022/10159/)
