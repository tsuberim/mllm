# Conclusions

## Current State

625 TPS at int4 on M4 (120 GB/s). Theoretical ceiling: ~2051 TPS (weights only), ~1551 TPS (with KV at T=512).

**We're at ~30% of ceiling.** Two bugs account for most of the gap — not missing optimizations.

---

## High Priority: Fix These First

### 1. Switch to `mx.fast.scaled_dot_product_attention`

Current `infer.py` computes attention manually:
```python
scores = (q @ k.swapaxes(-2, -1)) * self.scale
scores = mx.where(mask, scores, mx.full(..., float("-inf")))
y = (mx.softmax(scores, axis=-1) @ v)
```

This allocates a full `[B, H, T, T]` score matrix and falls through to MLX's generic fallback. It completely bypasses `sdpa_vector` — MLX's optimized decode kernel that uses online softmax with no intermediate materialization.

Replace with:
```python
y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
# Returns [B, H, T, D] directly
```

MLX selects `sdpa_vector` when `T <= 8` and `head_dim in {64, 80, 128}` — our decode path hits both.

### 2. Pre-allocate KV Cache

Current:
```python
k = mx.concatenate([past_k, k], axis=2)
v = mx.concatenate([past_v, v], axis=2)
```

Every decode step allocates a new tensor and copies the full cache. At T=512, 12 layers: **~18.87 MB copied per token**. At 625 TPS that's 11.8 GB/s — ~10% of the M4's 120 GB/s budget wasted on bookkeeping.

Fix: pre-allocate `[1, H, max_T, D]` at model load time; use slice assignment per step.

---

## Secondary (Diminishing Returns)

| Optimization | Est. Gain | Complexity |
|---|---|---|
| `half2` vectorized SwiGLU | <5% | Low |
| Fused RMSNorm + QKV matmul | 5–10% | Medium |
| Larger GEMM tiles (prefill) | Prefill only | Medium |
| Multi-token decode | Latency tradeoff | High |

LUT dequantization and tiled int4 matmul are **not worth implementing** — MLX's bitmask+prescaled-activation pattern is already near-optimal for Apple Silicon.

---

## iPhone (A18): Hardware Ceiling, Not an Optimization Problem

- Memory bandwidth: 60 GB/s (vs M4's 120 GB/s)
- RAM: 8 GB; safe model budget: ≤4 GB total app footprint
- No MLX on iOS (requires MPSGraph + JIT Metal compilation, both blocked)
- Path: PyTorch → CoreML → stateful KV cache → 4-bit palettization

**Expected TPS for 117M at int4 palettization: 20–50 TPS.** This is the bandwidth ceiling, not a bug. The model fits comfortably (~60 MB).

For iPhone, getting ANE utilization right (Conv2d substitution, RoPE restructure, iOS 18 fused SDPA) matters more than kernel tricks — the ANE at 6.6 TFLOPS/W is 5× more power-efficient than the GPU for prefill.

---

## Metal Architecture: What Matters for This Project

- All Apple Silicon GPUs use **32-thread SIMD groups** (Apple7/8/9 — M1 through A18)
- Threadgroup memory: **32 KB** on all generations
- **M3+ Dynamic Caching:** register allocation is dynamic — register pressure matters less for occupancy than on M1/M2
- M4 and A18 are both Apple9 Metal family — **shaders port directly between Mac and iPhone target**
- M5 (A19) adds dedicated GPU matrix multiply units accessible via Metal Performance Primitives — worth tracking when targeting iPhone 17/M5 Macs

## Roofline

Every transformer op at decode (batch=1) is **memory-bandwidth-bound**, not compute-bound. The ridge point (~35 FLOP/byte) is far above any decode operation's arithmetic intensity. Token throughput scales linearly with memory bandwidth. This is why M4 Max (546 GB/s) dominates for LLM serving, and why int4 gives ~8× over FP16 — not because of compute savings, but because it reads 8× less data.
