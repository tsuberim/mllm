# Conclusions

Key takeaways from the edge LLM inference research, scoped to this project (~117M param GPT, MLX on Apple Silicon).

## Theoretical ceiling

int4 base model ~70 MB. M4 base memory bandwidth: 120 GB/s → **~1,714 TPS theoretical max**.
Current: 625 TPS = **36% efficiency**. Significant headroom without touching the model.

## Ranked recommendations

### Tier 1 — Do immediately (zero model changes)

| Change | Expected gain | Effort |
|--------|--------------|--------|
| `mx.compile()` on inference loop | 10–30% TPS | 1 line |
| Dedicated decode attention GEMV kernel | 30–60% attn throughput | Medium |
| Tiled int4 matmul kernel | Closes gap toward bandwidth ceiling | High |

### Tier 2 — Requires retraining

| Change | Expected gain | Notes |
|--------|--------------|-------|
| GQA (n_kv_heads=4) | 3× smaller KV cache, lower decode bandwidth | MLX `mx.fast.scaled_dot_product_attention` supports GQA natively — just reshape k/v weights |
| RoPE positional embeddings | ~1.8% better val loss at 125M scale, 30% faster convergence | Enables future context extension; combine with GQA in same run |
| Muon optimizer | 30–35% faster convergence, same final loss | ~40-line addition to train.py, negligible compute overhead |

Combine GQA + RoPE + Muon into a single training rerun — no reason to do them separately.

### Tier 3 — CoreML / iPhone path

- Use **stateful KV cache API** (iOS 18+) — gives ~13× speedup over stateless KV export. Essential; don't attempt CoreML without it.
- Export with `ct.ComputeUnit.ALL`. Profile in Xcode Instruments to verify no CPU fallback on attention/matmul ops.
- int4 base model is 188 MB — well within 4 GB iPhone budget.

## Critical counterintuitive finding

**LUT-based dequantization is slower on Apple Silicon than scale+bias int4.**

llama.cpp K-quants and LUT-GEMM use random lookup table access patterns that are cache-unfriendly on Apple's unified memory subsystem. MLX's current scale+bias int4 (`group_size=64`) is already the correct format. **Do not implement LUT dequant.**

## Attention kernel strategy

- Prefill is **compute-bound** (GEMM) — standard tiled matmul, batch tiles across sequence.
- Decode is **memory-bound** (GEMV, Q shape `[1, n_head, head_dim]`) — needs SIMD-group reductions and vectorized KV loads, not GEMM tiling.
- These require **separate kernel implementations**. Using the same kernel for both leaves significant performance on the table.

## What not to pursue

- **Speculative decoding** — at 117M params, a meaningful draft model doesn't exist and self-speculation has limited applicability. Not worth it at this scale.
- **Mamba / linear attention** — quality/efficiency tradeoff not well-characterised at this size; adds major architectural complexity.
- **LUT dequantization** — see above.
- **Training on MPS** — confirmed non-viable: silent CPU fallbacks, no `torch.compile`, no int4, no FlashAttention.

## Open questions

1. How much TPS does `mx.compile()` actually recover for this specific model? (measure before other work)
2. Does GQA with 4 KV heads meaningfully hurt quality at 117M scale? (very few published ablations at this size)
3. What is the CoreML stateful KV cache latency on A17 Pro vs M4? (no published apples-to-apples comparison found)
4. Can MLX's `mx.fast.scaled_dot_product_attention` be replaced with a custom GEMV decode kernel without losing the prefill path? (need to profile the split)
