# Model Architecture

GPT-style decoder-only transformer. Design choices favour inference efficiency over raw benchmark scores.

## Configs

| Name | Params | n_embd | n_head | n_kv_head | n_layer | block_size | Use |
|------|--------|--------|--------|-----------|---------|------------|-----|
| sanity | ~1.6M | 32 | 2 | 2 | 2 | 64 | Pipeline sanity checks only |
| experiment | ~21M | 256 | 8 | 2 | 8 | 512 | Local training experiments |
| 3b | ~3.17B | 3072 | 24 | 8 | 20 | 4096 | 3B target |
| 7b | ~7.19B | 4096 | 32 | 8 | 26 | 4096 | 7B target |

Param counts are dominated by the vocabulary embedding (32016 × n_embd). Weight tying means tok_emb and head share the same tensor — counted once.

## Key Choices

**RMSNorm** over LayerNorm — simpler, slightly faster, no mean subtraction.

**SwiGLU MLP** — three linear projections (w1, w2, w3) with gated activation:
```
out = w3(silu(w1(x)) * w2(x))
```
Better loss/param than GELU at the same parameter count. Hidden dim rounded up to nearest multiple of 64.

**Weight tying** — `tok_emb.weight` and `head.weight` are the same tensor. Saves ~vocab_size × n_embd params (~39M for base).

**RoPE** (Rotary Position Embeddings) — applied to Q and K in each attention layer. Better length extrapolation than learned embeddings; no learned parameters.

**GQA** (Grouped Query Attention) — fewer KV heads than Q heads (4:1 ratio for base). 3× smaller KV cache; MLX sdpa handles GQA natively.

**No bias** on any linear layer — standard modern practice, marginal speed benefit.

**Pre-norm** (norm before attention/MLP, not after) — more stable training.
