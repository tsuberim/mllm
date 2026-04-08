# Model Architecture

GPT-style decoder-only transformer. Design choices favour inference efficiency over raw benchmark scores.

## Configs

| Name | Params | n_embd | n_head | n_layer | block_size | Use |
|------|--------|--------|--------|---------|------------|-----|
| tiny | ~1.6M | 32 | 2 | 2 | 64 | Pipeline sanity checks only |
| medium | ~21M | 256 | 8 | 8 | 512 | Local training experiments |
| base | ~117M | 768 | 12 | 12 | 1024 | Production target |

Param counts are dominated by the vocabulary embedding (50257 × n_embd).

## Key Choices

**RMSNorm** over LayerNorm — simpler, slightly faster, no mean subtraction.

**SwiGLU MLP** — three linear projections (w1, w2, w3) with gated activation:
```
out = w3(silu(w1(x)) * w2(x))
```
Better loss/param than GELU at the same parameter count. Hidden dim rounded up to nearest multiple of 64.

**Weight tying** — `tok_emb.weight` and `head.weight` are the same tensor. Saves ~vocab_size × n_embd params (~39M for base).

**Learned positional embeddings** — simple, good enough for now. RoPE is the next upgrade (better extrapolation, no learned parameters).

**No bias** on any linear layer — standard modern practice, marginal speed benefit.

**Pre-norm** (norm before attention/MLP, not after) — more stable training.
