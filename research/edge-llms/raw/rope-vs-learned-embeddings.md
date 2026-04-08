# RoPE vs Learned Positional Embeddings

## Sources
- RoPE paper (RoFormer): https://arxiv.org/abs/2104.09864
- EleutherAI RoPE blog: https://blog.eleuther.ai/rotary-embeddings/
- SmolLM2 paper: https://arxiv.org/html/2502.02737v1
- RoPE phase modulation paper: https://arxiv.org/html/2602.10959

---

## What Is RoPE

Rotary Position Embedding (RoPE) encodes position by rotating the query and key vectors in complex space. Position m multiplies Q[m] by rotation matrix R_m (block-diagonal complex rotations).

The dot product Q[m]·K[n] naturally depends on (m-n) only — gives relative position information without explicit relative position matrices.

### Key properties
- **No learned parameters**: position is injected via fixed rotation matrices
- **Inherently relative**: attention score depends on relative position m-n
- **Extrapolation**: can extend to longer sequences than trained on (with tricks like YaRN, RoPE scaling)
- **De facto standard**: Llama 2/3, Mistral, SmolLM2, Qwen, Gemma all use RoPE

---

## Compute Overhead

### Raw cost (from EleutherAI blog)
- RoPE applied to tensors of shape [2048, 16, 12, 64]: **5.3ms**
- Additive positional embeddings: **2.1ms**
- Raw overhead: **~2.5x** at the operation level

### Real-world overhead
- With compiler fusion: 2-2.5x
- As fraction of total model compute: **1-3%** (matrix multiplications dominate)
- At 117M scale: linear layers dominate, RoPE overhead is negligible in practice

### Inference pattern
- Learned embeddings: one lookup table access per token (trivially cheap)
- RoPE: apply sin/cos rotation to Q and K before attention (two operations per layer)
- Net effect on inference: essentially the same — both are bandwidth-limited at model scale

---

## Quality Comparison

### 125M parameter models (OpenWebText2, from EleutherAI)
| Method | Loss |
|--------|------|
| Learned Absolute | 2.809 |
| T5 Relative (RPE) | 2.801 |
| RoPE | **2.759** |

RoPE improves loss by **1.8%** over T5 RPE, **1.8%** over learned absolute at this scale.

### 1.4B parameter models (The Pile)
| Method | Loss |
|--------|------|
| Learned Absolute | 2.240 |
| T5 RPE | 2.223 |
| RoPE | **2.173** |

RoPE improves by **2.2%** over T5 RPE, **3%** over learned absolute at 1.4B scale.

### Convergence speed
RoPE shows **~30% faster convergence** vs learned absolute positional embeddings. This is a training efficiency win.

---

## RoPE vs Learned Absolute for mllm

**mllm currently uses learned positional embeddings.**

Arguments for switching to RoPE:
1. **Quality**: ~1.8% loss improvement at 125M scale — meaningful
2. **Context extrapolation**: can extend beyond block_size=1024 at inference time with RoPE scaling (ALiBi/YaRN tricks). Learned embeddings cannot extrapolate at all.
3. **Industry standard**: every modern small model (SmolLM2, Gemma, Phi) uses RoPE. Interoperability and weight loading will be easier.
4. **No memory cost**: saves the positional embedding table (vocab_size is 50257, pos table is block_size=1024 × 768 = 786K parameters → saved)
5. **Faster convergence during training**: 30% fewer steps to converge

Arguments against:
1. **Requires re-training from scratch**: changing PE scheme invalidates current checkpoints
2. **Slight complexity**: need to compute sin/cos rotations, apply to Q and K — but MLX has mx.fast.rope or similar

**Concrete implementation in MLX**:
```python
# RoPE in MLX
import mlx.core as mx

def rope(x, offset=0):
    # x: [B, T, n_heads, head_dim]
    B, T, H, D = x.shape
    positions = mx.arange(offset, offset + T)
    theta = 1.0 / (10000 ** (mx.arange(0, D, 2) / D))
    freqs = mx.outer(positions, theta)  # [T, D/2]
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)
    # Apply rotation: x1*cos - x2*sin, x1*sin + x2*cos
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return mx.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)
```

MLX also has `mx.fast.rope()` which is a Metal-optimized fused kernel.

---

## RoPE Scaling (Context Extension)

SmolLM2's context was extended from 2048 → 8000 tokens by adjusting RoPE theta (base=10000 → 130000) during continued pretraining. This is impossible with learned absolute embeddings.

For mllm: if targeting longer context in future (e.g., for RAG or document QA), RoPE + scaling is the only viable path without full retraining.

---

## Recommendation

Switch to RoPE in the next training run. The 1-3% inference overhead is negligible; the quality improvement at 125M scale is real (~1.8% better loss), convergence is faster (saves training compute), and context extendability is critical for future work. Use `mx.fast.rope()` in the MLX inference path for zero-overhead fused kernel.
