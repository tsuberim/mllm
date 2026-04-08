# Muon Optimizer: Training Efficiency for Language Models

## Sources
- Keller Jordan blog: https://kellerjordan.github.io/posts/muon/
- Muon GitHub: https://github.com/KellerJordan/Muon
- PyTorch docs: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html
- Muon scalability report: https://arxiv.org/pdf/2502.16982
- Nubank implementation: https://building.nubank.com/muon-for-improved-foundation-model-pretraining-data-efficiency/

---

## What is Muon

**Muon** = **Mo**mentum **o**rthogonalized by **N**ewton-Sch**u**lz

Introduced by Keller Jordan in 2024. An optimizer for 2D weight matrices in neural networks.

**Core algorithm**:
1. Compute Nesterov momentum gradient G = β * G_prev + grad
2. Apply Newton-Schulz iteration to orthogonalize G: find nearest semi-orthogonal matrix to G
3. Update: W -= lr * orthogonalize(G)

The NS iteration replaces G with U·V^T from its SVD, but computed cheaply via polynomial iteration rather than full SVD.

---

## Newton-Schulz Iteration

Polynomial: φ(X) = aX + bX³ + cX⁵
Coefficients: (a, b, c) = (3.4445, -4.7750, 2.0315) — tuned for convergence to semi-orthogonal matrix

**Why quintic polynomial?**
- Cubic: converges but slowly
- Quintic: 5 iterations sufficient for convergence
- Higher: diminishing returns vs compute overhead

**Implementation**:
```python
def newton_schulz(G, steps=5):
    # Normalize to spectral norm ≈ 1
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + 1e-7)
    if G.shape[0] > G.shape[1]:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * A @ X + c * A @ A @ X  # quintic step
    if G.shape[0] > G.shape[1]:
        X = X.T
    return X.to(G.dtype)
```

Runs in bfloat16 for numerical stability on tensor cores.

---

## Performance Numbers

### NanoGPT speedrun (124M parameters)
- AdamW: baseline
- Muon: **1.35x faster convergence** (same final loss reached in fewer steps)
- ~35% training speed improvement on NanoGPT speedrunning records

### 1.5B parameter LLM (Keller Jordan experiments)
- AdamW: 13.3 hours on 8×H100
- Muon: **10 hours on 8×H100** (1.33x faster)
- Same final validation loss

### General convergence
- Reaches same validation loss in ~2/3 of AdamW steps
- Converges to lower validation losses than AdamW at equal step count
- 10%-30% improvement in data efficiency

---

## Computational Overhead

**FLOP overhead** = T × m / B where:
- T = 5 (NS iteration steps)
- m = parameter dimension (e.g., 768 for mllm)
- B = batch size in tokens

**For mllm** (m=768, typical B=262144 tokens per step at batch_size=256, seq_len=1024):
- Overhead = 5 × 768 / 262144 = **0.015%** — essentially free

This is why Muon is compelling: the orthogonalization adds <0.1% FLOP overhead for typical training batch sizes.

**Memory overhead**: identical to SGD-momentum (just stores momentum buffer, same as Adam's first moment). No extra memory vs AdamW beyond the reduced second moment (Adam maintains two buffers; Muon only needs one).

---

## Implementation Guidelines

### Which parameters get Muon vs AdamW
- **Muon**: all 2D weight matrices in hidden layers (linear layers, attention projections)
- **AdamW**: embedding table, output head/LM head, biases, layer norms, 1D parameters

```python
muon_params = [p for name, p in model.named_parameters() 
               if p.ndim == 2 and 'embed' not in name and 'lm_head' not in name]
adamw_params = [p for name, p in model.named_parameters() 
                if p not in set(muon_params)]

optimizer = torch.optim.Muon(muon_params, lr=0.02, momentum=0.95)
optimizer_head = torch.optim.AdamW(adamw_params, lr=3e-4, betas=(0.9, 0.95))
```

### Transformer-specific: separate Q, K, V
Apply Muon separately to Q, K, V weight matrices rather than combined QKV projection. This is mentioned in the blog post as improving performance.

### Learning rate scaling
Muon's effective learning rate semantics differ from AdamW. Typical range: lr=0.01-0.05 for Muon vs lr=3e-4 for AdamW.

---

## Theoretical Foundation

Muon can be viewed as **steepest descent in the spectral norm** (a matrix norm). Orthogonalizing the gradient update means each parameter update has equal influence on all singular values of the weight matrix, preventing the common problem of some dimensions being over-updated while others stagnate.

This is analogous to Adam's per-parameter learning rate adaptation but operating at the matrix level rather than the element level.

---

## Relevance to mllm

mllm is planning to use Muon for training. Based on the evidence:

**Expected benefit**: ~30-35% faster convergence vs AdamW, meaning the same quality model can be achieved with 30% fewer training tokens/steps. At a training run of N steps, Muon would reach equivalent quality at 0.7N steps.

**Risks**:
- Less proven at very small model scales (<200M). NanoGPT (124M) data is encouraging but mllm (117M) is in a regime where the benefit might be smaller.
- Requires a two-optimizer setup (Muon for 2D params + AdamW for everything else) — slightly more complex training code
- Optimal hyperparameters (lr, momentum) may need tuning

**Implementation complexity**: Low. The Newton-Schulz iteration is ~10 lines of PyTorch. PyTorch 2.11 now ships Muon as `torch.optim.Muon` (no longer experimental).

**Recommendation**: Use Muon for the next training run. The evidence for 30%+ convergence speedup is strong, overhead is negligible, and the implementation is now in PyTorch core. Tune lr with a small ablation before the full training run.
