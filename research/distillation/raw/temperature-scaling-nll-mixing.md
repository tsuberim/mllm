# Temperature Scaling and NLL Mixing in Distillation

## Temperature scaling

### Effect on soft labels
Temperature T applied to logits: `p_i = softmax(z_i / T)`
- T=1: original distribution
- T>1: softer — more probability mass given to low-probability tokens ("dark knowledge")
- T<1: sharper — distribution approaches one-hot

For distillation:
- High T makes the teacher distribution more uniform, which helps the student learn from low-probability tokens that carry structural information
- Low T makes the teacher distribution sharper, which emphasizes the most likely tokens

### The T² scaling rule
When computing the KL loss with temperature, the gradient magnitude is proportional to 1/T. To keep gradients comparable across different T values, the distillation loss is conventionally scaled by T²:
```
L_distill = T² * KL(softmax(z_T / T) || softmax(z_S / T))
```
**Always apply T² rescaling** when varying T. Failing to do this will cause the distillation loss to shrink as T increases, effectively deweighting it.

### Empirical sensitivity (pretraining scale)
From the design-space paper ([arXiv:2410.16215](https://arxiv.org/abs/2410.16215)):
- τ ≤ 2.0: ~2.5% improvement over baseline, roughly flat across T=0.5, 1.0, 2.0
- τ ≥ 5.0: limited improvement or degradation
- Adaptive temperature methods (NormKD, WTTM) showed no significant additional benefit
- **Practical recommendation: use T=1.0 or T=2.0, don't over-tune it**

### Note on sparse logit generation
When generating logits offline with top-K sampling, the temperature is applied before truncation. If you later load the sparse logits and re-apply temperature at training time, you're applying it twice. Decide at generation time whether to store raw logits or pre-temperature logits.

## NLL / Cross-Entropy mixing ratio

### Why mix at all?
Pure distillation loss (λ=1.0) can cause:
- Distribution collapse if the teacher's distribution is systematically off
- Failure to learn from correct labels (the "ground truth" anchor)
- Instability when student distribution diverges far from teacher early in training

Adding a small NLL term on hard labels acts as a regularizer and keeps training grounded.

### Empirical findings on mixing ratio

From the design-space paper ([arXiv:2410.16215](https://arxiv.org/abs/2410.16215)):
- **Optimal α = 0.9 (90% distillation, 10% NLL)** — peak performance
- α=1.0 (pure distillation) slightly underperforms α=0.9
- WSD scheduler for α (warmup-stable-decay, ramping the distillation contribution over time) gave best results overall: **+8.0% improvement** with WSD-scheduled α=0.9

Combined NLL and KLD loss both outperform LM-only baseline; KLD slightly better than NLL alone except on hard benchmarks (MMLU, C-Eval) where NLL matches or exceeds KLD.

### Standard formulation
```python
loss = alpha * L_distill + (1 - alpha) * L_NLL
# where alpha = 0.9 is the recommended default for pretraining
```

### Fine-tuning vs pretraining
For instruction fine-tuning (GKD, MiniLLM context), the mixing ratio is controlled by `lambda` (fraction of on-policy data) and `beta` (forward/reverse KL interpolation). The 90/10 distillation/NLL split from the design-space paper is specifically for pretraining-scale offline distillation.

## Sources
- [Pre-training Distillation Design Space (arXiv:2410.16215)](https://arxiv.org/abs/2410.16215) — primary source for temperature and mixing ratio numbers
- [Hinton et al. 2015 "Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531) — T² scaling rule
- [GKD (arXiv:2306.13649)](https://arxiv.org/abs/2306.13649) — beta parameter for online fine-tuning scenarios
