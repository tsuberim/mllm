# Loss Functions for LLM Distillation: Forward KL, Reverse KL, JSD, Skew-KL

## Overview of divergences

### Forward KL (mode-covering / mean-seeking)
`D_KL(p_teacher || q_student)` — student is penalized for underestimating any region where the teacher has mass.
- Pushes the student to spread probability over all modes the teacher assigns mass to
- With a less expressive student (fewer parameters), the student "averages" over modes, placing density in low-probability interstitial regions
- This mode-averaging behavior is associated with hallucinations in generation tasks
- Standard word-level KD (Hinton et al. 2015) uses this implicitly

### Reverse KL (mode-seeking / zero-forcing)
`D_KL(q_student || p_teacher)` — student is penalized for assigning probability where the teacher assigns none.
- Student "picks" one mode and ignores others rather than spreading mass
- Prevents probability in low-density regions → fewer hallucinations
- On reasoning tasks (GSM8K, MATH), consistently outperforms forward KL
- Can underfit multimodal distributions by collapsing to a single mode
- MiniLLM (ICLR 2024) uses this exclusively

### JSD (Jensen-Shannon Divergence)
`JSD(p, q) = 0.5 * D_KL(p || m) + 0.5 * D_KL(q || m)` where `m = 0.5*(p+q)`
- Symmetric, bounded [0, log2], well-defined even when supports don't overlap
- GKD (ICLR 2024, Google DeepMind) uses generalized JSD via the `beta` parameter: `beta * RKL + (1-beta) * FKL`
- "Safe" general-purpose choice; less sensitive to distribution mismatch
- Default in GKD is `beta=0.5` (pure JSD); tuning toward 1.0 favors reverse KL

### Skew-KL / Skew-Reverse-KL (DistiLLM)
```
D_SKL^(α)(p, q) = D_KL(p, α*p + (1-α)*q)          # Skew forward KL
D_SRKL^(α)(p, q) = D_KL(q, (1-α)*p + α*q)          # Skew reverse KL
```
- α=0 recovers standard forward KL (SKL) or reverse KL (SRKL)
- α=1 recovers uniform/trivial loss; optimal α≈0.1 per DistiLLM experiments
- Motivation: standard KL has unbounded gradients when probability ratios are large (mode mismatch early in training); mixing with the other distribution bounds the density ratio
- DistiLLM (ICML 2024) uses α=0.1 for SRKL

## Empirical comparison

From the ICLR 2025 blogpost ([link](https://iclr-blogposts.github.io/2025/blog/llm-knowledge-distil/)):
- **Token-level distillation**: forward KL and reverse KL show minimal difference on ROUGE metrics (e.g., ROUGE1 0.5404 vs 0.5291 for 7B→1.5B distillation)
- **Forward KL may converge faster** in simplified settings
- Difference only becomes meaningful at sequence level or with high task structure (reasoning)

From on-policy distillation survey ([arXiv:2604.00626](https://arxiv.org/html/2604.00626)):
- Reverse KL dominates on mathematical reasoning (GSM8K, MATH) — mode-seeking prevents averaging over distinct solutions
- Forward KL and JSD better for open-ended generation (MT-Bench, AlpacaEval) — diversity preservation matters
- **Entropy-Aware OPD**: adaptively switches — forward KL when teacher is uncertain (high entropy), reverse KL when teacher is confident. +5.05 Pass@8 on math

## Practical guidance

| Task type | Recommended divergence | Reason |
|---|---|---|
| Pretraining (general) | Forward KL or JSD | Broad distribution coverage; mode-averaging less harmful without a "wrong" answer |
| Instruction following | JSD or Reverse KL | Balance between diversity and precision |
| Mathematical reasoning | Reverse KL | Single correct answer; mode-seeking prevents averaging |
| Calibration-sensitive | Importance-sampled + Forward KL | Top-K bias correction per ACL 2025 paper |

## The T² scaling issue

When using temperature T, gradients of the KL loss scale by T². The standard Hinton formulation scales the distillation loss by T² to compensate, keeping gradient magnitudes comparable. **Always apply T² scaling when using temperature.**

Source: Hinton et al. (2015), confirmed in multiple survey sources.

## AKL (Adaptive KL)

A 2024 approach that challenges "reverse KL is always better": AKL allocates weights to forward and reverse KL based on current teacher/student distributions. Under practical training budgets:
- Forward KL focuses on the head of the distribution (high-probability tokens)
- Reverse KL focuses on the tail (low-probability tokens)
Both are useful at different training stages; AKL blends them dynamically.

## Sources
- [MiniLLM (ICLR 2024)](https://openreview.net/forum?id=5h0qf7IBZZ)
- [GKD (ICLR 2024)](https://arxiv.org/abs/2306.13649)
- [DistiLLM (ICML 2024)](https://arxiv.org/html/2402.03898) — formulas extracted directly
- [On-policy distillation survey (arXiv:2604.00626)](https://arxiv.org/html/2604.00626)
- [ICLR 2025 blogpost: Forward KL vs Reverse KL comparison](https://iclr-blogposts.github.io/2025/blog/llm-knowledge-distil/)
