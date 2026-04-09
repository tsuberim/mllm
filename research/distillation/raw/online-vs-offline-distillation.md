# Online vs Offline Distillation — Tradeoffs

## Definitions

**Online distillation**: Teacher is loaded into memory alongside the student during training. Teacher runs forward pass on every batch (or on student-generated sequences). Logits flow directly into the loss. No pre-storage.

**Offline distillation**: Teacher runs inference over the corpus in a separate job. Logits stored to disk. Training loop loads stored logits; teacher not needed during training.

**Hybrid (semi-online)**: Teacher is called infrequently (e.g., once per epoch, or only on "hard" examples). DistiLLM's replay buffer approach approximates this.

## Quality comparison

From the design-space paper ([arXiv:2410.16215](https://arxiv.org/abs/2410.16215)):
- **Offline (pre-trained teacher)**: +1.6% average improvement over baseline
- **Online (early-checkpoint teacher, first 100B tokens)**: −20.9% — catastrophically bad
- **Online (late-checkpoint teacher)**: +0.5% with adjusted settings (α=0.1)
- Conclusion: **offline logits from a converged teacher beat online logits from a partially trained teacher**. If doing online distillation, the teacher must be fully trained first.

From MiniLLM/GKD literature: on-policy (student generates, teacher scores) generally outperforms off-policy (teacher generates, student imitates). But this is for fine-tuning scenarios, not pretraining from scratch.

## Practical constraints for a solo developer / single GPU

**Constraint**: Training a 117M student on a single NVIDIA GPU. Teacher model is Qwen2.5-1.5B or 3B.

**Online with small teacher (feasible)**: A Qwen2.5-1.5B teacher at fp16 requires ~3 GB VRAM. Combined with a 117M student (~450 MB fp16), total is ~3.5 GB — easily fits in a 24 GB RTX 4090 or similar. In theory, online distillation is possible.

**Online bottleneck**: Running teacher forward pass on every training batch increases training time by roughly 1× (doubling compute cost). For pretraining-scale runs (billions of tokens), this is prohibitive without careful batching.

**Offline bottleneck**: Requires generating logits first, then training. Two-pass process. Storage cost: ~500 MB–2.5 GB per billion tokens with top-64 sparse format (see design-space paper + sparse logit sampling notes).

## Recommendation for this project

**Offline is the better default for pretraining.** Rationale:
1. Logit generation is embarrassingly parallelizable and can be run separately (e.g., on a rented H100 for a few hours to generate 5B tokens of Qwen2.5-3B logits)
2. Teacher does not consume GPU memory during training, allowing larger batch sizes
3. Offline logits from a converged teacher are higher quality than online logits from a partially trained teacher
4. Storage cost is manageable (~2–5 GB for a multi-epoch pretraining run)

**Exception**: For fine-tuning / instruction-tuning phases after pretraining, online GKD-style distillation is practical (small dataset, teacher fits alongside student).

## DistiLLM's hybrid approach

DistiLLM uses a **replay buffer** of size 1000 student-generated outputs. Instead of running the teacher on every batch, it reuses previously scored sequences with probability λ_R = φ(1 - t/T). This gives most of the benefit of on-policy training at a fraction of the teacher inference cost. Achieves 2.5–4.3× speedup over pure on-policy methods.

Relevant if this project does online fine-tuning after offline pretraining distillation.

## Sources
- [Pre-training Distillation Design Space (arXiv:2410.16215)](https://arxiv.org/abs/2410.16215)
- [DistiLLM (ICML 2024, arXiv:2402.03898)](https://arxiv.org/html/2402.03898)
- [GKD (ICLR 2024)](https://arxiv.org/abs/2306.13649)
- [On-policy distillation survey (arXiv:2604.00626)](https://arxiv.org/html/2604.00626)
