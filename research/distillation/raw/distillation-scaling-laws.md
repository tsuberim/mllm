# Distillation Scaling Laws

## Paper
**"Distillation Scaling Laws"** (Busbridge et al., Graphcore Research)  
**URL:** https://arxiv.org/abs/2502.08606  
**Published:** ICML 2025  
**Summary:** Proposes power-law relationships between compute budget, teacher/student sizes, and distilled model performance.

## Key findings (from abstract + search results)

### When distillation outperforms supervised pretraining
- **Distillation wins** when: a teacher already exists (you don't need to train it) OR when you plan to distill into multiple students from the same teacher
- **Supervised pretraining wins** when: only one student is ever trained AND the teacher must also be trained from scratch (you'd spend more total compute on the teacher than you'd gain)

**Implication for Merlin**: Since we're using an existing open-weights teacher (Qwen2.5-1.5B or 3B), distillation is the compute-optimal path. The teacher is already trained; we get its knowledge for free.

### The capacity gap / power law behavior
The influence of teacher quality on student loss follows a **power law** that transitions between two regimes based on teacher/student relative capacity. The transition point is the "capacity gap":
- Below the threshold: student improves as teacher gets stronger
- Above the threshold: teacher quality gains no longer improve the student

From related literature ([design-space paper](https://arxiv.org/abs/2410.16215)):
- Largest tested student (6.8B) benefited most from distillation
- Smallest tested student (330M) benefited less — suggesting 117M may be at the lower end of where distillation reliably helps
- Distillation effectiveness requires student size roughly ≥ 5–10% of teacher size (inferred from capacity gap literature)

**Capacity gap guidance for 117M student:**
- Qwen2.5-1.5B teacher: 117M/1500M ≈ 8% — borderline; likely fine
- Qwen2.5-3B teacher: 117M/3000M ≈ 4% — below the typical threshold; may not outperform 1.5B teacher
- Qwen2.5-7B teacher: 117M/7000M ≈ 2% — likely counterproductive; capacity gap too large
- **Best candidate: Qwen2.5-1.5B** for this student size

### Distillation compute budget
Paper provides "compute-optimal distillation recipes" for:
1. Scenario A: Teacher already exists → always distill if you can
2. Scenario B: Teacher needs training → supervised is usually better unless many students

Full numerical details require reading the 69-page paper. The blog at graphcore-research.github.io (currently 404) was the primary public summary.

## Related: Law of Capacity Gap
**"Towards the Law of Capacity Gap in Distilling Language Models"** (ACL 2025)
**URL:** https://aclanthology.org/2025.acl-long.1097.pdf

Key claim: "the optimal teacher scale almost consistently follows a linear scaling with the student scale." Roughly: optimal teacher is 3–4× the student size for language models.

For a 117M student: optimal teacher is roughly **350M–500M**. This is smaller than common assumptions.
- Qwen2.5-0.5B (494M): within range — good teacher candidate
- Qwen2.5-1.5B (1500M): 13× larger — may hit capacity gap for very small student

**Caveat**: This is from the capacity gap paper, not the scaling laws paper. The design-space paper's empirical results with a 9B teacher show diminishing returns for small students but don't pinpoint an optimal ratio. These results are in tension — the capacity gap paper suggests smaller teachers may be better, the scaling laws paper says distillation always beats supervised given an existing teacher.

**Unresolved question**: For a 117M student, is Qwen2.5-0.5B a better teacher than Qwen2.5-1.5B? This is worth ablating empirically.

## Distillation scaling laws vs Chinchilla
The Distillation Scaling Laws paper fills a gap left by Chinchilla: Chinchilla tells you how many tokens to train a model of size N from scratch; the distillation scaling laws tell you how to allocate a compute budget when a teacher is available. The two are complementary.

## Sources
- [arXiv:2502.08606 — Distillation Scaling Laws (ICML 2025)](https://arxiv.org/abs/2502.08606)
- [ACL 2025 capacity gap paper](https://aclanthology.org/2025.acl-long.1097.pdf)
- [Design-space paper (arXiv:2410.16215)](https://arxiv.org/abs/2410.16215) — empirical capacity gap evidence
