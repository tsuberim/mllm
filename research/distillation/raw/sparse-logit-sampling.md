# Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs

**URL:** https://arxiv.org/abs/2503.16870  
**Published:** ACL 2025 (Oral) — Samsung Research  
**Code:** https://github.com/akhilkedia/RandomSamplingKD  
**License:** Open (ACL proceedings)

## Teacher models used
Various, covering student model sizes 300M–3B (specific teacher names not in abstract).

## Distribution data included
No pre-stored public logit dataset. The paper proposes a *method* for sparse logit caching that is better than top-k caching.

**Key insight:** Top-K caching gives a biased estimate of the teacher distribution. Storing only the highest-probability tokens discards the remaining probability mass unevenly, causing suboptimal student calibration.

**Proposed fix:** Random Sampling Knowledge Distillation — sample tokens proportional to their probability (importance sampling) rather than always taking the top-K. This:
- Gives an unbiased gradient estimate in expectation
- Requires significantly sparser stored logits than top-K for the same quality
- Adds <10% overhead vs standard cross-entropy training

## Format
Sparse (token_id, log_prob) pairs sampled by importance weighting. No fixed k — sparsity is dynamic.

## Key notes
- This is directly relevant to any project building logit distillation datasets: top-k is the wrong approach if you care about calibration
- The bias in top-k caching is an under-discussed practical problem
- Accepted as Oral at ACL 2025 — carries weight
- No dataset released, but the GitHub code could be used to generate and store datasets using this format
