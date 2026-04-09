# MiniLLM: Knowledge Distillation of Large Language Models

**URL:** https://arxiv.org/abs/2306.08543  
**DOI:** arXiv:2306.08543  
**Published:** ICLR 2024 (Microsoft Research)  
**Code:** https://github.com/microsoft/LMOps (minillm subdirectory)  
**License:** MIT (code)

## Teacher models used
GPT-2 XL (1.5B), OPT-13B, GPT-J 6B (in various experiments)

## Distribution data included
No pre-stored logit dataset released. The framework is on-policy: the teacher generates logit distributions on-the-fly during training. The student is trained against the teacher's token distribution using **reverse KL divergence** (not forward KL), which prevents the student from overestimating low-probability regions.

Training data: Databricks Dolly 15k (instruction-response pairs). No separate logit cache file released.

## Dataset size
Dolly 15k: ~15,000 instruction-following examples. No standalone logit dataset.

## Key notes
- Core contribution is the loss function change: reverse KLD instead of forward KLD
- Covers student scales 120M–13B
- Methodology is on-policy (live teacher inference during training), not offline logit caching
- Code and model checkpoints released; no pre-computed logit dumps provided
- DistiLLM (ICML 2024, https://github.com/jongwooko/distillm) extends this work with streamlined distillation; also on-policy, no released logit datasets
