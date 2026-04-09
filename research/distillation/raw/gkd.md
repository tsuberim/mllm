# GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models

**URL:** https://arxiv.org/abs/2306.13649  
**Published:** ICLR 2024 (Google DeepMind)  
**TRL implementation:** https://huggingface.co/docs/trl/main/en/gkd_trainer  
**License:** Apache-2.0 (TRL implementation)

## Teacher models used
Not fixed — the paper demonstrates on summarization (XSum), translation, arithmetic reasoning, and instruction-tuning tasks. The TRL implementation uses any HuggingFace model as teacher (example: Qwen2-1.5B-Instruct → Qwen2-0.5B-Instruct).

## Distribution data included
No pre-stored logit dataset. GKD is a training framework, not a dataset. The teacher is called live during training to get per-token log-probabilities for student-generated sequences.

Key config knobs:
- `lmbda`: fraction of on-policy (student-generated) vs off-policy data
- `beta`: interpolation between forward KL (0.0) and reverse KL (1.0) via generalized JSD
- `temperature`: controls softening of the teacher distribution

When `lmbda=0.0`, reduces to standard supervised KD with teacher's token probabilities as soft targets.

## Dataset format
Input dataset: conversational format (`messages` with `role`/`content` dicts). Logits computed on-the-fly during training — no pre-stored logit files.

## Key notes
- Addresses train/inference distribution mismatch by training on student's own generated sequences
- The TRL GKDTrainer wraps SFTTrainer; teacher model must fit in memory alongside student during training
- On-policy data (high `lmbda`) generally outperforms off-policy
- A GitHub issue (#2255 in huggingface/trl) requests offline teacher logit support — as of April 2026 this is not officially implemented in TRL
- For small-model distillation: requires teacher model in memory during training, which is the main practical bottleneck
