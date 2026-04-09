# TRL GOLD Trainer (General Online Logit Distillation)

**URL:** https://huggingface.co/docs/trl/main/en/gold_trainer  
**License:** Apache-2.0 (TRL)

## Overview
Extension of Universal Logit Distillation (ULD) in HuggingFace TRL. Supports student/teacher pairs with **different tokenizers**, aligning textual spans and merging logits so no tokens are dropped. Enables cross-tokenizer (cross-architecture) logit distillation.

## Teacher models
Any HuggingFace causal LM. Designed especially for teacher/student pairs that don't share a vocabulary (e.g., Llama teacher → GPT-2 student).

## Distribution data
Online (on-the-fly): teacher logits computed during training, not pre-stored. The tokenizer alignment step maps teacher vocabulary positions to student vocabulary positions at the sub-word span level.

## Format
No pre-stored logit dataset format — pure online inference.

## Key notes
- Cross-tokenizer support is the distinguishing feature vs GKDTrainer
- Relevant if this project later distills from a model with a different tokenizer (e.g., Llama 3 BPE → custom tokenizer)
- No standalone dataset; requires live teacher during training
- For a 117M student, the teacher must be loaded alongside, constraining GPU memory budget
