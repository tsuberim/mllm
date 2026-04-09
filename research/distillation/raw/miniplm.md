# MiniPLM: Knowledge Distillation for Pre-Training Language Models

**URL:** https://arxiv.org/abs/2410.17215  
**Published:** ICLR 2025 (Tsinghua University)  
**Code + data:** https://github.com/thu-coai/MiniPLM  
**HuggingFace org:** https://huggingface.co/MiniLLM  
**License:** MIT

## Teacher model used
Qwen1.5-1.8B (primary); reference model is a Qwen 104M trained on 5B tokens of the Pile.

## Distribution data included
A **refined pre-training corpus**, not raw logits. The method (Difference Sampling) uses the teacher's token log-probabilities to re-weight/filter the training data distribution, then releases the resulting curated text dataset.

**Released dataset:** `MiniLLM/pile-diff_samp-qwen_1.8B-qwen_104M-r0.5`  
- ~50 billion tokens (subset selected from 100B-token Pile corpus)  
- Format: binary shards (`.bin` + `.idx` pairs), ~1B tokens per shard  
- This is a **text corpus**, not a logit dump — the teacher's probabilities are used only to select/weight examples, not stored alongside them

## Key notes
- Offline teacher inference: teacher runs once over the corpus, student can be re-trained multiple times at no extra teacher cost
- Supports cross-architecture distillation (no tokenization matching required between teacher and student)
- Student models released: Qwen-based 200M/500M/1.2B, Mamba 130M, LLaMA3.1 212M
- The released artifact is a better pre-training corpus, not a logit dataset — useful for pretraining a student but without the per-token distribution signal of full logit distillation
