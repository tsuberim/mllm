# Lambda Labs Research Grant

Apply at: https://lambdalabs.com/research-grant (or via aicredits.dev link)
Value: up to $5K compute credits
Rolling deadline. CSO mentoring included.

---

## Application text

**Project:** Merlin — open-source 3B language model for agentic coding

**Description:**

Merlin is a 3B-parameter LLM pre-trained from scratch on a 100B-token corpus (54% code, 15% agentic traces) and post-trained with RL on verifiable bash/filesystem rewards. Target: local inference on Apple Silicon (int4, ~1.5GB, >500 tok/s on M3).

Design rationale: frontier models are expensive for agentic sub-tasks (grep, file I/O, script execution). Merlin is the cheap, fast worker that a smarter orchestrator can spawn in bulk — reducing API costs 50%+ with zero marginal cost for local users.

**What's built:** 1.19B-token corpus on HuggingFace, custom 32K BPE tokenizer, 49-task agentic eval suite (47% on 3B baseline), SFT infrastructure, E2E training loop on H100.

**Compute ask:** $5K credits toward pre-training experiments and ablations on Lambda H100 instances.

**Open-source commitment:** Apache 2.0 — weights, code, data pipeline, trace dataset, results writeup.

**Links:**
- HuggingFace: https://huggingface.co/tsuberim/merlin-tokenizer-v0
- Corpus: https://huggingface.co/datasets/tsuberim/merlin-corpus-v0
- GitHub: (public before submission)
