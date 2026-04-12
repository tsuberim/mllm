# Google TPU Research Cloud (TRC)

Apply at: https://sites.research.google/trc/about/
Value: access to thousands of TPU chips (v4, v5e)
Rolling. Requirement: publish/open-source results.

**Caveat:** Training stack is PyTorch + CUDA. TPU requires JAX or PyTorch/XLA — non-trivial porting effort. Only worth pursuing if credits are large enough to justify the stack switch, or if JAX port is planned anyway.

**Upside:** TRC is well-known to fund independent researchers and OSS projects. Competition is lower than cash grants since TPU access has a higher usage barrier.

---

## Application text

**Project title:** Merlin — pre-training a coding agent LLM on TPU

**Research summary:**

We are pre-training a 3B-parameter language model from scratch on a 100B-token corpus optimized for agentic coding tasks. The model uses a custom tokenizer with tool-call protocol tokens, a 6K context window, and will be post-trained with GRPO RL on verifiable bash/filesystem rewards. Target deployment: local inference on Apple Silicon (MLX, int4).

**Compute need:** ~1,520 H100-equivalent GPU-hours for the pre-training run. TPU v4 or v5e pods would cover this; we estimate ~380–760 TPU-hours depending on pod configuration and efficiency.

**Open-source commitment:** All outputs published under MIT — model weights, training code, data pipeline, trace dataset, and a research report on training efficiency and agentic eval results. We will specifically report TPU vs. GPU training throughput comparisons.

**What's already built:** 1.19B-token v0 corpus (HuggingFace), custom BPE tokenizer, agentic eval harness (49 tasks, 47% on 3B baseline), full PyTorch training loop.

**Stack note:** Current training uses PyTorch + CUDA. TPU deployment would use PyTorch/XLA or we will evaluate a JAX port — the simpler transformer architecture (no MoE, no custom CUDA kernels) makes this tractable.

**Links:**
- https://huggingface.co/tsuberim/merlin-tokenizer-v0
- https://huggingface.co/datasets/tsuberim/merlin-corpus-v0

---

*Note: if PyTorch/XLA porting effort is too high, deprioritize this and check back when JAX training is on the roadmap.*
