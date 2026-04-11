# AI Grant Application

Apply at: https://aigrant.org/
Format: short form (name, email, project description, links). No pitch deck required.
Requirement: 90-day blog post after funding.

---

## Project name

Merlin: an open-source 3B language model for agentic coding

## One-line description

A small, fast LLM pre-trained from scratch to be the reliable cheap worker that smarter AI orchestrators spawn in bulk — running locally on any MacBook.

## What are you building? (~300 words)

Most LLM agent costs are wasted on grunt work: grep a file, run a test, rename a function. Every sub-task hits a frontier model at $1–10/M tokens. A developer running parallel repo-wide refactors or a nightly pre-commit agent is spending real money on tasks that require execution, not reasoning.

Merlin is an open-source 3B parameter language model built specifically for this role. Not a general-purpose assistant — a fast, reliable worker that follows well-specified agentic instructions and reports results. Local inference on any M-series MacBook (int4, ~1.5GB), zero API cost, always on.

The design is deliberate:
- **Corpus**: 100B tokens weighted toward code (54%) and agentic traces (15%) — not general web text
- **Protocol**: custom tokenizer with special tokens for tool-call protocol; 6K context window sized for one large file + agent overhead
- **Post-training**: SFT on agentic traces → RL with verifiable rewards (bash exit codes, file diffs) — no proprietary model distillation, so the weights are clean for commercial use
- **Inference**: MLX on Apple Silicon; target >500 tok/s on M3

What's already built: full data pipeline (1.19B-token v0 corpus on HuggingFace), custom BPE tokenizer, 49-task agentic eval harness (47% on 3B-scale baseline), task mutation pipeline, SFT infrastructure, and an E2E training loop on H100 via Modal.

Total project cost to reach a trained 3B model: ~$5,100 (RunPod for pre-training, Modal CPU for scanning, DeepInfra API for trace generation).

Everything will be open-sourced: weights, training code, data pipeline, trace dataset, and a writeup of what worked and what didn't. The only ask is a 90-day blog post — already planned.

## What's the ask?

$10,000–$20,000 to cover pre-training compute ($4,089) and the full trace generation + repo scanning pipeline (~$500), with buffer for experiments and ablations.

If the full $50K is on offer: add a 7B variant and fund a proper benchmarking study comparing worker-orchestrator architectures vs. single-model baselines.

## Links

- HuggingFace (tokenizer): https://huggingface.co/tsuberim/merlin-tokenizer-v0
- HuggingFace (corpus v0): https://huggingface.co/datasets/tsuberim/merlin-corpus-v0
- GitHub: (will be public before submission)

## Prior work / background

Independent developer. Previously built production systems at [fill in]. This project is self-funded to this point — the v0 corpus, tokenizer, and training infrastructure are done on personal compute budget.

---

*Notes for submission: keep it under 500 words total. AI Grant values conciseness. Lead with the problem, not the model. The "no proprietary distillation" angle is worth emphasizing — they fund open work.*
