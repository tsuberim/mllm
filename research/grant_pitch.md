# Grant Application Pitch

Core materials to adapt per application. Keep the technical claims grounded in what's been built.

---

## One-liner

Merlin: an open-source 3B language model pre-trained from scratch to be a brute-force agentic worker — spawned in hundreds in parallel, locally or remotely, while a smarter orchestrator plans.

## Short description (100 words)

Merlin is a 3B-parameter language model purpose-built for agentic coding tasks — grep, file read/write, script execution, and parallel sub-task orchestration. Unlike general-purpose models fine-tuned on instruction data, Merlin is pre-trained from scratch on a 100B-token corpus of code and agentic traces, then post-trained with GRPO RL on verifiable bash/filesystem rewards. Target deployment: local inference on any M-series MacBook (int4, ~1.5GB). The model is fully open-source (weights, code, data pipeline). Primary use case: a cheap, fast worker that a smarter orchestrator (Claude, GPT-4) can spawn in bulk — reducing API costs by 50%+ for power users.

## Problem statement

LLM agents today are expensive. Every sub-task — grep a file, run a test, rename a variable — hits a frontier model API at $1–10/M tokens. A user running a repo-wide refactor or a nightly pre-commit agent spends real money on tasks that don't require frontier reasoning. Local models exist, but they're either too small to follow agentic protocols reliably, or too large and slow for low-latency local inference.

## Solution

A model trained specifically for the agentic worker role:
- **Corpus**: 100B tokens weighted toward code (54%), agentic traces (15%), technical NL — not general web text
- **Protocol**: custom tokenizer with 18 special tokens for tool-call protocol; context window sized at 6K (one large Python file + protocol overhead)
- **Post-training**: SFT on agentic traces → RL with verifiable rewards (bash exit codes, file diffs) → thinking fine-tune; no proprietary model distillation
- **Inference**: MLX on Apple Silicon; int4 quantization; target >500 tok/s on M3 MacBook

## What's been built

- Full data pipeline: streaming download + tokenization of 7 sources; 1.19B-token v0 corpus already on HuggingFace
- Custom BPE tokenizer (32K vocab, 20 special tokens) trained on Python + agent traces
- Agentic harness: 49-task eval suite; 47% on 3B-scale baseline
- Task mutation pipeline: clone + pytest scan → structured task dataset; 227 passing repos from 2,185 candidates so far
- SFT infrastructure: `sft.py` + data prep + Modal integration; validated on 330M proxy
- E2E training loop on H100 via Modal; 330M experiment model running

## Why open source

- Reproducibility: every design choice (corpus weights, protocol tokens, RL reward function) is auditable
- Community: agentic coding tooling is fast-moving; an open worker model lowers the floor for everyone building agents
- No proprietary distillation: weights, traces, and pipeline are clean for commercial use

## Budget

Total project cost: ~$5,900 (RunPod 8×H100 SXM for pre-training at $2.69/GPU-hr; Modal CPU for data pipeline and scanning; DeepInfra API for trace generation).

| Phase | Cost |
|---|---|
| Data pipeline | $20 |
| Repo scanning (50K→20K passing) | $250 |
| Trace generation (200K traces) | $166 |
| Pre-training 3B, 100B tokens | $4,089 |
| Post-training (SFT + RL + thinking) | $150 |
| Experiments & ablations | $450 |
| **Total** | **~$5,100** |

With grant coverage, remaining budget goes toward: 7B model experiments, larger trace corpus (500K+), and publishing results.

## Agentic AI angle (for Amazon Research Awards)

This project directly advances multi-agent software engineering systems:
- **Agentic trace generation at scale**: pipeline to generate 200K+ ground-truth traces of a 32B model solving real coding tasks in sandboxed Docker environments — a dataset contribution in its own right
- **Verifiable RL rewards**: bash exit codes and file diffs provide ground truth without a judge model; scalable to millions of tasks
- **Worker-orchestrator architecture**: explicit design for a two-tier agent system where Merlin handles atomic tasks and a frontier model handles planning — publishable architecture comparison vs. single-model baselines

## Links

- GitHub: (to be made public before submission)
- HuggingFace: [tsuberim/merlin-tokenizer-v0](https://huggingface.co/tsuberim/merlin-tokenizer-v0), [tsuberim/merlin-corpus-v0](https://huggingface.co/datasets/tsuberim/merlin-corpus-v0)
