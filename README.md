<img src="docs/logo.png" alt="Merlin wizard hat logo" width="36" align="left" style="margin-right:12px; image-rendering:pixelated;" />

# Merlin — Specialized Agentic Coding Model

A 3B language model built from scratch **exclusively** for agentic coding — not a general assistant, not a fine-tuned GPT. Every training token, every design decision, every protocol token is optimized for one job: executing code tasks fast, locally, and at scale.

Runs on any MacBook. No API key. Your code never leaves your machine.

→ **[tsuberim.github.io/merlin](https://tsuberim.github.io/merlin/)**

## The Idea

LLM agents spend most of their tokens on execution, not reasoning — grep a file, run a test, rename a function. Today that all hits a frontier model at $1–10/M tokens.

Merlin is the brute-force execution layer beneath a smarter orchestrator. A frontier model (Claude, GPT-4) plans; Merlin executes — locally, in parallel, at zero marginal cost.

| Mode | Workers | Cost |
|---|---|---|
| Local | 1, on your MacBook | $0 — always |
| Hosted | 100–1000 via GPU batching | pay-per-task |

## Design

| Choice | What | Why |
|---|---|---|
| Pre-trained from scratch | 100B tokens — code, bash, agentic traces, commits | No proprietary distillation; weights are commercially clean |
| Custom tokenizer | 32K BPE + 18 agent protocol special tokens | Tool-call protocol is first-class, not bolted on |
| 6K context window | Sized for one large Python file + agent overhead | Not a general-purpose model |
| RL post-training | GRPO on verifiable bash/filesystem rewards | Ground truth without a judge model |
| MLX int4 inference | ~1.5 GB weights, >500 tok/s on M3 | Fits any M-series Mac |

## Agent Protocol

18 special tokens define the tool-call format — the model learns to emit and parse tool calls natively:

```
<|task|> Read src/main.py and return the function names.
<|think|> I need to read the file first.<|/think|>
<|tool_call|><|tool_name|>read_file<|tool_args|>{"path": "src/main.py"}<|/tool_call|>
<|tool_result|>def train(): ...\ndef evaluate(): ...<|/tool_result|>
<|answer|> train, evaluate
```

## Architecture

GPT-style decoder-only. RMSNorm, SwiGLU, GQA (n_kv_head=8), no bias, weight tying, pre-norm.

| Config | Params | n_embd | n_head | n_layer | block_size |
|---|---|---|---|---|---|
| tiny | ~1.6M | 32 | 2 | 2 | 64 |
| medium | ~21M | 256 | 8 | 8 | 512 |
| base (330M) | ~330M | 1024 | 16 | 16 | 2048 |
| **3b** | **~3.17B** | **3072** | **24** | **20** | **4096** |

## Corpus

~100B tokens across 7 sources. Two-phase curriculum: 80B general mix → 20B upweighted traces + instruction data.

| Source | Share |
|---|---|
| Stack v2 — Python | ~38% |
| Stack v2 — Bash / Markdown | ~8% |
| Agentic traces (synthetic) | ~15% |
| GitHub commits + issues | ~11% |
| Stack Overflow | ~10% |
| Math + instruction mix | ~12% |
| tldr pages | <1% |

v0 corpus (1.19B tokens): [tsuberim/merlin-corpus-v0](https://huggingface.co/datasets/tsuberim/merlin-corpus-v0)
Tokenizer: [tsuberim/merlin-tokenizer-v0](https://huggingface.co/tsuberim/merlin-tokenizer-v0)

## Status

| Milestone | Status |
|---|---|
| Agentic protocol + eval harness (49 tasks, 47% on 3B baseline) | ✅ Done |
| Custom BPE tokenizer (32K vocab, 20 special tokens) | ✅ Done |
| Data pipeline (download → tokenize → pack) | ✅ Done |
| v0 corpus on HuggingFace (1.19B tokens) | ✅ Done |
| E2E training loop, 330M model on H100 | ✅ Done |
| SFT infrastructure | ✅ Done |
| Repo scanning pipeline (clone + pytest → passing repos) | 🔄 In progress |
| Agentic trace generation (target: 200K traces) | ⏸ Planned |
| Full 100B token corpus | ⏸ Planned |
| 3B pre-training run | ⏸ Planned |
| RL post-training (GRPO on verifiable rewards) | ⏸ Planned |
| MLX int4 3B model release | ⏸ Planned |

## Stack

| Role | Tool |
|---|---|
| Training | PyTorch + CUDA (NVIDIA H100) |
| Inference | MLX (Apple Silicon / Metal) |
| Cloud compute | Modal |
| Tokenizer | HuggingFace tokenizers (BPE, Rust) |
| Trace generation | vLLM + Qwen2.5-Coder-32B |
| Observability | W&B |
| Datasets + models | HuggingFace Hub |

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## License

MIT
