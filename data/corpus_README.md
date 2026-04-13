---
license: other
language:
- en
tags:
- code
- pretraining
- agentic
size_categories:
- 10B<n<100B
---

# Merlin Corpus v1

Pretraining corpus for [Merlin](https://github.com/tsuberim/merlin) — a small language model
purpose-built for agentic coding on Apple Silicon. Target: 3B parameters, 6K context, fast local inference.

Two scales are provided:

| Scale | Path | Tokens | Chunks | Use |
|---|---|---|---|---|
| Experiment | `experiment/` | ~7B | ~570K | Rapid iteration, ablations |
| Full | `full/` | ~88B | ~7.2M | Production pretraining run |

Both use identical format, tokenizer, and source mix — only the per-source document cap differs.

---

## Format

Binary files, one per split:

```
experiment/corpus_train.bin   # 90% of shuffled documents
experiment/corpus_val.bin     # 10% of shuffled documents
full/corpus_train.bin
full/corpus_val.bin
```

Each file is a flat array of **uint16** tokens packed into fixed-length **6144-token** chunks:

```python
import numpy as np

train = np.fromfile("experiment/corpus_train.bin", dtype=np.uint16).reshape(-1, 6144)
val   = np.fromfile("experiment/corpus_val.bin",   dtype=np.uint16).reshape(-1, 6144)
# train.shape → (N, 6144)
```

- **dtype**: `uint16` — vocab fits comfortably in 16 bits (vocab size 32,016)
- **packing**: documents are concatenated greedily; no padding — ~100% token utilisation
- **document separator**: `<|eos|>` (token ID 1) marks every document boundary
- **train/val split**: 90/10 at document level, shuffled with `seed=42`

### Loading a batch

```python
import numpy as np
import torch

data = np.fromfile("experiment/corpus_train.bin", dtype=np.uint16).reshape(-1, 6144)
idx  = np.random.randint(0, len(data), size=batch_size)
x    = torch.from_numpy(data[idx].astype(np.int32))  # (B, 6144)
```

### Attention masking across document boundaries

Documents are packed contiguously, so a block-diagonal causal mask is needed at training time
to prevent cross-document attention:

```python
EOS_ID = 1
is_eos = (x == EOS_ID)
doc_id = torch.cat([torch.zeros_like(is_eos[:, :1]), is_eos[:, :-1].cumsum(dim=1)], dim=1)
mask   = (doc_id.unsqueeze(2) == doc_id.unsqueeze(1)) & causal_mask  # (B, 1, T, T)
```

---

## Tokenizer

[tsuberim/merlin-tokenizer-v0](https://huggingface.co/tsuberim/merlin-tokenizer-v0)

- BPE, 32,016 tokens (32K base + 16 special tokens for agent protocol + `<|bos|>` / `<|eos|>`)
- Trained on Python, Bash, Markdown, shell traces, and agent protocol examples
- `<|bos|>` = 0, `<|eos|>` = 1

---

## Sources

~88B tokens across code, technical NL, math, and instruction data
(experiment scale uses ~7B via per-source document caps).

### Code (~54%)

| Source | Dataset | Token budget |
|---|---|---|
| The Stack v2 — Python | `bigcode/the-stack-v2-dedup` | 20B |
| The Stack v2 — TypeScript | `bigcode/the-stack-v2-dedup` | 5B |
| The Stack v2 — Go | `bigcode/the-stack-v2-dedup` | 3B |
| The Stack v2 — Rust | `bigcode/the-stack-v2-dedup` | 2B |
| The Stack v2 — Bash/Shell | `bigcode/the-stack-v2-dedup` | 2B |
| The Stack v2 — YAML | `bigcode/the-stack-v2-dedup` | 2B |
| The Stack v2 — Dockerfile | `bigcode/the-stack-v2-dedup` | 0.3B |
| The Stack v2 — SQL | `bigcode/the-stack-v2-dedup` | 3B |
| The Stack v2 — Markdown | `bigcode/the-stack-v2-dedup` | 5B |
| Jupyter notebooks (executed) | `codeparrot/github-jupyter-parsed` | 10B |
| PyPI package READMEs | `codeparrot/pypi-data` | 0.3B |
| GitHub commits | `bigcode/commitpackft` | 0.75B |
| GitHub issues | `bigcode/the-stack-github-issues` | 0.75B |
| Rosetta Code | `codeef/rosetta-code` | 0.2B |
| Papers with Code | `J0nasW/paperswithcode` | 0.5B |

### Q&A (~5%)

| Source | Dataset | Token budget |
|---|---|---|
| Stack Overflow | `bigcode/the-stack-v2-dedup` (SO subset) | 1B |
| Code Review / Unix.SE / ServerFault / AskUbuntu / SoftEng / DevOps / DataSci SE | Stack Exchange dump | ~4B |

### Reference (~2%)

| Source | Token budget |
|---|---|
| Full man pages | 0.1B |
| Python stdlib docs + tutorial | 0.3B |
| PEPs | 0.05B |
| Pro Git book + Docker docs + Bash manual | 0.5B |
| RFCs (HTTP, JSON, UNIX subset) | 0.1B |
| Library docs (NumPy, Pandas, scikit-learn, matplotlib, requests) | 0.1B |
| tldr-pages | 0.3B |

### Pedagogical (~3%)

| Source | Token budget |
|---|---|
| Wikibooks — Computing/Programming | 0.7B |
| Python Data Science Handbook | 0.2B |
| Fast.ai course notebooks | 0.2B |
| SICP | 0.05B |

### NL / General Knowledge (~11%)

| Source | Dataset | Token budget |
|---|---|---|
| FineWeb-Edu (education score ≥4) | `HuggingFaceFW/fineweb-edu` | 7B |
| ArXiv CS | `togethercomputer/RedPajama-Data-1T` | 3B |
| Wikipedia (CS/computing/math) | `wikimedia/wikipedia` | 0.8B |

### Instruction Following (~5%)

| Source | Dataset | Token budget |
|---|---|---|
| FLAN v2 (code + reasoning subsets) | `Muennighoff/flan` | 3B |
| Natural Instructions v2 | `Muennighoff/natural-instructions` | 1.5B |
| OpenHermes 2.5 | `teknium/OpenHermes-2.5` | 1B |
| NL2Bash | Dropbox archive | 0.01B |

### Math (~6%)

| Source | Dataset | Token budget |
|---|---|---|
| NuminaMath | `AI-MO/NuminaMath-CoT` | 1.5B |
| DeepMind Mathematics | `math-ai/orca-math-word-problems-200k` | 1.5B |
| Proof-Pile 2 (subset) | `EleutherAI/proof-pile-2` | 3B |
| MetaMathQA | `meta-math/MetaMathQA` | ~0.4B |

### What's not here

- **Synthetic agentic traces** (15B target): generated in a later pipeline stage (milestone 3b) — not yet available
- **Dev.to / HashNode** (2B target): no public dataset
- **Exercism**: only ~133 examples on HF — negligible, omitted

---

## Quality strategy

- **Stack v2**: BigCode already license-filtered, deduplicated, and curated. Filters: `is_generated=false`, `is_vendor=false`.
- **Q&A**: accepted answers only, score threshold, domain filter.
- **No NC-licensed content** — safe for commercial use.
- **Document-level shuffle** before packing; reproducible with `seed=42`.

---

## Pipeline

Built with [DataTrove](https://github.com/huggingface/datatrove) + custom adapters.
Source: [`tsuberim/merlin`](https://github.com/tsuberim/merlin), `data/pipeline/`.
