# Dataset

## v1 Target Corpus (planned)

Target: **~100B tokens** across high-quality code, technical NL, math, and instruction data.
Selection strategy: The Stack v2 is already license-filtered, deduplicated, and curated by BigCode — no quality classifier needed. Token budgets per language are hit via random sampling. Minimal filters applied only where the source has known noise (generated files, non-English, pure data dumps).

### Code (~54%)

Token budgets are sampled randomly from each source to hit the target; Stack v2 raw sizes from the StarCoder2 paper.

| Source | License | Token budget | Filter |
|---|---|---|---|
| The Stack v2 — Python | Apache 2.0 | 20B | `is_generated=false`, `is_vendor=false` |
| The Stack v2 — TypeScript | Apache 2.0 | 5B | `is_generated=false` |
| The Stack v2 — Go | Apache 2.0 | 3B | `is_generated=false` |
| The Stack v2 — Rust | Apache 2.0 | 2B | `is_generated=false` |
| The Stack v2 — Bash/Shell | Apache 2.0 | 2B | `is_generated=false`, ≥2 commands |
| The Stack v2 — YAML | Apache 2.0 | 2B | CI/docker/k8s files only |
| The Stack v2 — Dockerfile | Apache 2.0 | 0.3B | passthrough |
| The Stack v2 — SQL | Apache 2.0 | 3B | has SELECT/INSERT/CREATE, no pure data dumps |
| The Stack v2 — Markdown | Apache 2.0 | 5B | English, has code block, no templates |
| Jupyter notebooks (executed) | varies | 10B | has cell outputs + markdown cells |
| PyPI package READMEs | varies | 0.3B | has code examples |
| GitHub issues + commits | Apache 2.0 | 1.5B | non-trivial messages only |
| Rosetta Code + Exercism | Apache 2.0 | 0.2B | passthrough |
| Papers with Code | CC BY-SA | 0.5B | arXiv abstract + methods section |

### Q&A (~5%)

| Source | License | Token budget | Filter |
|---|---|---|---|
| Stack Overflow | CC BY-SA 4.0 | 1B | accepted answers, score >20, code block, domain filter |
| Code Review SE | CC BY-SA 4.0 | 1B | question + accepted review |
| Unix.SE + ServerFault + AskUbuntu | CC BY-SA 4.0 | 2B | accepted answers |
| Software Engineering SE | CC BY-SA 4.0 | 0.7B | accepted answers |
| DevOps SE + Data Science SE | CC BY-SA 4.0 | 0.3B | accepted answers |

### Reference (~2%)

| Source | License | Token budget | Notes |
|---|---|---|---|
| Full man pages | GPL/BSD | 0.1B | complete reference, not just tldr |
| Python stdlib docs | PSF | 0.3B | |
| PEPs | PSF | 0.05B | all PEPs |
| Git book + Docker docs + Bash manual | permissive | 0.5B | |
| RFCs (HTTP, JSON, UNIX subset) | IETF | 0.1B | key protocols only |
| Library docs (NumPy, Pandas, requests, etc.) | BSD/MIT | 0.1B | top-10 packages only; actual coverage is ~0.1B |
| tldr-pages | CC0 | 0.3B | |

### Pedagogical (~3%)

| Source | License | Token budget | Notes |
|---|---|---|---|
| Dev.to + HashNode technical posts | CC BY-SA 4.0 | 2B | developer tutorials |
| Wikibooks — Computing/Programming | CC BY-SA | 0.7B | structured reference |
| Python Data Science Handbook | MIT | 0.2B | executed Jupyter book |
| Fast.ai course notebooks | Apache 2.0 | 0.2B | practical ML |
| SICP | CC BY-SA 4.0 | 0.05B | computational thinking |
| Python official tutorial | PSF | 0.05B | |

### NL / General Knowledge (~11%)

| Source | License | Token budget | Filter |
|---|---|---|---|
| FineWeb-Edu (tech/science subset) | ODC-BY | 7B | education quality score ≥4 |
| ArXiv CS | arXiv ToS | 3B | abstracts + intro sections |
| Wikipedia (CS/computing/math) | CC BY-SA | 0.8B | category-filtered; actual coverage ~0.8B |

### Instruction Following (~5%)

| Source | License | Token budget | Notes |
|---|---|---|---|
| FLAN v2 (code + reasoning subsets) | Apache 2.0 | 3B | no LLM outputs |
| Natural Instructions v2 | Apache 2.0 | 1.5B | 1600+ task types |
| OpenHermes 2.5 | Apache 2.0 | 1B | Mistral-generated, not GPT-4; actual ~0.7–1B |

### Math (~6%)

| Source | License | Token budget | Notes |
|---|---|---|---|
| NuminaMath | Apache 2.0 | 1.5B | competition math, worked solutions |
| DeepMind Mathematics | Apache 2.0 | 1.5B | synthetic, broad coverage |
| Proof-Pile 2 (subset) | Apache 2.0 | 3B | arXiv math; 55B total, taking ~5% |
| Math Jupyter notebooks | varies | 0.5B | notebooks with math + Python |

### Synthetic (~15%)

| Source | License | Token budget | Notes |
|---|---|---|---|
| Agentic traces (Qwen2.5-Coder-32B) | Qianwen v1.1 | 5B | Docker-validated; generated in Milestone 3b |
| Synthetic pretraining (Qwen2.5-Coder-32B) | Qianwen v1.1 | 10B | exercises, docstrings, debug scenarios |

> Qianwen License v1.1 permits training other models on outputs. Verify before public release.

### Proportions summary

| Category | Token budget | % |
|---|---|---|
| Code | ~56B | ~54% |
| Synthetic | ~15B | ~14% |
| NL / general knowledge | ~11B | ~11% |
| Math | ~6B | ~6% |
| Instruction following | ~5B | ~5% |
| Q&A | ~5B | ~5% |
| Reference + pedagogical | ~5B | ~5% |
| **Total** | **~103B** | |

### Quality strategy

- **Stack v2**: already license-filtered, near-deduped, and curated by BigCode. Random sample to token budget. Only apply `is_generated=false` and `is_vendor=false` flags that are already in the dataset.
- **Q&A**: accepted answers only + score threshold — human-curation signal already baked in.
- **Cross-source dedup**: MinHash LSH dedup across all sources before packing — Stack v2 Python, Jupyter notebooks, and PyPI READMEs all overlap.
- **Docstring examples** (`>>>` lines): extract and deduplicate as a free filter pass — verified running code.
- **No NC-licensed content** (CC BY-NC, etc.) — safe for commercial release.

### Curriculum

Two-phase training schedule:
- **Phase 1 (~80% of steps)**: full mix above; general code + NL + math
- **Phase 2 (~20% of steps)**: upweight synthetic traces + instruction data; downweight general NL

---

## v0 Corpus (merlin-corpus-v0)

Pretraining corpus focused on code and technical text. No general web, no Wikipedia.

**1.19B tokens · 100% packing efficiency · fits in H100 HBM (~2.4 GB)**

HuggingFace: [tsuberim/merlin-corpus-v0](https://huggingface.co/datasets/tsuberim/merlin-corpus-v0)
Tokenizer: [tsuberim/merlin-tokenizer-v0](https://huggingface.co/tsuberim/merlin-tokenizer-v0)

> v0 used The Stack v1 at experiment scale. Superseded by v1 target above.

### Sources

| Source | Records | Notes |
|---|---|---|
| The Stack — Python | ~730K | AST-parsed; must have `def`/`class`; no generated files |
| The Stack — Bash | ~214K | ≥2 shell commands |
| The Stack — Markdown | ~63K | English-only; must have backtick; no HTML blobs |
| Stack Exchange Q&A | ~21K | Python/bash/linux/git/docker topic filter |
| GitHub commits | ~84K | Non-trivial commit messages |
| GitHub issues | ~71K | Topic-filtered; no bot posts |
| tldr-pages | ~7K | Clean command references |

### Format

```python
import numpy as np

train = np.fromfile("corpus_train.bin", dtype=np.uint16).reshape(-1, 6144)  # (173911, 6144)
val   = np.fromfile("corpus_val.bin",   dtype=np.uint16).reshape(-1, 6144)  # (19340, 6144)
```

- `dtype`: uint16 (2 bytes/token; vocab fits in 16 bits)
- `shape`: `[N, 6144]` — each row is a packed sequence of complete documents
- Documents separated by `<|eos|>` (ID=1); ~6 docs/chunk on average
- No padding — 100% token utilisation

### Train/Val Split

- 90/10 split at document boundaries
- Shuffled at document level with `seed=42`

### Attention Masking

Documents are packed contiguously. At training time a block-diagonal causal mask prevents attention across document boundaries:

```python
is_eos = (x == EOS_ID)
doc_id = torch.cat([zeros, is_eos[:, :-1].cumsum(dim=1)], dim=1)  # (B, T)
mask = (doc_id.unsqueeze(2) == doc_id.unsqueeze(1)) & causal       # (B, 1, T, T)
```

Note: custom mask disables FlashAttention in SDPA. For T > ~1024 (3b/7b configs), switch to `flash_attn_varlen_func`.
