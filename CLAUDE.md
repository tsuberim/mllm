# merlin

A small language model purpose-built for agentic coding — running locally on Apple Silicon.
Not a deep reasoner. A fast, reliable worker: executes well-specified tasks, spawned in bulk by a smarter orchestrator.
Open source. Two deployment modes:
- **Local**: 1 worker, private, offline, zero cost
- **Hosted**: 100–1000 parallel workers via GPU batching — the only way to get real parallel speedup; pay-per-task

## Killer Apps

- **API cost reduction** — offload grunt work from Claude/GPT-4 to Merlin; local execution costs nothing. Power users spending $X/day on cloud APIs cut that by 50%+.
- **Parallel repo-wide refactoring** — spawn one Merlin per file, rename/update/fix across 200 files in seconds. The smart orchestrator just reviews the diff.
- **Pre-commit agent** — git hook that runs Merlin before every push: checks for secrets, TODOs, failing tests, obvious bugs. Fast, local, zero API cost, always on.
- **Background file watcher** — Merlin watches for file changes, auto-fixes obvious issues (imports, formatting, unused vars) without being asked.

## Goals

- Be the fastest, most reliable cheap worker a smart orchestrator can spawn
- Target tasks: grep/search, file read/write, running scripts, parallel sub-task execution — shallow, atomic, well-specified
- Instructions come from a smarter orchestrator (Claude, GPT-4, etc.) — not ambiguous human input
- Filesystem as memory: read inputs from files, write outputs to files, keep context window minimal
- Context window 6K — fits a large single Python file (~800 lines) plus agent protocol overhead.
- Maximize inference TPS and minimize memory footprint on Apple Silicon (MacBook)
- Target: 3B (MacBook, v1). Scale to 7B once pipeline is proven.
- Training: pre-train from scratch on code + agent protocol data → SFT on agentic traces (thinking stripped) → thinking fine-tune → RL on verifiable bash/fs task rewards — no distillation
- Thinking: Qwen thinks freely during trace generation (better decisions → higher success rate); thinking stripped before SFT. Re-introduced in thinking fine-tune phase. RL optimizes when/how much to think — thinking that doesn't improve task success gets penalized by the context it wastes. RL is the ablation.
- Custom tokenizer trained on Python + agent traces; special tokens for tool call protocol
- Custom kernels from the start — no relying on framework defaults
- MacBook target: 3B int4 ≈ 1.5GB weights, comfortable on any M-series Mac

## Corpus (~100B training tokens)

Sources: Python, Bash, Markdown (The Stack dedup), Stack Overflow Q&A, GitHub Issues/commits, man pages, tldr-pages, synthetic agent traces. No general web text, no Wikipedia — pure technical corpus.

| Source | Tokens | License |
|---|---|---|
| The Stack dedup — Python | 35B | permissive (license-filtered) |
| The Stack dedup — Shell/Bash | 3B | permissive |
| The Stack dedup — Markdown | 5B | permissive |
| Stack Overflow (bash/python/git/docker Q&A) | 8B | CC BY-SA 4.0 |
| GitHub Issues + commit messages | 3B | Apache 2.0 |
| Man pages + tldr-pages | 300M | open |
| Exercism / Rosetta Code | 500M | Apache 2.0 |
| Synthetic agent traces (Qwen2.5-Coder-32B, Docker-validated) | 5B | TBD |
| 2nd epoch replay (code + traces) | ~40B | — |

Two-phase curriculum: 80B general pre-training → 20B protocol-heavy warmup (trace-heavy mix).

## Data Pipeline

`data/pipeline/` — staged scripts:
- `00_download.py` — stream and filter sources to JSONL shards
- `01_filter.py` — per-source quality filtering: Python (AST parse + must have def/class), Bash (≥2 commands), Markdown (English-only, no templates)
- `02_dedup.py` — MinHash LSH deduplication
- `03_train_tokenizer.py` — BPE tokenizer in Rust (HF tokenizers), 32K vocab, 20 special tokens: `<|bos|>` `<|eos|>` + 18 agent protocol tokens (see `harness/protocol.py`). Fresh training produces vocab_size=32020. `--patch <tokenizer.json>` adds missing tokens to an existing tokenizer without retraining BPE.
- `04_fetch_repos.py` — two-step repo harvesting: `search` (GitHub API → candidates JSONL) + `scan` (clone + service-dep filter + pytest → passing JSONL)
- `05_tokenize.py` — apply tokenizer, pack into 6K-token chunks, write uint16 .bin files

**Agentic trace pipeline** (separate from pretraining corpus):
- Step 1 — `04_fetch_repos.py search` → `data/repos/candidates.jsonl` (target: 50k repos)
- Step 2 — `modal run modal_app.py::scan_repos` → `data/repos/passing.jsonl` (target: 20k passing)
- Step 3 — `modal run modal_app.py::generate_task_dataset` → `data/tasks/tasks.jsonl` + `data/repos/zips/` (20 mutations/repo; zips stored on Modal volume for trace gen)
- Step 4 — `modal run modal_app.py::generate_traces` (TBD) → `data/traces/traces.jsonl`

**Task dataset schema** (`data/tasks/tasks.jsonl`): `id`, `task_name`, `task_category`, `repo`, `repo_commit`, `repo_zip_key`, `repo_stars`, `repo_topics`, `repo_license`, `mutation_kind`, `mutation_description`, `mutated_file`, `mutation_lineno`, `mutation_patch`, `failing_tests`, `n_failing_tests`, `task`

**Trace dataset schema** (`data/traces/traces.jsonl`): task fields + `trace_sft`, `trace_full`, `n_tokens_sft`, `n_tokens_full`, `n_tool_calls`, `duration_s`

## Infrastructure

- **Persistent volume**: `merlin-data` Modal volume, mounted at `/data` — holds tokenized corpus, checkpoints, HF cache (`/data/cache/hf`), Inductor cache (`/data/cache/inductor`), repo zips (`/data/repos/zips/`).

## Stack

- **Training**: PyTorch + CUDA + Triton (NVIDIA)
- **Inference**: MLX (Apple Silicon / Metal) — MacBook
- **Bridge**: weight conversion + parity tests; PyTorch is source of truth

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Docs

Keep `docs/` up-to-date whenever anything changes:
- `docs/architecture.md` — model design and config table
- `docs/dataset.md` — corpus sources, filters, format, train/val split
- `docs/stack.md` — training/inference split rationale, planned work
- `docs/training.md` — hyperparams, data, observability
- `docs/inference.md` — benchmarks, kernels, quantization

Update benchmarks whenever a new optimisation is measured. Update planned-work sections when something ships.

## Status

| Milestone | Status | Notes |
|---|---|---|
| 1. Agentic trace research | ✅ Done | Research complete |
| 2. Agentic harness | ✅ Done | Protocol defined; 20 special tokens; feasibility harness at 47% on 49 tasks; LocalSandbox for Modal |
| 3a. Task dataset (CPU) | 🔄 In progress | Repo search + Modal scan working (227 passing from 2185 candidates); mutation pipeline built; target 20k repos × 20 mutations |
| 3b. Trace generation (GPU) | ⏸ Blocked on 3a | Qwen on vLLM; target ~200K successful traces |
| 4. Tokenizer training | ✅ Done (v0) | 32016 vocab; will retrain clean after traces |
| 5. Data pipeline | ✅ Done | 1.19B tokens; corpus_train.bin + corpus_val.bin on HF |
| 6. Benchmarking | ✅ Done | checkpoint + release eval integrated into train.py |
| 7. E2E experiment loop | ✅ Done | H100 via Modal; 330M model; baseline running |
| 8. Full corpus pipeline | ⏸ Not started | ~100B tokens; blocked on nothing — can run in parallel |
| 9. Full 3B run | ⏸ Not started | Blocked on milestone 8 |
| 10. Post-training | ⏸ Not started | SFT → RL → thinking fusion; blocked on 3b + 9 |
| 10a. SFT infra | ✅ Done | sft.py + 06_prepare_sft.py + Modal; validated on 330M with CodeAlpaca proxy; swap in real traces when ready |

**Data state:**
- `data/raw/` — 1M Python + 250K Bash + 200K Markdown + 7K tldr
- `data/filtered/` — 752K Python + 214K Bash + 142K Markdown + 7K tldr
- `data/deduped/` — same (exact dedup; Stack v1 pre-deduped by BigCode)
- `data/tokenizer/` — 32016-token BPE, `tokenizer.json` → [tsuberim/merlin-tokenizer-v0](https://huggingface.co/tsuberim/merlin-tokenizer-v0)
- `data/tokenized/` — `corpus_train.bin` (173,911 chunks) + `corpus_val.bin` (19,340 chunks); 6144 tokens each; 1.19B tokens; 100% packing efficiency → [tsuberim/merlin-corpus-v0](https://huggingface.co/datasets/tsuberim/merlin-corpus-v0)

## Milestones

1. **Agentic trace research** — research optimal strategy for generating high-quality agentic traces using Qwen2.5-Coder-32B; produce cost estimate for full 5B-token run with parallelized H100 fleet.

2. **Agentic harness** *(blocks trace generation)* — research and solidify the agent protocol (tool call format, special tokens, sandbox interface); implement sandbox infra reused for both trace generation and inference; this is the shared contract everything else builds on.

3a. **Task dataset** *(CPU, in progress)* — gather 50k candidate repos via GitHub search; scan on Modal CPU for passing pytest suites (target: 20k passing); generate 20 AST mutations per repo (6 mutation types); zip each repo and store on Modal volume; output `data/tasks/tasks.jsonl`. Schema includes `repo_commit`, `repo_zip_key`, `mutation_patch`, `test_snapshots` (for tamper-proof RL validation).

3b. **Trace generation** *(GPU, blocked by 3a)* — run Qwen2.5-Coder-32B on task dataset via vLLM; unzip repo + apply mutation patch per task; generate agentic traces; filter to successful; output `data/traces/traces.jsonl` with `trace_sft` + `trace_full`. Target: ~200K successful traces (~500M tokens).

4. **Tokenizer training** *(blocked by 3)* — train BPE tokenizer on final corpus including agentic traces; 32K vocab; special tokens for agent protocol; validate compression ratio on held-out traces.

5. **Data pipeline** — research and implement download + cleaning pipeline for pretraining corpus: Python, Bash, Markdown, and technical natural language. Staged scripts in `data/pipeline/`.

6. **Benchmarking** — research and implement staged benchmarks at three levels:
   - **Checkpoint** (every N steps, cheap): basic sanity checks first; abort the run early on failure
   - **Release** (end of run): full eval suite
   - **One-time** (new task/dataset only): expensive evals run once and cached
   Cost-aware: benchmark cost informs which checks run at which cadence.

7. **E2E experiment loop** — ability to launch a training run locally or remote using real corpus data (not blocked on agentic traces); auto-benchmark each checkpoint; iterate on model config and training recipe.

8. **Full corpus pipeline** *(blocks 9)* — scale data pipeline to ~100B tokens: parallelized shard downloads, distributed MinHash dedup, multiprocess tokenization. Target: corpus_train.bin + corpus_val.bin at full scale uploaded to HF.

9. **Full 3B run** *(blocked by 8)* — train the 3B model on the full corpus; multi-GPU DDP, long-running Modal job (~2 days on 4× H100). Produce a released checkpoint on HF.

10. **Post-training** *(blocked by 3 + 9)* — three-stage recipe on top of the pre-trained 3B checkpoint (see `research/post_training/README.md`):
    - **SFT** — 70/30 mix of validated agentic traces + code-flavored instruction-following data; response-only loss.
    - **RL** — GRPO on binary Docker sandbox rewards; curriculum single-step → multi-step; veRL on Modal. Target: >70% pass@1 (vs. 47% feasibility baseline).
    - **Thinking fusion SFT** — conditional; comes *after* RL (SAIL-RL, 2025). Abort if pass@1 on hard tasks improves <10%.

## Research Ideas

- **Importance sampling** — per-chunk grad-norm EMA as a self-weighting dataset mechanism; validate on 330M experiment model vs baseline; if promising, apply to 3B run with EMA warm-started from experiment run. Potentially publish.

## Research

At the start of each conversation, read `research/index.md` to load context on prior research.

## Non-goals

- iPhone / CoreML deployment
- Training on MPS
- Maximizing general benchmark scores over efficiency
- General world knowledge, reasoning, or broad language understanding
- Long-context capability (summarization of long docs, RAG over large corpora)
- Knowledge distillation from a teacher model
- Training data generated by proprietary models (OpenAI, Anthropic ToS prohibit use in publicly released models)
