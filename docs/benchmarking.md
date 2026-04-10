# Benchmarking

Three cadences, each with a different cost/depth tradeoff.

## Checkpoint eval — every N steps, during training

**File:** `eval_ckpt.py`, called from `train.py` via `--eval_every`

Runs on the PyTorch model in-place (CUDA). Two checks:

| Check | Method | Abort condition |
|---|---|---|
| Coherence | Generate 50 tokens for 2 prompts; reject if output < 10 tokens or >80% same token | Any step |
| HumanEval-5 | pass@1 on 5 fixed problems, greedy decoding | 0% after step 1000 |

Greedy decoding for consistency across checkpoints — this is a collapse detector, not a quality measure. Results logged to wandb as `eval/coherence` and `eval/humaneval_5`.

HumanEval dataset downloaded once at startup via `eval_ckpt.preload()` and cached at `/workspace/hf_cache` (persistent volume).

## Release eval — end of run, on MacBook

**File:** `eval.py`

Runs on the converted MLX model. By default samples 20 problems per benchmark for a ~20 min run; `--full` runs the complete suites (~60 min).

| Stage | Benchmark | Method | Floor (early exit) |
|---|---|---|---|
| 0 | Coherence | Same as above | Fail → stop |
| 1 | HumanEval | pass@1, temp=0.2, 20 or 164 problems | < 5% → stop |
| 2 | MBPP | pass@1, temp=0.2, 20 or 500 problems | < 5% → stop |
| 3 | Perf | tok/s + peak RAM @ Q4 on M-series | — |
| 4 | Agent harness | 53 agentic tasks via `research/feasibility/` | < 15% (opt-in: `--agent`) |

temp=0.2 matches the HumanEval paper convention — release numbers are comparable to published leaderboard figures.

Stage 4 requires an instruction-tuned model in mlx_lm format. Not applicable until post-SFT.

## One-time eval — new architecture or dataset only

Not yet implemented. Intended for SWE-bench Lite (300 real GitHub issues) and BFCL (tool-call format correctness). Run once and results cached — too expensive for routine use.

## What we don't use and why

| Benchmark | Reason skipped |
|---|---|
| HellaSwag / ARC-Challenge | Commonsense NLI — not in training domain; low scores are expected and uninformative |
| GSM8K | Math reasoning proxy — not relevant for an agentic coding worker |
| MT-Bench | Chat assistant quality, LLM-judged ($10–20/run) — wrong use case |
| BoolQ | Saturated at 1B+, no signal |
| MMLU | Reported for external comparability only; small-model scores contaminated |

## Competitive targets

Based on `research/benchmarks/README.md`:

- **HumanEval:** beat SmolLM2 1.7B (no published score) and Qwen2.5 1.5B (37.2%) at our parameter count
- **MBPP:** same targets
- **Tok/s @ Q4:** beat Llama 3.2 3B on equivalent M-series hardware (25–35 tok/s on M1 Pro)
- **Peak RAM @ Q4:** stay under 2 GB (3B int4 ≈ 1.5 GB)
