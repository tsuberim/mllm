"""
Accuracy + perf eval pipeline with early exit.

Stages (cheapest → most expensive):
  0  coherence   — non-garbage generation             (~30s)
  1  humaneval   — pass@1, Python function completion  (~5 min, 20 sampled)
  2  mbpp        — pass@1, Python function completion  (~5 min, 20 sampled)
  3  perf        — tok/s + peak RAM                   (~1 min)
  4  agent       — feasibility harness, 53 tasks      (~30 min, --agent)

Exits non-zero if any stage falls below its floor threshold.
Stage 4 requires an instruction-tuned model loadable by mlx_lm (post-SFT only).

Usage:
    python eval.py --model iphone --weights checkpoints/weights.npz
    python eval.py --model iphone --weights checkpoints/weights.npz --bits 4
    python eval.py --model iphone --weights checkpoints/weights.npz --full --agent
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

import mlx.core as mx
from datasets import load_dataset

import tok as _tok
from infer import load_model, Config

# ─── thresholds ───────────────────────────────────────────────────────────────

FLOOR_HUMANEVAL = 0.05   # < 5% → model can't write Python yet
FLOOR_MBPP      = 0.05   # < 5% → same signal, different distribution
FLOOR_AGENT     = 0.15   # < 15% → agent protocol not learned

# ─── generation ───────────────────────────────────────────────────────────────

def gen(model, enc, prompt: str, max_new: int = 256, temp: float = 0.2) -> str:
    tokens = enc.encode(prompt)
    idx = mx.array([tokens])
    out = model.generate(idx, max_new, temperature=temp)
    mx.eval(out)
    return enc.decode(out[0].tolist()[len(tokens):])


# ─── stage 0: coherence ───────────────────────────────────────────────────────

_COHERENCE_PROMPTS = [
    "def fibonacci(n):\n    ",
    "import os\n# list all .py files recursively\n",
    "# run grep to find TODO comments\nresult = ",
]

def run_coherence(model, enc) -> bool:
    for prompt in _COHERENCE_PROMPTS:
        out = gen(model, enc, prompt, max_new=50)
        toks = enc.encode(out)
        if len(toks) < 10:
            return False  # output too short / crashed
        # repetition collapse: >80% same token
        if max(toks.count(t) for t in set(toks)) / len(toks) > 0.8:
            return False
    return True


# ─── stage 1+2: code eval (HumanEval / MBPP) ─────────────────────────────────

def _exec(code: str, timeout: int = 10) -> bool:
    """Execute code in a subprocess; return True if exit code == 0."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        fname = f.name
        f.write(code)
    try:
        r = subprocess.run(["python3", fname], capture_output=True, timeout=timeout)
        return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(fname)


def run_humaneval(model, enc, n: int) -> float:
    ds = load_dataset("openai/openai_humaneval", split="test", trust_remote_code=True)
    ds = ds.select(range(min(n, len(ds))))
    passed = 0
    for i, ex in enumerate(ds):
        completion = gen(model, enc, ex["prompt"], max_new=256)
        code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\ncheck({ex['entry_point']})"
        ok = _exec(code)
        passed += ok
        print(f"  {'.' if ok else 'F'}", end="", flush=True)
    print(f"  {passed}/{len(ds)}")
    return passed / len(ds)


def run_mbpp(model, enc, n: int) -> float:
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test", trust_remote_code=True)
    ds = ds.select(range(min(n, len(ds))))
    passed = 0
    for ex in ds:
        stub = next((ln + "\n    " for ln in ex["code"].splitlines() if ln.startswith("def ")), "")
        prompt = f"# Task: {ex['text']}\n{stub}"
        completion = gen(model, enc, prompt, max_new=256)
        setup = ex.get("test_setup_code", "") or ""
        tests = "\n".join(ex["test_list"])
        ok = _exec(setup + "\n" + prompt + completion + "\n\n" + tests)
        passed += ok
        print(f"  {'.' if ok else 'F'}", end="", flush=True)
    print(f"  {passed}/{len(ds)}")
    return passed / len(ds)


# ─── stage 3: perf ────────────────────────────────────────────────────────────

@dataclass
class PerfResult:
    tps: float
    peak_mb: float

def run_perf(model, enc, n_tokens: int = 200) -> PerfResult:
    prompt = "def quicksort(arr):\n    "
    tokens = enc.encode(prompt)
    idx = mx.array([tokens])
    # warmup
    model.generate(idx, 10); mx.eval()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    out = model.generate(idx, n_tokens)
    mx.eval(out)
    return PerfResult(
        tps=n_tokens / (time.perf_counter() - t0),
        peak_mb=mx.get_peak_memory() / 1024 ** 2,
    )


# ─── stage 4: agent harness (opt-in, post-SFT) ───────────────────────────────

def run_agent(mlx_model_id: str) -> float | None:
    """
    Runs the feasibility harness via mlx_lm.
    Requires an instruction-tuned model in mlx-community HF format.
    The custom weights/tokenizer are not compatible with this harness — call
    this only after converting + uploading a fine-tuned checkpoint.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "research/feasibility"))
    try:
        from mlx_lm import load
        from harness import run_tasks
        from tasks import TASKS
    except ImportError as e:
        print(f"  [skip] {e}")
        return None

    model, tokenizer = load(mlx_model_id)
    results = run_tasks(TASKS, model, tokenizer)
    return sum(r.success for r in results) / len(results)


# ─── output helpers ───────────────────────────────────────────────────────────

def _fmt(label: str, value: str, ok: bool | None = None, note: str = "") -> None:
    mark = {True: "\033[32m✓\033[0m", False: "\033[31m✗\033[0m", None: " "}[ok]
    print(f"  {mark} {label:<16} {value}  {note}")

def _bail(msg: str) -> None:
    print(f"\n\033[31m{msg}\033[0m")
    sys.exit(1)


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   choices=["sanity", "experiment", "iphone", "macbook"], default="iphone")
    parser.add_argument("--weights", default="checkpoints/weights.npz")
    parser.add_argument("--bits",    type=int, default=0, choices=[0, 4, 8, 16])
    parser.add_argument("--full",    action="store_true", help="Full eval: 164 HumanEval + 500 MBPP problems")
    parser.add_argument("--agent",   action="store_true", help="Run agent harness (post-SFT; requires --agent-model)")
    parser.add_argument("--agent-model", default="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
                        help="mlx_lm-compatible model ID for the agent stage")
    args = parser.parse_args()

    n_he   = 164 if args.full else 20
    n_mbpp = 500 if args.full else 20
    cfg    = {"sanity": Config.sanity, "experiment": Config.experiment,
              "iphone": Config.iphone, "macbook": Config.macbook}[args.model]()

    print(f"model    {args.weights}  ({args.model}, bits={args.bits})")
    print(f"problems humaneval={n_he}  mbpp={n_mbpp}\n")
    model = load_model(args.weights, cfg, bits=args.bits)
    enc   = _tok.load()

    # ── stage 0 ──
    print("stage 0 · coherence")
    ok = run_coherence(model, enc)
    _fmt("coherence", "pass" if ok else "fail", ok)
    if not ok:
        _bail("model is producing garbage — not converged yet")
    print()

    # ── stage 1 ──
    print(f"stage 1 · humaneval  (n={n_he})")
    he = run_humaneval(model, enc, n=n_he)
    _fmt("pass@1", f"{he:.1%}", he >= FLOOR_HUMANEVAL, f"(floor {FLOOR_HUMANEVAL:.0%})")
    if he < FLOOR_HUMANEVAL:
        _bail("below floor on HumanEval — can't write basic Python yet")
    print()

    # ── stage 2 ──
    print(f"stage 2 · mbpp  (n={n_mbpp})")
    mbpp = run_mbpp(model, enc, n=n_mbpp)
    _fmt("pass@1", f"{mbpp:.1%}", mbpp >= FLOOR_MBPP, f"(floor {FLOOR_MBPP:.0%})")
    if mbpp < FLOOR_MBPP:
        _bail("below floor on MBPP")
    print()

    # ── stage 3 ──
    print("stage 3 · perf")
    perf = run_perf(model, enc)
    _fmt("tok/s",    f"{perf.tps:.1f}")
    _fmt("peak RAM", f"{perf.peak_mb:.0f} MB")
    print()

    # ── stage 4 (opt-in) ──
    if args.agent:
        print(f"stage 4 · agent harness  ({args.agent_model})")
        agent_score = run_agent(args.agent_model)
        if agent_score is not None:
            ok = agent_score >= FLOOR_AGENT
            _fmt("pass rate", f"{agent_score:.1%}", ok, f"(floor {FLOOR_AGENT:.0%})")
        print()

    print("done.")


if __name__ == "__main__":
    main()
