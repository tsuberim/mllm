"""
Lightweight checkpoint eval for use inside the training loop.
Works directly with the PyTorch model — no MLX, no weight conversion.

Checks two things:
  1. coherence     — model generates non-garbage text (binary)
  2. humaneval_5   — pass@1 on 5 sampled HumanEval problems (0–100%)

Aborts the run (sys.exit(1)) if:
  - coherence fails at any step
  - humaneval_5 == 0% after step 1000 (early training legitimately scores 0)

Call eval_ckpt.preload() at training startup to avoid a mid-run HF download.
"""

import os
import subprocess
import sys
import tempfile

import torch

N_HUMANEVAL   = 5
ABORT_AFTER   = 1000   # don't abort on 0% humaneval before this step

_ds = None  # loaded once via preload()


def preload():
    global _ds
    from datasets import load_dataset
    _ds = load_dataset("openai/openai_humaneval", split="test")
    _ds = _ds.select(range(N_HUMANEVAL))


def _gen(model, enc, cfg, device, autocast, prompt: str, max_new: int) -> str:
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], device=device)
    with torch.no_grad(), autocast:
        for _ in range(max_new):
            logits, _ = model(idx[:, -cfg.block_size:])
            next_tok = logits[:, -1, :].argmax(-1, keepdim=True)  # [B, 1]
            idx = torch.cat([idx, next_tok], dim=1)
    return enc.decode(idx[0].tolist()[len(tokens):])


def _exec(code: str, timeout: int = 10) -> bool:
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


_COHERENCE_PROMPTS = [
    "def fibonacci(n):\n    ",
    "import os\n# list all .py files\n",
]


def run(model, enc, cfg, device, autocast, step: int) -> dict:
    """
    Run checkpoint eval. Returns metrics dict for wandb.log().
    Calls sys.exit(1) on collapse.
    """
    model.eval()
    metrics = {}

    # ── coherence ──────────────────────────────────────────────────────────────
    coherent = True
    for prompt in _COHERENCE_PROMPTS:
        out = _gen(model, enc, cfg, device, autocast, prompt, max_new=50)
        toks = enc.encode(out)
        if len(toks) < 10 or max(toks.count(t) for t in set(toks)) / len(toks) > 0.8:
            coherent = False
            break
    metrics["eval/coherence"] = int(coherent)
    if not coherent:
        print(f"\n[eval step={step}] coherence FAIL — aborting run")
        sys.exit(1)

    # ── humaneval mini ─────────────────────────────────────────────────────────
    if _ds is None:
        preload()
    passed = 0
    results = []
    for ex in _ds:
        completion = _gen(model, enc, cfg, device, autocast, ex["prompt"], max_new=256)
        code = ex["prompt"] + completion + "\n\n" + ex["test"] + f"\ncheck({ex['entry_point']})"
        ok = _exec(code)
        passed += ok
        results.append("." if ok else "F")
    score = passed / N_HUMANEVAL
    metrics["eval/humaneval_5"] = score
    print(f"  [eval step={step}]  coherence=OK  humaneval_5={''.join(results)}  {score:.0%} ({passed}/{N_HUMANEVAL})")

    if passed == 0 and step >= ABORT_AFTER:
        print(f"\n[eval step={step}] humaneval_5=0% past step {ABORT_AFTER} — aborting run")
        sys.exit(1)

    return metrics
