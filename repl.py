#!/usr/bin/env python3
"""
Merlin REPL — play with a trained checkpoint locally via MLX.

Usage:
    python repl.py                          # latest checkpoint for experiment model
    python repl.py --tag <commit>           # specific run
    python repl.py --model iphone --tag ... # different model size
    python repl.py --weights path/to/w.npz  # pre-converted weights
"""

import argparse
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

import tok as _tok
from infer import Config, GPT, load_model

HF_MODEL_REPO  = os.environ.get("HF_REPO", "tsuberim/merlin")
WEIGHTS_CACHE  = Path("checkpoints/weights")   # local cache for converted .npz files

# ── weight conversion ─────────────────────────────────────────────────────────

def _convert(pt_path: Path, out_path: Path):
    """Convert PyTorch checkpoint → MLX .npz (fp32 numpy arrays)."""
    import torch
    print(f"converting {pt_path.name} → {out_path.name} ...")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"]

    arrays = {}
    for k, v in sd.items():
        k = k.removeprefix("_orig_mod.")
        if "rope_cos" in k or "rope_sin" in k:
            continue
        arrays[k] = v.float().numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **arrays)
    print(f"  saved {out_path}")


def resolve_weights(tag: str, model: str) -> Path:
    """Return path to .npz weights, downloading + converting as needed."""
    WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    npz_path = WEIGHTS_CACHE / f"{tag}.npz"
    if npz_path.exists():
        return npz_path

    # download PyTorch checkpoint from HF
    pt_path = WEIGHTS_CACHE / f"{tag}.pt"
    if not pt_path.exists():
        from huggingface_hub import hf_hub_download
        print(f"downloading checkpoint {tag} from {HF_MODEL_REPO} ...")
        hf_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=f"checkpoints/ckpt_{tag}.pt",
            token=os.environ.get("HF_TOKEN"),
        )
        import shutil
        shutil.copy(hf_path, pt_path)

    _convert(pt_path, npz_path)
    return npz_path


# ── generation ────────────────────────────────────────────────────────────────

def stream_generate(model: GPT, idx: mx.array, max_new: int, temperature: float):
    """Yield tokens one at a time for streaming output."""
    cache = model.make_cache()
    logits, cache = model(idx, cache)
    next_tok = mx.argmax(logits[:, -1, :] / temperature, axis=-1, keepdims=True)
    mx.eval(next_tok, *[c.k for c in cache], *[c.v for c in cache])
    yield next_tok[0, 0].item()

    for _ in range(max_new - 1):
        logits, cache = model(next_tok, cache)
        next_tok = mx.argmax(logits[:, -1, :] / temperature, axis=-1, keepdims=True)
        mx.eval(next_tok)
        yield next_tok[0, 0].item()


# ── REPL ──────────────────────────────────────────────────────────────────────

BANNER = """
merlin repl  (ctrl-d or /quit to exit, /clear to reset context)
"""

def run_repl(model: GPT, enc, max_new: int, temperature: float):
    print(BANNER)
    context = ""  # accumulate conversation as plain text
    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user == "/quit":
            break
        if user == "/clear":
            context = ""
            print("(context cleared)")
            continue
        if not user:
            continue

        context += user + "\n"
        prompt_ids = enc.encode(context)

        # trim to leave room for generated tokens
        max_prompt = model.cfg.block_size - max_new
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        idx = mx.array([prompt_ids])
        print("<<< ", end="", flush=True)

        completion_ids = []
        eos = enc.token_to_id("<|eos|>")
        t0 = None
        try:
            for tok_id in stream_generate(model, idx, max_new, temperature):
                if t0 is None:
                    t0 = __import__("time").perf_counter()  # start after prefill
                token_str = enc.decode([tok_id])
                print(token_str, end="", flush=True)
                completion_ids.append(tok_id)
                if eos is not None and tok_id == eos:
                    break
        except KeyboardInterrupt:
            pass

        elapsed = __import__("time").perf_counter() - (t0 or __import__("time").perf_counter())
        n_new   = max(len(completion_ids) - 1, 1)  # exclude prefill token
        tps     = n_new / elapsed if elapsed > 0 else 0
        mem_mb  = mx.get_active_memory() / 1e6

        print(f"\n\033[2m[{n_new} tokens  {tps:.1f} t/s  {mem_mb:.0f} MB]\033[0m")
        context += enc.decode(completion_ids) + "\n"


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=["sanity", "experiment", "iphone", "macbook"],
                        default="experiment")
    parser.add_argument("--tag",         default=None, help="commit tag (default: latest ckpt)")
    parser.add_argument("--weights",     default=None, help="path to pre-converted .npz")
    parser.add_argument("--max_new",     type=int,   default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--bits",        type=int,   default=0, choices=[0, 4, 8, 16])
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    cfg = {"sanity": Config.sanity, "experiment": Config.experiment,
           "iphone": Config.iphone, "macbook": Config.macbook}[args.model]()
    cfg.vocab_size = 32016  # our tokenizer

    if args.weights:
        weights_path = args.weights
    elif args.tag:
        weights_path = str(resolve_weights(args.tag, args.model))
    else:
        parser.error("provide --tag <commit> or --weights <path>")

    enc = _tok.load()

    print(f"loading {weights_path} ...")
    model = load_model(weights_path, cfg, bits=args.bits)
    from mlx.utils import tree_flatten
    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"model: {args.model}  params: {nparams:,}")

    run_repl(model, enc, args.max_new, args.temperature)
