#!/usr/bin/env python3
"""
Merlin REPL — play with a trained checkpoint locally via MLX.

Usage:
    python repl.py                          # latest checkpoint (auto-detected from HF)
    python repl.py --tag <tag>              # specific checkpoint
    python repl.py --model 3b              # different model size
    python repl.py --weights path/to/w.npz  # pre-converted weights
"""

import argparse
import os
import readline  # noqa: F401 — enables up-arrow history in input()
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from merlin import tok as _tok
from merlin.infer import Config, GPT, load_model

HF_MODEL_REPO  = os.environ.get("HF_REPO", "tsuberim/merlin")
WEIGHTS_CACHE  = Path.home() / ".cache" / "merlin" / "weights"

# ── weight conversion ─────────────────────────────────────────────────────────

def _convert(pt_path: Path, out_path: Path):
    """Convert PyTorch checkpoint → MLX .npz, streaming one tensor at a time.

    Peak memory = checkpoint size only (~6 GB for 3B bf16).
    Each tensor is written to the zip and freed before the next is loaded.
    """
    import io, zipfile, torch
    print(f"converting {pt_path.name} → {out_path.name} ...")
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    sd   = ckpt["model"]
    del ckpt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(out_path), "w", compression=zipfile.ZIP_STORED) as zf:
        for k, v in sd.items():
            k = k.removeprefix("_orig_mod.")
            if "rope_cos" in k or "rope_sin" in k:
                continue
            buf = io.BytesIO()
            np.lib.format.write_array(buf, v.half().numpy())
            zf.writestr(k + ".npy", buf.getvalue())
    print(f"  saved {out_path}")


def _hf_blob_sha(hf_path: str) -> str:
    """Return the current blob SHA for a file in HF_MODEL_REPO."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.get_paths_info(HF_MODEL_REPO, [hf_path], repo_type="model")
    return info[0].blob_id


def latest_tag() -> str:
    """Return the tag of the most recently pushed checkpoint on HF."""
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    all_files = list(api.list_repo_files(HF_MODEL_REPO, repo_type="model"))
    ckpt_paths = [f for f in all_files if f.startswith("checkpoints/ckpt_") and f.endswith(".pt")]
    if not ckpt_paths:
        raise RuntimeError(f"No checkpoints found in {HF_MODEL_REPO}")
    infos = list(api.get_paths_info(HF_MODEL_REPO, ckpt_paths, repo_type="model"))
    dated = [f for f in infos if f.last_commit is not None]
    latest = max(dated, key=lambda f: f.last_commit.date) if dated else infos[-1]
    return latest.path.removeprefix("checkpoints/ckpt_").removesuffix(".pt")


def resolve_weights(tag: str) -> Path:
    """Return path to .npz weights, downloading + converting as needed.

    Always checks HF blob SHA; re-downloads only if it changed since last cache.
    """
    WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)
    npz_path = WEIGHTS_CACHE / f"{tag}.npz"
    sha_path = WEIGHTS_CACHE / f"{tag}.sha"
    hf_file  = f"checkpoints/ckpt_{tag}.pt"

    blob_sha = _hf_blob_sha(hf_file)
    if npz_path.exists() and sha_path.exists() and sha_path.read_text().strip() == blob_sha:
        return npz_path

    from huggingface_hub import hf_hub_download
    import shutil
    print(f"downloading checkpoint {tag} from {HF_MODEL_REPO} ...")
    hf_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename=hf_file,
        token=os.environ.get("HF_TOKEN"),
        force_download=True,
    )
    pt_path = WEIGHTS_CACHE / f"{tag}.pt"
    shutil.copy(hf_path, pt_path)
    _convert(pt_path, npz_path)
    pt_path.unlink()  # don't keep the .pt after conversion
    sha_path.write_text(blob_sha)

    # evict oldest checkpoints, keep 3 most recent
    npzs = sorted(WEIGHTS_CACHE.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in npzs[3:]:
        old.unlink(missing_ok=True)
        old.with_suffix(".sha").unlink(missing_ok=True)

    return npz_path


# ── generation ────────────────────────────────────────────────────────────────

def _sample(logits: mx.array, temperature: float) -> mx.array:
    if temperature == 0:
        return mx.argmax(logits, axis=-1, keepdims=True)
    return mx.random.categorical(logits / temperature)[..., None]


def stream_generate(model: GPT, idx: mx.array, max_new: int, temperature: float):
    """Yield tokens one at a time for streaming output."""
    # allocate only what we need: avoids 1.7GB KV for block_size=4096 on a 7B model
    cache = model.make_cache(max_T=idx.shape[1] + max_new, batch_size=idx.shape[0])
    logits, cache = model(idx, cache)
    next_tok = _sample(logits[:, -1, :], temperature)
    mx.eval(next_tok, *[c.k for c in cache], *[c.v for c in cache])
    yield next_tok[0, 0].item()

    for _ in range(max_new - 1):
        logits, cache = model(next_tok, cache)
        next_tok = _sample(logits[:, -1, :], temperature)
        mx.eval(next_tok)
        yield next_tok[0, 0].item()


# ── REPL ──────────────────────────────────────────────────────────────────────

BANNER = """
merlin repl  (ctrl-d or /quit to exit, /temp <value>, /batch <n>)
"""

def run_repl(model: GPT, enc, max_new: int, temperature: float, batch_size: int = 1):
    print(BANNER)
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user == "/quit":
            break
        if user.startswith("/temp"):
            parts = user.split()
            if len(parts) == 2:
                try:
                    temperature = float(parts[1])
                    print(f"(temperature: {temperature})")
                except ValueError:
                    print("usage: /temp <float>")
            else:
                print(f"(temperature: {temperature})")
            continue
        if user.startswith("/batch"):
            parts = user.split()
            if len(parts) == 2:
                try:
                    batch_size = int(parts[1])
                    print(f"(batch_size: {batch_size})")
                except ValueError:
                    print("usage: /batch <int>")
            else:
                print(f"(batch_size: {batch_size})")
            continue
        if not user:
            continue

        prompt_ids = enc.encode(user)

        # trim to leave room for generated tokens
        max_prompt = model.cfg.block_size - max_new
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        idx = mx.array([prompt_ids] * batch_size)
        # move cursor up to end of input line; emit the \n we appended to prompt
        col = len("> ") + len(user) + 1
        print(f"\033[1A\033[{col}G", end="", flush=True)

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

        print(f"\n\033[2m[{n_new} tokens  {tps:.1f} t/s  {mem_mb:.0f} MB]\033[0m\n")


# ── entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(prog="merlin")
    parser.add_argument("--model",       choices=["sanity", "experiment", "3b", "7b"],
                        default="experiment")
    parser.add_argument("--tag",         default=None, help="checkpoint tag (default: latest on HF)")
    parser.add_argument("--weights",     default=None, help="path to pre-converted .npz")
    parser.add_argument("--max_new",     type=int,   default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--batch_size",  type=int,   default=1, help="batch size for TPS benchmarking (duplicates input)")
    parser.add_argument("--bits",        type=int,   default=16, choices=[0, 4, 8, 16])
    parser.add_argument("--random",      action="store_true", help="use random weights (for benchmarking)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    cfg = {"sanity": Config.sanity, "experiment": Config.experiment,
           "3b": Config.b3, "7b": Config.b7}[args.model]()
    cfg.vocab_size = 32016  # our tokenizer

    enc = _tok.load()

    # resolve tag before deciding load path
    tag = None
    if not args.random and not args.weights:
        if args.tag:
            tag = args.tag
        elif args.model == "experiment":
            tag = latest_tag()
        else:
            print(f"no checkpoint for --model {args.model}, using random weights")
            args.random = True

    if args.random:
        from mlx.nn import quantize
        model = GPT(cfg)
        dtype = {16: mx.float16, 0: mx.float32}.get(args.bits, mx.float32)
        if dtype != mx.float32:
            model.load_weights([(k, v.astype(dtype)) for k, v in tree_flatten(model.parameters())])
        mx.eval(model.parameters())
        if args.bits in (4, 8):
            quantize(model, group_size=64, bits=args.bits)
        for block in model.blocks:
            block.mlp = mx.compile(block.mlp)
        print("using random weights")
    elif args.weights:
        print(f"loading {args.weights} ...")
        model = load_model(args.weights, cfg, bits=args.bits)
    else:
        print(f"checkpoint: {tag}")
        weights_path = str(resolve_weights(tag))
        print(f"loading {weights_path} ...")
        model = load_model(weights_path, cfg, bits=args.bits)

    nparams = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"model: {args.model}  params: {nparams:,}")

    print("warming up...", end="", flush=True)
    _w = mx.zeros((1, 1), dtype=mx.uint32)
    _wc = model.make_cache(max_T=args.max_new + 1, batch_size=1)
    model(_w, _wc)
    mx.eval(model.parameters())
    del _w, _wc
    print(" done")

    run_repl(model, enc, args.max_new, args.temperature, args.batch_size)


if __name__ == "__main__":
    main()
