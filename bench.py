"""Measure inference TPS and peak Metal memory."""
import time
import argparse
import mlx.core as mx
import tiktoken

from infer import load_model, Config


def bench(weights_path: str, cfg: Config, prompt: str, n_tokens: int):
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(weights_path, cfg)

    tokens = enc.encode(prompt)
    idx = mx.array([tokens])
    mx.eval(idx)

    # warmup
    _ = model(idx)
    mx.eval()

    mx.metal.reset_peak_memory()
    t0 = time.perf_counter()

    for _ in range(n_tokens):
        idx_cond = idx[:, -cfg.block_size:]
        logits = model(idx_cond)[:, -1, :]
        next_tok = mx.argmax(logits, axis=-1, keepdims=True)
        idx = mx.concatenate([idx, next_tok], axis=1)
        mx.eval(idx)

    elapsed = time.perf_counter() - t0
    peak_mb = mx.metal.get_peak_memory() / 1024 ** 2

    tps = n_tokens / elapsed
    print(f"model     : {args.model}")
    print(f"tokens    : {n_tokens}")
    print(f"time      : {elapsed:.2f}s")
    print(f"tps       : {tps:.1f}")
    print(f"peak mem  : {peak_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["tiny", "base"], default="base")
    parser.add_argument("--weights",  default="weights.npz")
    parser.add_argument("--prompt",   default="Once upon a time")
    parser.add_argument("--n_tokens", type=int, default=100)
    args = parser.parse_args()

    cfg = Config.tiny() if args.model == "tiny" else Config.base()
    bench(args.weights, cfg, args.prompt, args.n_tokens)
