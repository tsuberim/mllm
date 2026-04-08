"""Measure inference TPS, peak memory, and avg memory."""
import time
import argparse
import mlx.core as mx
import tiktoken

from infer import load_model, Config


def bench(weights_path: str, cfg: Config, prompt: str, n_tokens: int, bits: int = 0):
    enc = tiktoken.get_encoding("gpt2")
    model = load_model(weights_path, cfg, bits=bits)

    tokens = enc.encode(prompt)
    idx = mx.array([tokens])
    mx.eval(idx)

    # warmup
    _ = model(idx)
    mx.eval()

    mx.reset_peak_memory()
    mem_samples = []

    # prefill
    logits, cache = model(idx)
    next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    idx = mx.concatenate([idx, next_tok], axis=1)
    mx.eval(idx, *[t for k, v in cache for t in (k, v)])

    t0 = time.perf_counter()

    for _ in range(n_tokens):
        logits, cache = model(next_tok, cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        idx = mx.concatenate([idx, next_tok], axis=1)
        mx.eval(idx, *[t for k, v in cache for t in (k, v)])
        mem_samples.append(mx.get_active_memory())

    elapsed = time.perf_counter() - t0
    peak_mb = mx.get_peak_memory() / 1024 ** 2
    avg_mb  = (sum(mem_samples) / len(mem_samples)) / 1024 ** 2

    quant = f"int{bits}" if bits else "fp32"
    print(f"model     : {args.model} ({quant})")
    print(f"tokens    : {n_tokens}")
    print(f"time      : {elapsed:.2f}s")
    print(f"tps       : {n_tokens / elapsed:.1f}")
    print(f"peak mem  : {peak_mb:.1f} MB")
    print(f"avg mem   : {avg_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["tiny", "medium", "base"], default="base")
    parser.add_argument("--weights",  default="base_random.npz")
    parser.add_argument("--prompt",   default="Once upon a time")
    parser.add_argument("--n_tokens", type=int, default=500)
    parser.add_argument("--bits",     type=int, default=0, choices=[0, 4, 8],
                        help="quantization bits (0 = none)")
    args = parser.parse_args()

    cfg = {"tiny": Config.tiny, "medium": Config.medium, "base": Config.base}[args.model]()
    bench(args.weights, cfg, args.prompt, args.n_tokens, bits=args.bits)
