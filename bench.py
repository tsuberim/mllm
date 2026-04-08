"""Measure inference TPS, peak memory, and avg memory."""
import time
import argparse
import mlx.core as mx
import tiktoken

from infer import load_model, Config


def bench_once(model, prompt: str, n_tokens: int) -> dict:
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    idx = mx.array([tokens])
    mx.eval(idx)

    # warmup
    _ = model(idx)
    mx.eval()

    mx.reset_peak_memory()
    mem_samples = []

    # prefill
    cache = model.make_cache()
    logits, cache = model(idx, cache)
    next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
    idx = mx.concatenate([idx, next_tok], axis=1)
    mx.eval(idx, *[c.k for c in cache], *[c.v for c in cache])

    t0 = time.perf_counter()

    for _ in range(n_tokens):
        logits, cache = model(next_tok, cache)
        next_tok = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        idx = mx.concatenate([idx, next_tok], axis=1)
        mx.eval(idx, *[c.k for c in cache], *[c.v for c in cache])
        mem_samples.append(mx.get_active_memory())

    elapsed = time.perf_counter() - t0
    return {
        "tps":      n_tokens / elapsed,
        "peak_mb":  mx.get_peak_memory() / 1024 ** 2,
        "avg_mb":   (sum(mem_samples) / len(mem_samples)) / 1024 ** 2,
    }


def bench(weights_path: str, cfg: Config, prompt: str, n_tokens: int,
          bits: int = 0, runs: int = 3):
    model = load_model(weights_path, cfg, bits=bits)

    results = [bench_once(model, prompt, n_tokens) for _ in range(runs)]

    tps_vals = [r["tps"] for r in results]
    avg_tps  = sum(tps_vals) / runs
    std_tps  = (sum((t - avg_tps) ** 2 for t in tps_vals) / runs) ** 0.5

    quant = f"int{bits}" if bits else "fp32"
    print(f"model     : {args.model} ({quant})")
    print(f"tokens    : {n_tokens}  runs: {runs}")
    print(f"tps       : {avg_tps:.1f} ± {std_tps:.1f}")
    print(f"peak mem  : {results[-1]['peak_mb']:.1f} MB")
    print(f"avg mem   : {results[-1]['avg_mb']:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["tiny", "medium", "base"], default="base")
    parser.add_argument("--weights",  default="base_random.npz")
    parser.add_argument("--prompt",   default="Once upon a time")
    parser.add_argument("--n_tokens", type=int, default=500)
    parser.add_argument("--bits",     type=int, default=0, choices=[0, 4, 8],
                        help="quantization bits (0 = none)")
    parser.add_argument("--runs",     type=int, default=3,
                        help="number of timed runs to average")
    args = parser.parse_args()

    cfg = {"tiny": Config.tiny, "medium": Config.medium, "base": Config.base}[args.model]()
    bench(args.weights, cfg, args.prompt, args.n_tokens, bits=args.bits, runs=args.runs)
