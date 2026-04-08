import sys
import mlx.core as mx
import mlx.nn as nn
import tiktoken
from dataclasses import dataclass


# ── custom kernel: fused SwiGLU ───────────────────────────────────────────────
# Reads gate + up once, writes silu(gate)*up once — saves one memory round-trip
# vs. computing silu and multiply as separate ops.

_swiglu_kernel = mx.fast.metal_kernel(
    name="swiglu",
    input_names=["gate", "up"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        T g = gate[elem];
        T u = up[elem];
        T sig = T(1) / (T(1) + metal::exp(-g));
        out[elem] = g * sig * u;
    """,
)

def swiglu(gate: mx.array, up: mx.array) -> mx.array:
    size = gate.size
    return _swiglu_kernel(
        inputs=[gate, up],
        template=[("T", gate.dtype)],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )[0]


# ── model ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    vocab_size: int   = 50257
    block_size: int   = 1024
    n_embd:     int   = 768
    n_head:     int   = 12
    n_layer:    int   = 12
    mlp_ratio:  float = 4.0

    @classmethod
    def base(cls): return cls()

    @classmethod
    def tiny(cls): return cls(n_embd=32, n_head=2, n_layer=2, block_size=64)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, 1e-6)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale    = self.head_dim ** -0.5
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def __call__(self, x: mx.array, cache=None):
        B, T, C = x.shape
        q, k, v = mx.split(self.qkv(x), 3, axis=-1)
        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            past_k, past_v = cache
            k = mx.concatenate([past_k, k], axis=2)
            v = mx.concatenate([past_v, v], axis=2)

        scores = (q @ k.swapaxes(-2, -1)) * self.scale  # [B, H, T, S]

        # causal mask only needed during prefill (T > 1);
        # during decode T == 1 and the single query attends to all past freely
        if T > 1:
            mask = mx.tril(mx.ones((T, T), dtype=mx.bool_))
            scores = mx.where(mask, scores, mx.full(scores.shape, float("-inf")))

        y = (mx.softmax(scores, axis=-1) @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden = (int(cfg.n_embd * cfg.mlp_ratio) + 63) // 64 * 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w3(swiglu(self.w1(x), self.w2(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn  = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp   = MLP(cfg)

    def __call__(self, x: mx.array, cache=None):
        attn_out, new_cache = self.attn(self.norm1(x), cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks  = [Block(cfg) for _ in range(cfg.n_layer)]
        self.norm    = RMSNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def __call__(self, idx: mx.array, cache=None):
        B, T = idx.shape
        past_len = cache[0][0].shape[2] if cache is not None else 0
        positions = mx.arange(past_len, past_len + T)
        x = self.tok_emb(idx) + self.pos_emb(positions)
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, layer_cache = block(x, cache[i] if cache is not None else None)
            new_cache.append(layer_cache)
        return self.head(self.norm(x)), new_cache

    def generate(self, idx: mx.array, max_new_tokens: int, temperature: float = 1.0) -> mx.array:
        # prefill: process full prompt, build KV cache
        logits, cache = self(idx)
        next_tok = mx.argmax(logits[:, -1, :] / temperature, axis=-1, keepdims=True)
        result = mx.concatenate([idx, next_tok], axis=1)
        mx.eval(result, *[t for k, v in cache for t in (k, v)])

        # decode: one new token per step, O(1) attention via cache
        for _ in range(max_new_tokens - 1):
            logits, cache = self(next_tok, cache)
            next_tok = mx.argmax(logits[:, -1, :] / temperature, axis=-1, keepdims=True)
            result = mx.concatenate([result, next_tok], axis=1)
            mx.eval(result, *[t for k, v in cache for t in (k, v)])

        return result


def load_model(weights_path: str, cfg: Config, bits: int = 0) -> GPT:
    model = GPT(cfg)
    weights = list(mx.load(weights_path).items())
    model.load_weights(weights)
    if bits in (4, 8):
        nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    return model


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["tiny", "base"], default="base")
    parser.add_argument("--weights",  default="weights.npz")
    parser.add_argument("--prompt",   default="Once upon a time")
    parser.add_argument("--max_new",  type=int,   default=200)
    parser.add_argument("--temp",     type=float, default=0.8)
    args = parser.parse_args()

    weights_path   = args.weights
    prompt         = args.prompt
    max_new_tokens = args.max_new
    temperature    = args.temp

    cfg = Config.tiny() if args.model == "tiny" else Config.base()
    enc = tiktoken.get_encoding("gpt2")

    print(f"loading {weights_path}...")
    model = load_model(weights_path, cfg)

    tokens = enc.encode(prompt)
    idx = mx.array([tokens])
    out = model.generate(idx, max_new_tokens, temperature)
    mx.eval(out)

    print(enc.decode(out[0].tolist()))
