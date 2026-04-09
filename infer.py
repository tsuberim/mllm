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


# ── RoPE ──────────────────────────────────────────────────────────────────────
# Fused Metal kernel: reads x and cos/sin once, writes rotated x in one pass.
# Avoids the 4+ separate kernel dispatches that rotate_half+apply_rope would
# generate (slice, negate, concat, two multiplies, add) — critical for decode
# where T=1 and per-launch overhead dominates.

_rope_cache: dict = {}  # (head_dim, max_seq_len) → (cos, sin); shared across layers

def get_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    key = (head_dim, max_seq_len)
    if key not in _rope_cache:
        freqs = 1.0 / (theta ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        t     = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = mx.outer(t, freqs)
        emb   = mx.concatenate([freqs, freqs], axis=-1)  # [T, D]
        cos, sin = mx.cos(emb), mx.sin(emb)
        mx.eval(cos, sin)
        _rope_cache[key] = (cos, sin)
    return _rope_cache[key]

def get_rope_slice(head_dim: int, max_seq_len: int, past_len: int, T: int):
    # No slice cache — caching creates accumulating stale MLX nodes that slow
    # down MLX's memory manager over the decode sequence. The base freqs are
    # already cached; a fresh slice each call is cheap and stays GC'd promptly.
    cos, sin = get_rope_freqs(head_dim, max_seq_len)
    return cos[past_len:past_len+T], sin[past_len:past_len+T]

# Fused q+k RoPE kernel: one Metal dispatch per layer instead of two.
# rotate_half convention:
#   d < D/2 → out[d] = x[d]*cos[d] - x[d+D/2]*sin[d]
#   d ≥ D/2 → out[d] = x[d]*cos[d] + x[d-D/2]*sin[d]
#
# GQA support: dispatches q.size threads; for elem < k_size the thread also
# writes k_out. k has fewer heads than q (n_kv_head < n_head) but the same
# D and T, so the same rotation formula applies — just a smaller array.
_rope2_kernel = mx.fast.metal_kernel(
    name="rope2",
    input_names=["q", "k", "cos_vals", "sin_vals", "params"],
    output_names=["q_out", "k_out"],
    source="""
        uint elem   = thread_position_in_grid.x;
        uint D      = params[0];
        uint T      = params[1];
        uint k_size = params[2];
        uint d      = elem % D;
        uint t      = (elem / D) % T;
        uint hd     = D / 2;
        float c     = cos_vals[t * D + d];
        float s     = sin_vals[t * D + d];
        uint  pair  = (d < hd) ? elem + hd : elem - hd;
        float sgn   = (d < hd) ? -1.0f : 1.0f;
        q_out[elem] = q[elem] * c + sgn * q[pair] * s;
        if (elem < k_size) {
            k_out[elem] = k[elem] * c + sgn * k[pair] * s;
        }
    """,
)

_rope2_params_cache: dict = {}   # (D, T, k_size) → params array

def rope2(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array):
    """Apply RoPE to q and k in one Metal dispatch. Supports GQA (k smaller than q)."""
    *_, T, D = q.shape
    k_size = k.size
    key = (D, T, k_size)
    if key not in _rope2_params_cache:
        _rope2_params_cache[key] = mx.array([D, T, k_size], dtype=mx.uint32)
    params = _rope2_params_cache[key]
    size   = q.size
    out = _rope2_kernel(
        inputs=[q, k, cos, sin, params],
        grid=(size, 1, 1),
        threadgroup=(min(256, size), 1, 1),
        output_shapes=[q.shape, k.shape],
        output_dtypes=[q.dtype, k.dtype],
    )
    return out[0], out[1]


# ── KV cache ──────────────────────────────────────────────────────────────────

class KVCache:
    """Pre-allocated KV cache for one attention layer.

    Pre-allocating [1, H, max_T, D] and writing new tokens via slice assignment
    avoids the per-step mx.concatenate that would otherwise allocate a new tensor
    and copy the full cache every token (~18.9 MB/token at T=512 for the base model).
    After mx.eval(), MLX can reuse the existing buffer for subsequent scatter ops.
    """
    def __init__(self, n_head: int, head_dim: int, max_T: int, dtype=mx.float32):
        self.offset = 0
        self.k = mx.zeros((1, n_head, max_T, head_dim), dtype=dtype)
        self.v = mx.zeros((1, n_head, max_T, head_dim), dtype=dtype)
        mx.eval(self.k, self.v)

    def update(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        T = k.shape[2]
        if self.offset + T > self.k.shape[2]:
            raise ValueError(
                f"KV cache overflow: offset {self.offset} + T {T} > max_T {self.k.shape[2]}"
            )
        self.k[:, :, self.offset : self.offset + T, :] = k
        self.v[:, :, self.offset : self.offset + T, :] = v
        self.offset += T
        return self.k[:, :, : self.offset, :], self.v[:, :, : self.offset, :]


# ── model ─────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    vocab_size: int   = 50257
    block_size: int   = 1024
    n_embd:     int   = 768
    n_head:     int   = 12
    n_kv_head:  int   = 4
    n_layer:    int   = 12
    mlp_ratio:  float = 4.0

    @classmethod
    def sanity(cls): return cls(n_embd=32, n_head=2, n_kv_head=2, n_layer=2, block_size=64)

    @classmethod
    def experiment(cls): return cls(n_embd=256, n_head=8, n_kv_head=2, n_layer=8, block_size=512)

    @classmethod
    def iphone(cls): return cls(n_embd=3072, n_head=24, n_kv_head=8, n_layer=20, block_size=4096)

    @classmethod
    def macbook(cls): return cls(n_embd=4096, n_head=32, n_kv_head=8, n_layer=26, block_size=4096)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, 1e-6)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head    = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim  = cfg.n_embd // cfg.n_head
        self.scale     = self.head_dim ** -0.5
        self._rope_len = cfg.block_size
        self.q_proj  = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(cfg.n_embd, 2 * cfg.n_kv_head * self.head_dim, bias=False)
        self.proj    = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def __call__(self, x: mx.array, cache: KVCache | None = None):
        B, T, C = x.shape
        q  = self.q_proj(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        kv = self.kv_proj(x).reshape(B, T, 2 * self.n_kv_head, self.head_dim)
        k  = kv[:, :, :self.n_kv_head, :].transpose(0, 2, 1, 3)
        v  = kv[:, :, self.n_kv_head:, :].transpose(0, 2, 1, 3)

        past_len = cache.offset if cache is not None else 0
        cos, sin = get_rope_slice(self.head_dim, self._rope_len, past_len, T)
        q, k = rope2(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(k, v)

        # MLX sdpa handles GQA natively (broadcasts k/v over q heads)
        y = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask="causal"
        ).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(y)


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

    def __call__(self, x: mx.array, cache: KVCache | None = None):
        x = x + self.attn(self.norm1(x), cache)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks  = [Block(cfg) for _ in range(cfg.n_layer)]
        self.norm    = RMSNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def make_cache(self) -> list[KVCache]:
        dtype = self.tok_emb.weight.dtype
        return [KVCache(b.attn.n_kv_head, b.attn.head_dim, self.cfg.block_size, dtype=dtype)
                for b in self.blocks]

    def __call__(self, idx: mx.array, cache: list[KVCache] | None = None):
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cache[i] if cache is not None else None)
        return self.head(self.norm(x)), cache

    def generate(self, idx: mx.array, max_new_tokens: int, temperature: float = 1.0) -> mx.array:
        cache = self.make_cache()

        # prefill: process full prompt, build KV cache
        logits, cache = self(idx, cache)
        next_tok = mx.argmax(logits[:, -1, :] / temperature, axis=-1, keepdims=True)
        result = mx.concatenate([idx, next_tok], axis=1)
        mx.eval(result, *[c.k for c in cache], *[c.v for c in cache])

        # decode: one new token per step, O(1) attention via cache
        for _ in range(max_new_tokens - 1):
            logits, cache = self(next_tok, cache)
            next_tok = mx.argmax(logits[:, -1, :] / temperature, axis=-1, keepdims=True)
            result = mx.concatenate([result, next_tok], axis=1)
            mx.eval(result, *[c.k for c in cache], *[c.v for c in cache])

        return result


def load_model(weights_path: str, cfg: Config, bits: int = 0) -> GPT:
    model = GPT(cfg)
    weights = list(mx.load(weights_path).items())
    if bits == 16:
        weights = [(k, v.astype(mx.float16)) for k, v in weights]
    model.load_weights(weights)
    if bits in (4, 8):
        nn.quantize(model, group_size=64, bits=bits)
    mx.eval(model.parameters())
    # Compile each MLP: fuses w1/w2 (independent), swiglu, and w3 into one
    # Metal graph, eliminating per-op dispatch overhead. ~5% TPS, 18× lower
    # run-to-run variance. Compiling norms or attention projections individually
    # adds wrapper overhead without fusion — MLP is the sweet spot.
    for block in model.blocks:
        block.mlp = mx.compile(block.mlp)
    return model


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    choices=["sanity", "experiment", "iphone", "macbook"], default="iphone")
    parser.add_argument("--weights",  default="checkpoints/weights.npz")
    parser.add_argument("--prompt",   default="Once upon a time")
    parser.add_argument("--max_new",  type=int,   default=200)
    parser.add_argument("--temp",     type=float, default=0.8)
    parser.add_argument("--bits",     type=int,   default=0, choices=[0, 4, 8, 16],
                        help="weight precision (0=fp32, 16=fp16, 4/8=int quantized)")
    args = parser.parse_args()

    weights_path   = args.weights
    prompt         = args.prompt
    max_new_tokens = args.max_new
    temperature    = args.temp

    cfg = {"sanity": Config.sanity, "experiment": Config.experiment, "iphone": Config.iphone, "macbook": Config.macbook}[args.model]()
    enc = tiktoken.get_encoding("gpt2")

    print(f"loading {weights_path}...")
    model = load_model(weights_path, cfg, bits=args.bits)

    tokens = enc.encode(prompt)
    idx = mx.array([tokens])
    out = model.generate(idx, max_new_tokens, temperature)
    mx.eval(out)

    print(enc.decode(out[0].tolist()))
