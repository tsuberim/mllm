import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    vocab_size:  int   = 32016   # 32000 BPE + 16 special tokens
    block_size:  int   = 1024
    n_embd:      int   = 768
    n_head:      int   = 12
    n_kv_head:   int   = 4       # GQA: fewer KV heads than Q heads
    n_layer:     int   = 12
    mlp_ratio:   float = 4.0

    @classmethod
    def sanity(cls):
        """~200K params — local pipeline testing only."""
        return cls(n_embd=32, n_head=2, n_kv_head=2, n_layer=2, block_size=64)

    @classmethod
    def experiment(cls):
        """~16M params — cheap experiment runs. head_dim=32, block_size=512."""
        return cls(n_embd=256, n_head=8, n_kv_head=2, n_layer=8, block_size=512)


    @classmethod
    def b3(cls):
        """~3.17B params — 3B target. head_dim=128, block_size=4096."""
        return cls(n_embd=3072, n_head=24, n_kv_head=8, n_layer=20, block_size=4096)

    @classmethod
    def b7(cls):
        """~7.19B params — 7B target. head_dim=128, block_size=4096."""
        return cls(n_embd=4096, n_head=32, n_kv_head=8, n_layer=26, block_size=4096)


# ── RoPE ──────────────────────────────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t     = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)          # [T, D/2]
    emb   = torch.cat([freqs, freqs], -1)  # [T, D]  — rotate_half convention
    return emb.cos(), emb.sin()

def rotate_half(x):
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)

def apply_rope(q, k, cos, sin):
    # q: [B, n_head, T, D]; k: [B, n_kv_head, T, D]; cos/sin: [T, D]
    cos = cos[None, None]
    sin = sin[None, None]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


# ── layers ────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head    = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim  = cfg.n_embd // cfg.n_head
        self.q_proj  = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(cfg.n_embd, 2 * cfg.n_kv_head * self.head_dim, bias=False)
        self.proj    = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        cos, sin = precompute_rope_freqs(self.head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x, attn_mask=None):
        B, T, C = x.shape
        q  = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, T, 2 * self.n_kv_head, self.head_dim)
        k, v = kv.split(self.n_kv_head, dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, self.rope_cos[:T], self.rope_sin[:T])
        # Expand KV heads to match Q heads for SDPA (zero-copy view)
        gqa = self.n_head // self.n_kv_head
        k = k.unsqueeze(2).expand(B, self.n_kv_head, gqa, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.n_kv_head, gqa, T, self.head_dim).reshape(B, self.n_head, T, self.head_dim)
        # attn_mask: (B, 1, T, T) bool, True = attend. None → standard causal.
        # NOTE: custom mask disables FlashAttention kernel; for T > ~1024 use
        # flash_attn_varlen_func instead (handles packed docs natively).
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    """SwiGLU"""
    def __init__(self, cfg):
        super().__init__()
        hidden = (int(cfg.n_embd * cfg.mlp_ratio) + 63) // 64 * 64
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd)
        self.attn  = Attention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd)
        self.mlp   = MLP(cfg)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg             = cfg
        self.grad_checkpoint = False
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm    = RMSNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, attn_mask=None, sample_weights=None):
        from torch.utils.checkpoint import checkpoint
        B, T = idx.shape
        x = self.tok_emb(idx)
        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                x = checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask)
        x = self.norm(x)
        if targets is not None:
            # chunked cross-entropy: avoids materialising full [B*T, V] logit tensor
            # critical for large models where B*T*V can exceed 1.5 GB
            chunk = max(1, T // 4)
            if sample_weights is not None:
                # per-sample loss weighting by external weights (e.g. EMA grad norm)
                per_sample = torch.zeros(B, device=x.device, dtype=x.dtype)
                n_chunks = 0
                for i in range(0, T, chunk):
                    ce = F.cross_entropy(
                        (x[:, i:i+chunk] @ self.head.weight.T).reshape(-1, self.cfg.vocab_size),
                        targets[:, i:i+chunk].contiguous().reshape(-1),
                        reduction="none",
                    ).reshape(B, -1).mean(1)
                    per_sample = per_sample + ce
                    n_chunks += 1
                per_sample = per_sample / n_chunks
                w = sample_weights / sample_weights.sum()
                loss = (w * per_sample).sum()
            else:
                chunks = range(0, T, chunk)
                loss = sum(
                    F.cross_entropy(
                        (x[:, i:i+chunk] @ self.head.weight.T).reshape(-1, self.cfg.vocab_size),
                        targets[:, i:i+chunk].contiguous().reshape(-1),
                    )
                    for i in chunks
                ) / len(chunks)
            return None, loss
        logits = self.head(x)
        return logits, None

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
