import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class Config:
    vocab_size:  int   = 50257   # tiktoken gpt2
    block_size:  int   = 1024
    n_embd:      int   = 768
    n_head:      int   = 12
    n_layer:     int   = 12
    mlp_ratio:   float = 4.0

    @classmethod
    def tiny(cls):
        """~200K params — local pipeline testing only."""
        return cls(n_embd=32, n_head=2, n_layer=2, block_size=64)

    @classmethod
    def medium(cls):
        """~21M params — local training experiments and cheap remote runs."""
        return cls(n_embd=256, n_head=8, n_layer=8, block_size=512)

    @classmethod
    def base(cls):
        """~117M params — production target."""
        return cls(n_embd=768, n_head=12, n_layer=12, block_size=1024)


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
    # q, k: [B, H, T, D]; cos/sin: [T, D] → broadcast as [1, 1, T, D]
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
        return x / x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt() * self.weight


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv  = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        cos, sin = precompute_rope_freqs(self.head_dim, cfg.block_size)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, self.rope_cos[:T], self.rope_sin[:T])
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg     = cfg
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

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
