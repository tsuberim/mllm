"""
End-to-end parity test: PyTorch (CPU) vs MLX (Metal).

"Same to the bit" caveat: float32 ops involving transcendentals (exp in silu,
softmax) may differ by 1-2 ULP across CPU and Metal math libraries. We assert
allclose(atol=1e-5) for logits — tight enough to catch any real divergence —
and require exact greedy token match, which is the meaningful correctness bar.
"""
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx

from model import GPT, Config
import convert as conv
from infer import swiglu, load_model
from infer import Config as InferConfig


# ── helpers ───────────────────────────────────────────────────────────────────

def make_checkpoint(cfg: Config, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = GPT(cfg)
    model.eval()
    return {"step": 0, "model": model.state_dict(), "optim": {}}


# ── swiglu kernel parity ──────────────────────────────────────────────────────

def test_swiglu_parity():
    rng = np.random.default_rng(42)
    gate_np = rng.standard_normal((4, 16, 128)).astype(np.float32)
    up_np   = rng.standard_normal((4, 16, 128)).astype(np.float32)

    # PyTorch reference (CPU)
    ref = (F.silu(torch.from_numpy(gate_np)) * torch.from_numpy(up_np)).numpy()

    # Metal kernel
    out = swiglu(mx.array(gate_np), mx.array(up_np))
    mx.eval(out)
    got = np.array(out)

    # max observed diff is ~5e-7 (exp differs by ~4 ULP between CPU and Metal)
    np.testing.assert_allclose(got, ref, atol=1e-6,
                               err_msg="swiglu kernel diverges from PyTorch reference")


# ── full model parity ─────────────────────────────────────────────────────────

def test_e2e_parity():
    cfg = Config.tiny()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path    = f"{tmp}/ckpt.pt"
        weights_path = f"{tmp}/weights.npz"

        # save a freshly initialised model
        ckpt = make_checkpoint(cfg)
        torch.save(ckpt, ckpt_path)

        # convert to MLX
        conv.main(ckpt_path, weights_path)

        # fixed input
        torch.manual_seed(1)
        tokens_np = torch.randint(0, cfg.vocab_size, (1, cfg.block_size)).numpy()

        # ── PyTorch forward (CPU) ──────────────────────────────────────────────
        pt_model = GPT(cfg)
        pt_model.load_state_dict(ckpt["model"])
        pt_model.eval()

        with torch.no_grad():
            logits_pt, _ = pt_model(torch.from_numpy(tokens_np))
        logits_pt = logits_pt.numpy()          # [1, T, vocab]

        # ── MLX forward ───────────────────────────────────────────────────────
        infer_cfg = InferConfig.tiny()
        mlx_model = load_model(weights_path, infer_cfg)
        logits_mx, _ = mlx_model(mx.array(tokens_np))
        mx.eval(logits_mx)
        logits_mx = np.array(logits_mx)        # [1, T, vocab]

        # max observed diff is ~1.5e-7 (float32 machine epsilon range)
        np.testing.assert_allclose(
            logits_mx, logits_pt, atol=1e-6,
            err_msg="logits diverge between PyTorch and MLX",
        )

        # greedy token predictions must match exactly
        greedy_pt = np.argmax(logits_pt,  axis=-1)
        greedy_mx = np.argmax(logits_mx, axis=-1)
        assert np.array_equal(greedy_pt, greedy_mx), (
            f"greedy tokens differ:\nPT:  {greedy_pt}\nMLX: {greedy_mx}"
        )


# ── KV cache parity ───────────────────────────────────────────────────────────

def test_kv_cache_parity():
    """Prefill+decode via KV cache must match a single full-sequence forward pass."""
    cfg = Config.tiny()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path    = f"{tmp}/ckpt.pt"
        weights_path = f"{tmp}/weights.npz"

        ckpt = make_checkpoint(cfg)
        torch.save(ckpt, ckpt_path)
        conv.main(ckpt_path, weights_path)

        torch.manual_seed(2)
        prompt_len  = 8
        decode_steps = 4
        seq_len = prompt_len + decode_steps

        tokens_np = torch.randint(0, cfg.vocab_size, (1, seq_len)).numpy()

        infer_cfg = InferConfig.tiny()
        mlx_model = load_model(weights_path, infer_cfg)

        # ── reference: full sequence in one forward pass ───────────────────
        logits_full, _ = mlx_model(mx.array(tokens_np))
        mx.eval(logits_full)
        logits_full_np = np.array(logits_full)  # [1, seq_len, vocab]

        # ── cached: prefill prompt, then decode one token at a time ────────
        prompt = mx.array(tokens_np[:, :prompt_len])
        logits_prefill, cache = mlx_model(prompt)
        mx.eval(logits_prefill, *[t for k, v in cache for t in (k, v)])

        cached_logits = [np.array(logits_prefill)]
        for i in range(decode_steps):
            tok = mx.array(tokens_np[:, prompt_len + i : prompt_len + i + 1])
            logits_step, cache = mlx_model(tok, cache)
            mx.eval(logits_step, *[t for k, v in cache for t in (k, v)])
            cached_logits.append(np.array(logits_step))

        logits_cached_np = np.concatenate(cached_logits, axis=1)  # [1, seq_len, vocab]

        np.testing.assert_allclose(
            logits_cached_np, logits_full_np, atol=1e-5,
            err_msg="KV-cache logits diverge from full-sequence forward pass",
        )


# ── RoPE offset correctness ───────────────────────────────────────────────────

def test_rope_offset():
    """A single token decoded at position P must get RoPE freqs for position P, not 0."""
    cfg = Config.tiny()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path    = f"{tmp}/ckpt.pt"
        weights_path = f"{tmp}/weights.npz"

        ckpt = make_checkpoint(cfg, seed=7)
        torch.save(ckpt, ckpt_path)
        conv.main(ckpt_path, weights_path)

        infer_cfg = InferConfig.tiny()
        mlx_model = load_model(weights_path, infer_cfg)

        torch.manual_seed(3)
        tok_np = torch.randint(0, cfg.vocab_size, (1, 1)).numpy()
        tok = mx.array(tok_np)

        # decode at position 0 (no cache)
        logits_pos0, _ = mlx_model(tok)
        mx.eval(logits_pos0)

        # decode at position 4 (after a 4-token prefill)
        prefix = mx.array(torch.randint(0, cfg.vocab_size, (1, 4)).numpy())
        _, cache = mlx_model(prefix)
        mx.eval(*[t for k, v in cache for t in (k, v)])
        logits_pos4, _ = mlx_model(tok, cache)
        mx.eval(logits_pos4)

        # same token at different positions must produce different logits
        assert not np.allclose(np.array(logits_pos0), np.array(logits_pos4), atol=1e-4), \
            "RoPE offset broken: same token at position 0 and 4 produced identical logits"
