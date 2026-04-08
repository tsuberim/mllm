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
