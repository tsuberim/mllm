"""Convert PyTorch checkpoint to MLX-loadable npz."""
import sys
import torch
import numpy as np

_SKIP = {"rope_cos", "rope_sin"}  # computed from scratch in MLX — not saved

def main(ckpt_path="ckpt.pt", out_path="weights.npz"):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    weights = {k: v.float().numpy() for k, v in ckpt["model"].items()
               if not any(k.endswith(s) for s in _SKIP)}
    # head.weight is tied to tok_emb.weight in training — make it explicit for inference
    weights["head.weight"] = weights["tok_emb.weight"].copy()
    np.savez(out_path, **weights)
    print(f"saved {len(weights)} tensors → {out_path}")
    for k, v in sorted(weights.items()):
        print(f"  {k:50s} {str(v.shape):25s} {v.dtype}")

if __name__ == "__main__":
    main(*sys.argv[1:])
