"""
SFT (Supervised Fine-Tuning) — response-only loss on packed instruction-response data.

Loads a pre-trained Merlin checkpoint, fine-tunes on SFT binary data.
Data format: uint16 ids + uint8 loss_mask produced by data/pipeline/06_prepare_sft.py.

Usage (local):
    python sft.py --model experiment --base experiment --max_steps 1000 \\
                  --val_every 100 --val_steps 5 --save_every 500

Usage (Modal):
    modal run modal_app.py::sft_entrypoint --model experiment --base experiment
"""
import os
import io
import argparse
import contextlib
import math
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn.functional as F
import numpy as np
import wandb
from tqdm import tqdm
from huggingface_hub import HfApi

import tok
from model import GPT, Config

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",      choices=["sanity", "experiment", "3b", "7b"], required=True)
parser.add_argument("--base",       type=str, required=True,
                    help="checkpoint tag to load as starting point (e.g. 'experiment')")
parser.add_argument("--max_steps",  type=int, required=True)
parser.add_argument("--val_every",  type=int, required=True)
parser.add_argument("--val_steps",  type=int, required=True)
parser.add_argument("--save_every", type=int, required=True)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr",         type=float, default=1e-5,  help="AdamW lr")
parser.add_argument("--lr_min",     type=float, default=1e-6,  help="min lr at end of cosine decay")
parser.add_argument("--warmup",     type=int,   default=50,    help="linear warmup steps")
parser.add_argument("--grad_clip",  type=float, default=1.0)
parser.add_argument("--bf16",             action="store_true")
parser.add_argument("--grad_checkpoint",  action="store_true")
parser.add_argument("--wandb",      choices=["online", "disabled"], default="online")
parser.add_argument("--tag",        type=str, default=None,
                    help="checkpoint/wandb tag (default: sft-{model})")
args = parser.parse_args()

model_cfg = {"sanity": Config.sanity, "experiment": Config.experiment, "3b": Config.b3, "7b": Config.b7}[args.model]()
Path("checkpoints").mkdir(exist_ok=True)
_tag      = args.tag if args.tag else f"sft-{args.model}"
CKPT_NAME = f"checkpoints/ckpt_{_tag}.pt"
HF_REPO   = os.environ["HF_REPO"]

# ── device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device: {device}")
torch.set_float32_matmul_precision("high")
if device == "cuda":
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda" else contextlib.nullcontext())

# ── data ──────────────────────────────────────────────────────────────────────
enc = tok.load()

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/tokenized"))
SFT_DIR  = Path(os.environ.get("SFT_DIR", DATA_DIR.parent / "sft"))

_seq_len = model_cfg.block_size


def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    ids_path  = SFT_DIR / f"sft_{name}.bin"
    mask_path = SFT_DIR / f"sft_{name}_mask.bin"
    if not ids_path.exists():
        raise FileNotFoundError(
            f"{ids_path} not found — run data/pipeline/06_prepare_sft.py first"
        )
    ids  = np.fromfile(ids_path,  dtype=np.uint16).reshape(-1, _seq_len)
    mask = np.fromfile(mask_path, dtype=np.uint8 ).reshape(-1, _seq_len)
    return ids, mask


train_ids, train_mask = _load_split("train")
val_ids,   val_mask   = _load_split("val")
print(f"sft data: train={len(train_ids)} chunks, val={len(val_ids)} chunks")
n_resp_train = int(train_mask.sum())
print(f"  response tokens in train: {n_resp_train:,} "
      f"({100*n_resp_train/train_ids.size:.1f}%)")


def get_batch(ids: np.ndarray, mask: np.ndarray, batch_size: int):
    ix     = torch.randint(len(ids), (batch_size,))
    ix_np  = ix.numpy()
    x_np   = ids[ix_np].astype(np.int64)
    y_np   = np.roll(x_np, -1, axis=1)   # targets: shifted by 1
    y_np[:, -1] = 0                       # last target is padding; loss masked anyway
    m_np   = mask[ix_np].astype(np.int64)
    # shift mask: label at position t is predicted from position t-1
    m_np   = np.roll(m_np, -1, axis=1)
    m_np[:, -1] = 0

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    m = torch.from_numpy(m_np)
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        m = m.pin_memory().to(device, non_blocking=True)
    else:
        x, y, m = x.to(device), y.to(device), m.to(device)
    return x, y, m

# ── model ─────────────────────────────────────────────────────────────────────
dtype = torch.bfloat16 if args.bf16 else torch.float32
model = GPT(model_cfg).to(device=device, dtype=dtype)
if args.grad_checkpoint:
    model.grad_checkpoint = True
if device == "cuda":
    model = torch.compile(model)
print(f"params: {model.num_params():,}  dtype: {dtype}")

# AdamW only — Muon is for pre-training; fine-tuning at 1e-5 doesn't need it
optim = torch.optim.AdamW(
    model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
    fused=(device == "cuda"),
)

# ── load pre-trained base checkpoint ──────────────────────────────────────────
hf = HfApi() if args.wandb == "online" else None


def _load_base():
    base_ckpt = f"checkpoints/ckpt_{args.base}.pt"
    ckpt = None
    if Path(base_ckpt).exists():
        ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
        print(f"loaded base from {base_ckpt}")
    elif hf:
        try:
            path = hf.hf_hub_download(repo_id=HF_REPO, filename=base_ckpt, repo_type="model")
            ckpt = torch.load(path, map_location=device, weights_only=False)
            print(f"loaded base from HF: {HF_REPO}/{base_ckpt}")
        except Exception as e:
            print(f"WARNING: could not load base checkpoint ({e}); training from scratch")
    if ckpt:
        try:
            missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
            if missing:
                print(f"  missing keys ({len(missing)}): {missing[:3]}{'...' if len(missing)>3 else ''}")
            if unexpected:
                print(f"  unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected)>3 else ''}")
        except RuntimeError as e:
            print(f"WARNING: checkpoint shape mismatch — starting from scratch. ({e})"[:200])


_load_base()

# ── checkpoint save/load ──────────────────────────────────────────────────────
def save_checkpoint(step: int):
    # Strip _orig_mod. prefix added by torch.compile so checkpoints load on any device
    raw_sd = model.state_dict()
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    ckpt = {"step": step, "model": clean_sd, "optim": optim.state_dict()}
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    with open(CKPT_NAME, "wb") as f:
        f.write(buf.getbuffer())
    if hf:
        from huggingface_hub import create_repo
        create_repo(HF_REPO, repo_type="model", exist_ok=True)
        buf.seek(0)
        hf.upload_file(path_or_fileobj=buf, path_in_repo=CKPT_NAME,
                       repo_id=HF_REPO, repo_type="model")
        print(f"  checkpoint pushed → {HF_REPO}/{CKPT_NAME}")

# ── lr schedule ───────────────────────────────────────────────────────────────
def get_lr(step: int) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / max(args.warmup, 1)
    progress = (step - args.warmup) / max(args.max_steps - args.warmup, 1)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.lr_min + (args.lr - args.lr_min) * cosine

# ── wandb ─────────────────────────────────────────────────────────────────────
wandb.init(
    project="merlin", name=_tag,
    mode=args.wandb,
    config={
        **model_cfg.__dict__,
        "base": args.base, "batch_size": args.batch_size,
        "lr": args.lr, "lr_min": args.lr_min, "warmup": args.warmup,
        "max_steps": args.max_steps, "device": device,
        "train_chunks": len(train_ids), "val_chunks": len(val_ids),
    },
)

# ── training loop ─────────────────────────────────────────────────────────────
pbar = tqdm(range(args.max_steps), total=args.max_steps, unit="step")

for step in pbar:
    lr = get_lr(step)
    for g in optim.param_groups:
        g["lr"] = lr

    model.train()
    x, y, m = get_batch(train_ids, train_mask, args.batch_size)
    with autocast:
        _, loss = model(x, y, loss_mask=m)

    optim.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
    optim.step()

    log = {"train/loss": loss.item(), "train/grad_norm": grad_norm, "train/lr": lr}

    if step % args.val_every == 0 and step > 0:
        model.eval()
        with torch.no_grad(), autocast:
            val_loss = 0.0
            for _ in range(args.val_steps):
                xv, yv, mv = get_batch(val_ids, val_mask, args.batch_size)
                val_loss += model(xv, yv, loss_mask=mv)[1].item()
            val_loss /= args.val_steps
        log["val/loss"] = val_loss
        pbar.set_postfix(loss=f"{loss.item():.4f}", val=f"{val_loss:.4f}",
                         gnorm=f"{grad_norm:.2f}")
    else:
        pbar.set_postfix(loss=f"{loss.item():.4f}", gnorm=f"{grad_norm:.2f}")

    if step % args.save_every == 0 and step > 0:
        save_checkpoint(step)

    wandb.log(log, step=step)

save_checkpoint(args.max_steps - 1)
wandb.finish()
