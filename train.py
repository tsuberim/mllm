import os
import io
import argparse
from dotenv import load_dotenv
load_dotenv()
import torch
import numpy as np
import wandb
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi

from model import GPT, Config

# ── config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["tiny", "base"], default="base")
args = parser.parse_args()

is_tiny = args.model == "tiny"
model_cfg = Config.tiny() if is_tiny else Config.base()

lr          = 3e-4
max_steps   = 2        if is_tiny else 20_000
batch_size  = 2        if is_tiny else 64
val_every   = 1        if is_tiny else 500
val_steps   = 2        if is_tiny else 50
save_every  = 1        if is_tiny else 1_000
grad_clip   = 1.0

HF_REPO     = os.environ["HF_REPO"]
CKPT_NAME   = "ckpt.pt"

# ── device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device: {device}")

# ── data ──────────────────────────────────────────────────────────────────────
assert Path("data_train.bin").exists(), "run data.py first"

def load_tokens(path):
    return np.memmap(path, dtype=np.uint16, mode="r")

def get_batch(tokens, batch_size, block_size):
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

train_tokens = load_tokens("data_train.bin")
val_tokens   = load_tokens("data_validation.bin")

# ── model ─────────────────────────────────────────────────────────────────────
model = GPT(model_cfg).to(device)
if device == "cuda":
    model = torch.compile(model)
print(f"params: {model.num_params():,}")

optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

# ── checkpoint ────────────────────────────────────────────────────────────────
hf = None if is_tiny else HfApi()
start_step = 0

def save_checkpoint(step):
    ckpt = {"step": step, "model": model.state_dict(), "optim": optim.state_dict()}
    torch.save(ckpt, CKPT_NAME)
    if hf:
        buf = io.BytesIO()
        torch.save(ckpt, buf)
        buf.seek(0)
        hf.upload_file(path_or_fileobj=buf, path_in_repo=CKPT_NAME,
                       repo_id=HF_REPO, repo_type="model")
        print(f"  checkpoint pushed to {HF_REPO}")

def load_checkpoint():
    global start_step
    ckpt = None
    if Path(CKPT_NAME).exists():
        ckpt = torch.load(CKPT_NAME, map_location=device)
    elif hf:
        try:
            path = hf.hf_hub_download(repo_id=HF_REPO, filename=CKPT_NAME, repo_type="model")
            ckpt = torch.load(path, map_location=device)
            print(f"resumed checkpoint from {HF_REPO}")
        except Exception:
            pass
    if ckpt:
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_step = ckpt["step"] + 1
        print(f"resumed from step {start_step}")

load_checkpoint()

# ── wandb ─────────────────────────────────────────────────────────────────────
wandb.init(
    project="mllm", resume="allow", mode="disabled" if is_tiny else "online",
    config={**model_cfg.__dict__, "batch_size": batch_size, "lr": lr, "max_steps": max_steps, "device": device},
)

# ── training loop ─────────────────────────────────────────────────────────────
pbar = tqdm(range(start_step, max_steps), initial=start_step, total=max_steps)

for step in pbar:
    model.train()
    x, y = get_batch(train_tokens, batch_size, model_cfg.block_size)
    _, loss = model(x, y)

    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optim.step()

    log = {"train/loss": loss.item()}

    if step % val_every == 0:
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                model(*get_batch(val_tokens, batch_size, model_cfg.block_size))[1].item()
                for _ in range(val_steps)
            ) / val_steps
        log["val/loss"] = val_loss
        pbar.set_postfix(train=f"{loss.item():.4f}", val=f"{val_loss:.4f}")
    else:
        pbar.set_postfix(train=f"{loss.item():.4f}")

    if step % save_every == 0:
        save_checkpoint(step)

    wandb.log(log, step=step)

save_checkpoint(max_steps - 1)
wandb.finish()
