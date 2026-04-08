import os
import io
import argparse
from dotenv import load_dotenv
load_dotenv()
import torch
import numpy as np
import tiktoken
import wandb
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi

from model import GPT, Config

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",      choices=["tiny", "medium", "base"], required=True)
parser.add_argument("--batch_size", type=int,   required=True)
parser.add_argument("--max_steps",  type=int,   required=True)
parser.add_argument("--val_every",  type=int,   required=True)
parser.add_argument("--val_steps",  type=int,   required=True)
parser.add_argument("--save_every", type=int,   required=True)
parser.add_argument("--lr",         type=float, default=3e-4)
parser.add_argument("--grad_clip",  type=float, default=1.0)
parser.add_argument("--wandb",      choices=["online", "disabled"], default="online")
args = parser.parse_args()

model_cfg = {"tiny": Config.tiny, "medium": Config.medium, "base": Config.base}[args.model]()
CKPT_NAME = f"ckpt_{args.model}.pt"
HF_REPO   = os.environ["HF_REPO"]

N_SAMPLE_PROMPTS = 4
PROMPT_TOKENS    = 20   # tokens fed as context
SAMPLE_NEW       = 20   # tokens to generate

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
enc = tiktoken.get_encoding("gpt2")

def load_tokens(path):
    return np.memmap(path, dtype=np.uint16, mode="r")

def get_batch(tokens, batch_size, block_size):
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

train_tokens = load_tokens("data_train.bin")
val_tokens   = load_tokens("data_validation.bin")

# extract story openings from the val set as fixed sample prompts
def _val_prompts(n: int, prompt_tokens: int) -> list[str]:
    prompts, i = [], 0
    while len(prompts) < n and i < len(val_tokens) - prompt_tokens:
        if val_tokens[i] == enc.eot_token:
            toks = val_tokens[i+1 : i+1+prompt_tokens].tolist()
            prompts.append(enc.decode(toks))
            i += prompt_tokens
        else:
            i += 1
    return prompts

SAMPLE_PROMPTS = _val_prompts(N_SAMPLE_PROMPTS, PROMPT_TOKENS)

# ── model ─────────────────────────────────────────────────────────────────────
model = GPT(model_cfg).to(device)
if device == "cuda":
    model = torch.compile(model)
print(f"params: {model.num_params():,}")

optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

# ── sampling ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt: str, max_new: int = SAMPLE_NEW, temperature: float = 0.8) -> str:
    model.eval()
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], device=device)
    for _ in range(max_new):
        idx_cond = idx[:, -model_cfg.block_size:]
        logits, _ = model(idx_cond)
        next_tok = torch.multinomial(
            torch.softmax(logits[:, -1, :] / temperature, dim=-1), 1)
        idx = torch.cat([idx, next_tok], dim=1)
    return enc.decode(idx[0].tolist())

# ── checkpoint ────────────────────────────────────────────────────────────────
hf = HfApi() if args.wandb == "online" else None
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
    project="mllm", resume="allow", mode=args.wandb,
    config={**model_cfg.__dict__, "batch_size": args.batch_size, "lr": args.lr,
            "max_steps": args.max_steps, "device": device},
)

# ── training loop ─────────────────────────────────────────────────────────────
pbar = tqdm(range(start_step, args.max_steps), initial=start_step,
            total=args.max_steps, unit="step")

for step in pbar:
    model.train()
    x, y = get_batch(train_tokens, args.batch_size, model_cfg.block_size)
    _, loss = model(x, y)

    optim.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
    optim.step()

    log = {"train/loss": loss.item(), "train/grad_norm": grad_norm}

    if step % args.val_every == 0:
        model.eval()
        with torch.no_grad():
            val_loss = sum(
                model(*get_batch(val_tokens, args.batch_size, model_cfg.block_size))[1].item()
                for _ in range(args.val_steps)
            ) / args.val_steps

        samples = wandb.Table(columns=["prompt", "sample"])
        for prompt in SAMPLE_PROMPTS:
            samples.add_data(prompt, generate(prompt))

        log["val/loss"] = val_loss
        log["samples"]  = samples
        pbar.set_postfix(loss=f"{loss.item():.4f}", val=f"{val_loss:.4f}",
                         gnorm=f"{grad_norm:.2f}")
    else:
        pbar.set_postfix(loss=f"{loss.item():.4f}", gnorm=f"{grad_norm:.2f}")

    if step % args.save_every == 0:
        save_checkpoint(step)

    wandb.log(log, step=step)

save_checkpoint(args.max_steps - 1)
wandb.finish()
