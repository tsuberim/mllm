import os
import io
import argparse
import contextlib
from dotenv import load_dotenv
load_dotenv()
import torch
import torch.nn.functional as F
import numpy as np
import tiktoken
import wandb
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi

from model import GPT, Config
from optim import Muon

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",      choices=["sanity", "experiment", "iphone", "macbook"], required=True)
parser.add_argument("--batch_size", type=int,   required=True)
parser.add_argument("--max_steps",  type=int,   required=True)
parser.add_argument("--val_every",  type=int,   required=True)
parser.add_argument("--val_steps",  type=int,   required=True)
parser.add_argument("--save_every", type=int,   required=True)
parser.add_argument("--lr",         type=float, default=3e-4,  help="AdamW lr (embeddings, norms)")
parser.add_argument("--lr_muon",    type=float, default=0.02,  help="Muon lr (2-D weight matrices)")
parser.add_argument("--grad_clip",  type=float, default=1.0)
parser.add_argument("--wandb",      choices=["online", "disabled"], default="online")
parser.add_argument("--bf16",       action="store_true", help="cast model to bfloat16 (required for ~7B on single GPU)")
args = parser.parse_args()

model_cfg = {"sanity": Config.sanity, "experiment": Config.experiment, "iphone": Config.iphone, "macbook": Config.macbook}[args.model]()
Path("checkpoints").mkdir(exist_ok=True)
CKPT_NAME = f"checkpoints/ckpt_{args.model}.pt"
HF_REPO   = os.environ["HF_REPO"]

N_SAMPLE_PROMPTS = 4
PROMPT_TOKENS    = 20   # tokens fed as context
SAMPLE_NEW       = 40   # tokens to generate

# ── device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"device: {device}")
torch.set_float32_matmul_precision("high")  # TF32 on Ampere+; no-op elsewhere
if device == "cuda":
    torch.backends.cudnn.allow_tf32 = True
    # expandable_segments avoids OOM from allocator fragmentation on large models
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# bf16 autocast helps on CUDA (tensor cores); hurts on MPS (fp32 is native)
autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda" else contextlib.nullcontext())

# ── data ──────────────────────────────────────────────────────────────────────
assert Path("data_train.bin").exists(), "run data.py first"
enc = tiktoken.get_encoding("gpt2")

def load_tokens(path):
    return np.memmap(path, dtype=np.uint16, mode="r")

def get_batch(tokens, batch_size, block_size):
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    if device == "cuda":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y

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
dtype = torch.bfloat16 if args.bf16 else torch.float32
model = GPT(model_cfg).to(device=device, dtype=dtype)
if device == "cuda":
    # max-autotune: Triton finds optimal tile sizes for this GPU's SM count/config
    # one-time cost per model shape, cached in TORCHINDUCTOR_CACHE_DIR
    model = torch.compile(model, mode="max-autotune")
print(f"params: {model.num_params():,}  dtype: {dtype}")

# Muon for 2-D weight matrices inside transformer blocks;
# AdamW for embeddings, head, and 1-D norm weights.
# tok_emb.weight and head.weight are tied — deduplicate by id().
seen_ids: set = set()
muon_params, adam_params = [], []
for name, p in model.named_parameters():
    if id(p) in seen_ids:
        continue
    seen_ids.add(id(p))
    if p.ndim == 2 and "tok_emb" not in name:
        muon_params.append(p)
    else:
        adam_params.append(p)

optim_muon = Muon(muon_params, lr=args.lr_muon)
optim_adam = torch.optim.AdamW(adam_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1,
                               fused=(device == "cuda"))

# ── sampling ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(prompt: str, max_new: int = SAMPLE_NEW, temperature: float = 0.8) -> str:
    model.eval()
    tokens = enc.encode(prompt)
    idx = torch.tensor([tokens], device=device)
    with autocast:
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
    ckpt = {
        "step":       step,
        "model":      model.state_dict(),
        "optim_muon": optim_muon.state_dict(),
        "optim_adam": optim_adam.state_dict(),
    }
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    with open(CKPT_NAME, "wb") as f:
        f.write(buf.getbuffer())
    if hf:
        buf.seek(0)
        hf.upload_file(path_or_fileobj=buf, path_in_repo=CKPT_NAME,
                       repo_id=HF_REPO, repo_type="model")
        print(f"  checkpoint pushed to {HF_REPO}")

def load_checkpoint():
    global start_step
    ckpt = None
    if Path(CKPT_NAME).exists():
        ckpt = torch.load(CKPT_NAME, map_location=device, weights_only=False)
    elif hf:
        try:
            path = hf.hf_hub_download(repo_id=HF_REPO, filename=CKPT_NAME, repo_type="model")
            ckpt = torch.load(path, map_location=device, weights_only=False)
            print(f"resumed checkpoint from {HF_REPO}")
        except Exception:
            pass
    if ckpt:
        model.load_state_dict(ckpt["model"])
        if "optim_muon" in ckpt:
            optim_muon.load_state_dict(ckpt["optim_muon"])
            optim_adam.load_state_dict(ckpt["optim_adam"])
        start_step = ckpt["step"] + 1
        print(f"resumed from step {start_step}")

load_checkpoint()

# ── wandb ─────────────────────────────────────────────────────────────────────
wandb.init(
    project="merlin", resume="allow", mode=args.wandb,
    config={**model_cfg.__dict__, "batch_size": args.batch_size,
            "lr": args.lr, "lr_muon": args.lr_muon,
            "max_steps": args.max_steps, "device": device},
)

# ── training loop ─────────────────────────────────────────────────────────────
pbar = tqdm(range(start_step, args.max_steps), initial=start_step,
            total=args.max_steps, unit="step")

for step in pbar:
    model.train()
    x, y = get_batch(train_tokens, args.batch_size, model_cfg.block_size)
    with autocast:
        _, loss = model(x, y)

    optim_muon.zero_grad(set_to_none=True)
    optim_adam.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
    optim_muon.step()
    optim_adam.step()

    log = {"train/loss": loss.item(), "train/grad_norm": grad_norm}

    if step % args.val_every == 0:
        model.eval()
        with torch.no_grad(), autocast:
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
