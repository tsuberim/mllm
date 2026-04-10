import os
import io
import argparse
import contextlib
from dotenv import load_dotenv
load_dotenv()
import torch
import torch.nn.functional as F
import numpy as np
import tok
import wandb
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import HfApi

from model import GPT, Config
from optim import Muon
import eval_ckpt

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--model",      choices=["sanity", "experiment", "iphone", "macbook"], required=True)
parser.add_argument("--batch_size", type=int,   required=True)
parser.add_argument("--max_steps",  type=int,   required=True)
parser.add_argument("--val_every",  type=int,   required=True)
parser.add_argument("--val_steps",  type=int,   required=True)
parser.add_argument("--save_every", type=int,   required=True)
parser.add_argument("--eval_every", type=int,   default=None,
                    help="run checkpoint eval every N steps (default: same as save_every)")
parser.add_argument("--lr",         type=float, default=3e-4,  help="AdamW lr (embeddings, norms)")
parser.add_argument("--lr_muon",    type=float, default=0.02,  help="Muon lr (2-D weight matrices)")
parser.add_argument("--grad_clip",  type=float, default=1.0)
parser.add_argument("--wandb",      choices=["online", "disabled"], default="online")
parser.add_argument("--bf16",       action="store_true", help="cast model to bfloat16 (required for ~7B on single GPU)")
parser.add_argument("--from_scratch", action="store_true", help="ignore existing checkpoint and train from step 0")
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
enc = tok.load()

DATA_DIR = Path(os.environ.get("DATA_DIR", "data/tokenized"))
_train_path = DATA_DIR / "corpus_train.bin"
_val_path   = DATA_DIR / "corpus_val.bin"
assert _train_path.exists(), f"corpus_train.bin not found in {DATA_DIR}"
assert _val_path.exists(),   f"corpus_val.bin not found in {DATA_DIR}"

# Load pre-shuffled, pre-split corpus — fits in H100 HBM (~2.4GB total)
# Chunks are doc-boundary aligned: no document spans a chunk boundary; gaps filled with EOS.
_seq_len = 6144
train_data = np.fromfile(_train_path, dtype=np.uint16).reshape(-1, _seq_len)
val_data   = np.fromfile(_val_path,   dtype=np.uint16).reshape(-1, _seq_len)

_eos_id = enc.token_to_id("<|eos|>")

def _doc_mask(x: torch.Tensor) -> torch.Tensor:
    """Block-diagonal causal mask: prevents attention across document boundaries.
    x: (B, T) — input token ids (on CPU or CUDA)
    returns: (B, 1, T, T) bool mask, True = allowed to attend

    NOTE: disables FlashAttention kernel in SDPA. For T > ~1024 switch to
    flash_attn_varlen_func which handles packed docs natively via cu_seqlens.
    """
    B, T = x.shape
    # doc_id[b, t] = number of EOS tokens seen before position t
    is_eos = (x == _eos_id)
    doc_id = torch.cat([
        torch.zeros(B, 1, dtype=torch.long, device=x.device),
        is_eos[:, :-1].long().cumsum(dim=1),
    ], dim=1)  # (B, T)
    same_doc = doc_id.unsqueeze(2) == doc_id.unsqueeze(1)          # (B, T, T)
    causal   = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    return (same_doc & causal).unsqueeze(1)                         # (B, 1, T, T)

def get_batch(data, batch_size, block_size):
    """Sample batch_size chunks, slice to block_size, return (x, y, attn_mask)."""
    ix = torch.randint(len(data), (batch_size,))
    chunks = torch.from_numpy(data[ix.numpy()].astype(np.int64))   # (B, _seq_len)
    x = chunks[:, :block_size]
    y = chunks[:, 1:block_size + 1]
    mask = _doc_mask(x)
    if device == "cuda":
        x    = x.pin_memory().to(device, non_blocking=True)
        y    = y.pin_memory().to(device, non_blocking=True)
        mask = mask.pin_memory().to(device, non_blocking=True)
    else:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
    return x, y, mask

# extract document openings from the val split as fixed sample prompts
def _val_prompts(n: int, prompt_tokens: int) -> list[str]:
    eos = enc.token_to_id("<|eos|>")
    flat = val_data.ravel()
    prompts, i = [], 0
    while len(prompts) < n and i < len(flat) - prompt_tokens:
        if flat[i] == eos:
            toks = flat[i+1 : i+1+prompt_tokens].tolist()
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
    model = torch.compile(model)
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
        from huggingface_hub import create_repo
        create_repo(HF_REPO, repo_type="model", exist_ok=True)
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

if not args.from_scratch:
    load_checkpoint()
eval_ckpt.preload()
EVAL_EVERY = args.eval_every if args.eval_every is not None else args.save_every

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
    x, y, attn_mask = get_batch(train_data, args.batch_size, model_cfg.block_size)
    with autocast:
        _, loss = model(x, y, attn_mask)

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
                model(*get_batch(val_data, args.batch_size, model_cfg.block_size))[1].item()
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

    if step % EVAL_EVERY == 0:
        log.update(eval_ckpt.run(model, enc, model_cfg, device, autocast, step))
        model.train()

    wandb.log(log, step=step)

save_checkpoint(args.max_steps - 1)
wandb.finish()
