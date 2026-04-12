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
parser.add_argument("--model",      choices=["sanity", "experiment", "3b", "7b"], required=True)
parser.add_argument("--batch_size", type=int,   required=True)
parser.add_argument("--max_steps",  type=int,   required=True)
parser.add_argument("--val_every",  type=int,   required=True)
parser.add_argument("--val_steps",  type=int,   required=True)
parser.add_argument("--save_every", type=int,   required=True)
parser.add_argument("--eval_every", type=int,   default=None,
                    help="run checkpoint eval every N steps (default: same as save_every)")
parser.add_argument("--lr",         type=float, default=3e-4,  help="AdamW peak lr (embeddings, norms)")
parser.add_argument("--lr_muon",    type=float, default=0.02,  help="Muon peak lr (2-D weight matrices)")
parser.add_argument("--grad_clip",  type=float, default=1.0)
parser.add_argument("--plateau_patience", type=int,   default=1000, help="steps with no improvement before LR decay (converted to val checks internally; 0 = disabled)")
parser.add_argument("--plateau_factor",   type=float, default=0.5,  help="multiply LR by this on plateau")
parser.add_argument("--wandb",      choices=["online", "disabled"], default="online")
parser.add_argument("--bf16",             action="store_true", help="cast model to bfloat16 (required for ~7B on single GPU)")
parser.add_argument("--grad_checkpoint",  action="store_true", help="gradient checkpointing (saves activation memory; enables large batches on big models)")
parser.add_argument("--resume",       action="store_true", help="resume from existing checkpoint")
parser.add_argument("--tag",          type=str, default=None, help="tag for HF checkpoint filename (default: model name)")
args = parser.parse_args()

model_cfg = {"sanity": Config.sanity, "experiment": Config.experiment, "3b": Config.b3, "7b": Config.b7}[args.model]()
Path("checkpoints").mkdir(exist_ok=True)
_tag      = args.tag if args.tag else args.model
CKPT_NAME = f"checkpoints/ckpt_{_tag}.pt"
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

def _ensure_corpus():
    if _train_path.exists() and _val_path.exists():
        return
    from huggingface_hub import hf_hub_download
    import shutil
    HF_CORPUS = "tsuberim/merlin-corpus-v0"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname, path in [("corpus_train.bin", _train_path), ("corpus_val.bin", _val_path)]:
        if not path.exists():
            print(f"[data] downloading {fname} from {HF_CORPUS} ...")
            src = hf_hub_download(HF_CORPUS, f"data/{fname}", repo_type="dataset",
                                  token=os.environ.get("HF_TOKEN"))
            shutil.copy(src, path)
            print(f"[data] {fname} ready ({path.stat().st_size / 1e9:.2f} GB)")

_ensure_corpus()

# Load pre-shuffled, pre-split corpus — fits in H100 HBM (~2.4GB total)
# Chunks are doc-boundary aligned: no document spans a chunk boundary; gaps filled with EOS.
_seq_len = 6144
train_data = np.fromfile(_train_path, dtype=np.uint16).reshape(-1, _seq_len)
val_data   = np.fromfile(_val_path,   dtype=np.uint16).reshape(-1, _seq_len)

def get_batch(data, batch_size, block_size):
    ix = torch.randint(len(data), (batch_size,))
    chunks = torch.from_numpy(data[ix.numpy()].astype(np.int64))   # (B, _seq_len)
    x = chunks[:, :block_size]
    y = chunks[:, 1:block_size + 1]
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# extract document openings from the val split as fixed sample prompts + ground truth continuations
def _val_prompts(n: int, prompt_tokens: int, continuation_tokens: int) -> list[tuple[str, str]]:
    eos = enc.token_to_id("<|eos|>")
    flat = val_data.ravel()
    prompts, i = [], 0
    while len(prompts) < n and i < len(flat) - prompt_tokens - continuation_tokens:
        if flat[i] == eos:
            p_toks = flat[i+1 : i+1+prompt_tokens].tolist()
            c_toks = flat[i+1+prompt_tokens : i+1+prompt_tokens+continuation_tokens].tolist()
            prompts.append((enc.decode(p_toks), enc.decode(c_toks)))
            i += prompt_tokens + continuation_tokens
        else:
            i += 1
    return prompts

SAMPLE_PROMPTS = _val_prompts(N_SAMPLE_PROMPTS, PROMPT_TOKENS, SAMPLE_NEW)

# ── model ─────────────────────────────────────────────────────────────────────
dtype = torch.bfloat16 if args.bf16 else torch.float32
model = GPT(model_cfg).to(device=device, dtype=dtype)
if args.grad_checkpoint:
    model.grad_checkpoint = True
if device == "cuda":
    model = torch.compile(model)
print(f"params: {model.num_params():,}  dtype: {dtype}  grad_checkpoint: {args.grad_checkpoint}")

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

optim_muon = Muon(muon_params, lr=args.lr_muon, weight_decay=0.01)
optim_adam = torch.optim.AdamW(adam_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
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
    return enc.decode(idx[0].tolist()[len(tokens):])

# ── checkpoint ────────────────────────────────────────────────────────────────
hf = HfApi() if args.wandb == "online" else None
start_step = 0

def save_checkpoint(step):
    ckpt = {
        "step":       step,
        "model":      model.state_dict(),
        "optim_muon": optim_muon.state_dict(),
        "optim_adam": optim_adam.state_dict(),
        "sched_adam": _sched_adam.state_dict() if _sched_adam else None,
        "sched_muon": _sched_muon.state_dict() if _sched_muon else None,
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
        if _sched_adam and ckpt.get("sched_adam"):
            _sched_adam.load_state_dict(ckpt["sched_adam"])
            _sched_muon.load_state_dict(ckpt["sched_muon"])
        start_step = ckpt["step"] + 1
        print(f"resumed from step {start_step}")

if args.resume:
    load_checkpoint()
eval_ckpt.preload()
EVAL_EVERY = args.eval_every if args.eval_every is not None else args.save_every

# ── wandb ─────────────────────────────────────────────────────────────────────
wandb.init(
    project="merlin", name=_tag,
    resume="must" if args.resume else "never",
    mode=args.wandb,
    config={**model_cfg.__dict__, "batch_size": args.batch_size,
            "lr": args.lr, "lr_muon": args.lr_muon,
            "weight_decay": 0.01,
            "plateau_patience": args.plateau_patience, "plateau_factor": args.plateau_factor,
            "max_steps": args.max_steps, "device": device},
)

# ── lr schedule ───────────────────────────────────────────────────────────────
_sched_adam = _sched_muon = None
if args.plateau_patience > 0:
    _kw = dict(mode="min", factor=args.plateau_factor, patience=args.plateau_patience)
    _sched_adam = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_adam, **_kw)
    _sched_muon = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_muon, **_kw)

def maybe_decay_lr(val_loss):
    if _sched_adam is None:
        return
    _sched_adam.step(val_loss)
    _sched_muon.step(val_loss)

def current_lrs():
    return (optim_adam.param_groups[0]["lr"],
            optim_muon.param_groups[0]["lr"])

# ── training loop ─────────────────────────────────────────────────────────────
pbar = tqdm(range(start_step, args.max_steps), initial=start_step,
            total=args.max_steps, unit="step")

for step in pbar:
    model.train()
    x, y = get_batch(train_data, args.batch_size, model_cfg.block_size)
    with autocast:
        _, loss = model(x, y)

    optim_muon.zero_grad(set_to_none=True)
    optim_adam.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
    optim_muon.step()
    optim_adam.step()
    maybe_decay_lr(loss.item())
    _lr_now, _lr_muon_now = current_lrs()
    log = {"train/loss": loss.item(), "train/grad_norm": grad_norm,
           "train/lr": _lr_now, "train/lr_muon": _lr_muon_now}

    if step % args.val_every == 0 and step > 0:
        model.eval()
        with torch.no_grad(), autocast:
            val_loss = 0.0
            for _ in range(args.val_steps):
                xv, yv = get_batch(val_data, args.batch_size, model_cfg.block_size)
                val_loss += model(xv, yv)[1].item()
            val_loss /= args.val_steps

        tbl = wandb.Table(columns=["prompt", "completion", "ground_truth"])
        for prompt, ground_truth in SAMPLE_PROMPTS:
            tbl.add_data(prompt, prompt + generate(prompt), prompt + ground_truth)

        log["val/loss"] = val_loss
        log["samples"]  = tbl
        pbar.set_postfix(loss=f"{loss.item():.4f}", val=f"{val_loss:.4f}",
                         gnorm=f"{grad_norm:.2f}")
    else:
        pbar.set_postfix(loss=f"{loss.item():.4f}", gnorm=f"{grad_norm:.2f}")

    if step % args.save_every == 0:
        save_checkpoint(step)

    if step % EVAL_EVERY == 0 and step > 0:
        log.update(eval_ckpt.run(model, enc, model_cfg, device, autocast, step))
        model.train()

    wandb.log(log, step=step)

last_step = args.max_steps - 1
if last_step % args.save_every != 0:
    save_checkpoint(last_step)
wandb.finish()
