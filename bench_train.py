"""
Training throughput benchmark — steps/sec, ms/step.

Runs WARMUP steps (discarded), then MEASURE steps (timed).
No wandb, no checkpoints, no val, no sampling — pure loop overhead.

Usage:
    python bench_train.py --model experiment --batch_size 8
"""
import time
import argparse
import contextlib
import numpy as np
import torch
from pathlib import Path

from model import GPT, Config
from optim import Muon

WARMUP  = 10
MEASURE = 50

parser = argparse.ArgumentParser()
parser.add_argument("--model",      choices=["sanity", "experiment", "iphone", "macbook"], default="experiment")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--autocast",   action=argparse.BooleanOptionalAction, default=None,
                    help="force autocast on/off; default: on for cuda, off elsewhere")
args = parser.parse_args()

# ── device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.set_float32_matmul_precision("high")

use_autocast = args.autocast if args.autocast is not None else (device == "cuda")
autocast = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_autocast else contextlib.nullcontext())

# ── data ──────────────────────────────────────────────────────────────────────
assert Path("data_train.bin").exists(), "run data.py first"
tokens = np.memmap("data_train.bin", dtype=np.uint16, mode="r")

def get_batch(batch_size, block_size):
    ix = torch.randint(len(tokens) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(tokens[i  :i+block_size  ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(tokens[i+1:i+block_size+1].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

# ── model + optimizers ────────────────────────────────────────────────────────
cfg = {"sanity": Config.sanity, "experiment": Config.experiment,
       "iphone": Config.iphone, "macbook": Config.macbook}[args.model]()
model = GPT(cfg).to(device)
if device == "cuda":
    model = torch.compile(model)

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

optim_muon = Muon(muon_params, lr=0.02)
optim_adam = torch.optim.AdamW(adam_params, lr=3e-4, betas=(0.9, 0.95),
                                weight_decay=0.1, fused=(device == "cuda"))

print(f"device={device}  model={args.model}  params={model.num_params():,}  "
      f"batch={args.batch_size}  autocast={use_autocast}")
print(f"warmup={WARMUP}  measure={MEASURE}")

# ── benchmark loop ────────────────────────────────────────────────────────────
def step():
    model.train()
    x, y = get_batch(args.batch_size, cfg.block_size)
    with autocast:
        _, loss = model(x, y)
    optim_muon.zero_grad(set_to_none=True)
    optim_adam.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optim_muon.step()
    optim_adam.step()

def sync():
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

print("warming up...")
for _ in range(WARMUP):
    step()
sync()

print("measuring...")
t0 = time.perf_counter()
for _ in range(MEASURE):
    step()
sync()
elapsed = time.perf_counter() - t0

ms_per_step    = elapsed / MEASURE * 1000
steps_per_sec  = MEASURE / elapsed
tokens_per_sec = steps_per_sec * args.batch_size * cfg.block_size

print(f"\n{'ms/step':>12}  {'steps/sec':>10}  {'tok/sec':>10}")
print(f"{ms_per_step:>12.1f}  {steps_per_sec:>10.1f}  {tokens_per_sec:>10,.0f}")
