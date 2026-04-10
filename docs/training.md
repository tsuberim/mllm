# Training

## Remote (RunPod Serverless)

The Docker image bakes only torch + CUDA. Source code is fetched from GitHub at job start — rebuild only when `requirements.txt` changes.

```sh
# Build and push image (only needed when torch/CUDA version changes)
docker build -t tsuberim/merlin-experiment:latest .
docker push tsuberim/merlin-experiment:latest

# Submit a job (commits must be pushed first)
python run_experiment.py --endpoint <RUNPOD_ENDPOINT_ID> \
  --model experiment --batch_size 256 --max_steps 20000

# Prints W&B URL as soon as training starts
```

Set `RUNPOD_API_KEY` in `.env`. The worker expects `corpus_train.bin` + `corpus_val.bin` at `/workspace/data/tokenized/` on the network volume.

## Local Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set `HF_REPO` and `HF_TOKEN` in `.env`. Data is loaded from `data/tokenized/` by default (override with `DATA_DIR` env var).

## Running

```sh
# Local sanity check (2 steps, no CUDA needed)
python train.py --model sanity --batch_size 2 --max_steps 2 --val_every 1 --val_steps 1 --save_every 1 --wandb disabled

# Experiment run (~21M params, local or cheap remote)
python train.py --model experiment --batch_size 256 --max_steps 20000 --val_every 500 --val_steps 10 --save_every 500

# iPhone scale (~3B params, H100 required)
python train.py --model iphone --batch_size 16 --max_steps 100000 --val_every 1000 --val_steps 20 --save_every 1000 --bf16
```

## Hyperparameters

| | sanity | experiment | iphone |
|---|---|---|---|
| params | ~200K | ~21M | ~3.17B |
| batch_size | 2 | 256 | 16 |
| max_steps | 2 | 20000 | 100000 |
| val_every | 1 | 500 | 1000 |
| lr (AdamW) | 3e-4 | 3e-4 | 3e-4 |
| lr (Muon) | 0.02 | 0.02 | 0.02 |
| grad_clip | 1.0 | 1.0 | 1.0 |
| bf16 | no | no | yes |

## Optimisers

Two optimisers running in parallel (see `stack.md` for rationale):
- **Muon** (`lr=0.02`): 2D `Linear.weight` matrices inside transformer blocks
- **AdamW** (`lr=3e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`): embeddings, head, RMSNorm weights

`tok_emb.weight` and `head.weight` are tied — deduplicated by `id()` before assigning to optimisers.

## Data

Corpus: `data/tokenized/corpus_train.bin` + `corpus_val.bin` — see [dataset.md](dataset.md) for full details.

- Loaded fully into memory at startup (~2.4 GB total; fits in H100 HBM)
- `dtype`: uint16, `shape`: `[N, 6144]`
- Each training step samples `batch_size` random rows, slices to `block_size`, builds document-aware attention mask

## Attention Masking

Packed sequences contain multiple documents separated by `<|eos|>`. A block-diagonal causal mask ensures tokens only attend within their own document:

```python
doc_id = torch.cat([zeros, is_eos[:, :-1].cumsum(dim=1)], dim=1)
mask = (doc_id.unsqueeze(2) == doc_id.unsqueeze(1)) & causal  # (B, 1, T, T)
```

## Observability

Each checkpoint logs to wandb (`merlin` project):
- `train/loss`, `val/loss`, `train/grad_norm`
- `samples` table: 4 val-set document openings (20 prompt tokens) + 40 generated tokens
- Checkpoint eval results (see [benchmarking.md](benchmarking.md))

## Checkpoints

Saved to `checkpoints/ckpt_{model}.pt` and pushed to `HF_REPO` on HuggingFace. Training is resumable — picks up from last saved step automatically (checks local first, then HF).
