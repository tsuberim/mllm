# Training

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data.py   # tokenizes TinyStories → data_train.bin, data_validation.bin (~474M tokens)
```

## Running

```sh
./train_sanity.sh     # 2 steps, batch 2 — pipeline sanity check only
./train_experiment.sh # 2k steps, batch 8, MPS-safe (~10 min locally)
./train_iphone.sh     # iphone scale — needs NVIDIA
```

## Hyperparameters

| | sanity | experiment |
|---|---|---|
| batch_size | 2 | 8 |
| max_steps | 2 | 2000 |
| val_every | 1 | 200 |
| save_every | 1 | 200 |
| lr | 3e-4 | 3e-4 |
| wandb | disabled | online |

## Observability

Each val checkpoint logs to wandb:
- `train/loss`, `val/loss`
- `train/grad_norm`
- `samples` table: 4 real val-set story openings (20 prompt tokens) + 20 generated tokens

## Checkpoints

Saved locally as `ckpt_{model}.pt` and pushed to `tsuberim/merlin` on HuggingFace. Set `HF_REPO` in `.env`. Resumable — training picks up from last saved step.

## Data

TinyStories (roneneldan/TinyStories): 2.1M short children's stories, 474M train tokens, 4.7M val tokens. Tokenized with tiktoken GPT-2 encoding (vocab size 50257), stored as uint16 memmap.
