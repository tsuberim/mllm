# Merlin

An efficient small language model built from scratch for local inference on Apple Silicon — MacBook Pro and iPhone.

Educational in spirit, but aimed at being genuinely useful and open source.

## Goals

- Maximize inference TPS and minimize memory on Apple Silicon (Mac + iPhone)
- Train from scratch on real data
- Custom Metal kernels — no relying on framework defaults
- iPhone target: model fits within ~4 GB RAM via int4 quantization

## Benchmarks

Base model (~117M params) on M4 MacBook Pro:

| Setup | TPS | Peak Memory |
|---|---|---|
| fp32, no KV cache | 38.7 | 1536 MB |
| fp32 + KV cache | 242.6 | 802 MB |
| **int4 + KV cache** | **625.3** | **188 MB** |

int4 + KV cache hits 625 TPS at 188 MB — well within iPhone's ~4 GB budget.

## Architecture

GPT-style decoder-only transformer with four configs:

| Config | Params | n_embd | n_head | n_layer | block_size |
|---|---|---|---|---|---|
| sanity | ~1.6M | 32 | 2 | 2 | 64 |
| experiment | ~21M | 256 | 8 | 8 | 512 |
| iphone | ~3.17B | 3072 | 24 | 20 | 4096 |
| macbook | ~7.19B | 4096 | 32 | 26 | 4096 |

Key design choices:
- **RMSNorm** — faster, no mean subtraction
- **SwiGLU MLP** — better loss/param ratio vs GELU
- **Weight tying** — tok_emb and head share weights; saves ~39M params on base
- **No bias** on linear layers
- **Pre-norm** — more stable training

## Stack

| Role | Tool |
|---|---|
| Training | PyTorch + CUDA + Triton (NVIDIA) |
| Inference (Mac) | MLX + custom Metal kernels |
| Inference (iPhone) | CoreML (planned) |
| Data | TinyStories via tiktoken GPT-2 |
| Observability | W&B |

PyTorch is the source of truth. MLX is inference-only with custom kernels. Weight conversion is explicit and parity-tested.

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Training

```sh
# Sanity check (2 steps, ~seconds)
./train_sanity.sh

# Local experiment (2000 steps, ~10 min on MPS)
./train_experiment.sh

# iPhone scale (requires NVIDIA)
./train_iphone.sh
```

## Inference

```sh
# Convert latest checkpoint and sample
./sample.sh

# Benchmark quantization levels
python bench.py
```

## Project Layout

```
model.py      — PyTorch transformer (training)
infer.py      — MLX inference with KV cache + int4 quant + custom kernels
train.py      — Training loop (AdamW, grad clipping, W&B, HF Hub checkpoints)
data.py       — TinyStories tokenization → memmap
convert.py    — PyTorch → MLX weight conversion
bench.py      — TPS and memory benchmarks
test_e2e.py   — PyTorch/MLX parity tests (atol=1e-6, greedy token matching)
docs/         — Architecture, stack, training, inference docs
```

## Planned

- Muon optimizer (Newton-Schulz orthogonalised momentum)
- Tiled int4 matmul Metal kernel (3–5× speedup potential)
- Separate prefill / decode kernels (different optimal tiling per phase)
- RoPE (replace learned positional embeddings)
- CoreML export for iPhone deployment

## License

MIT
