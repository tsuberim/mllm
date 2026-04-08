# mllm

Build the most efficient small language model that runs locally on a MacBook Pro.
Educational, but aimed at being genuinely useful and open source.

## Goals

- Maximize inference TPS and minimize memory footprint on Apple Silicon
- Train from scratch
- Custom kernels from the start — no relying on framework defaults

## Stack

- **Training**: PyTorch + CUDA + Triton (NVIDIA)
- **Inference**: MLX (Apple Silicon / Metal)
- **Bridge**: weight conversion + parity tests; PyTorch is source of truth

## Setup

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Docs

Keep `docs/` up-to-date whenever anything changes:
- `docs/architecture.md` — model design and config table
- `docs/stack.md` — training/inference split rationale, planned work
- `docs/training.md` — hyperparams, data, observability
- `docs/inference.md` — benchmarks, kernels, quantization

Update benchmarks whenever a new optimisation is measured. Update planned-work sections when something ships.

## Non-goals

- Training on MPS
- Maximizing benchmark scores over efficiency
