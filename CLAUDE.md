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

## Non-goals

- Training on MPS
- Maximizing benchmark scores over efficiency
