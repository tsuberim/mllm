# mllm

Build the most efficient small language model that runs locally on Apple Silicon — MacBook Pro and iPhone.
Educational, but aimed at being genuinely useful and open source.

## Goals

- Maximize inference TPS and minimize memory footprint on Apple Silicon (Mac and iPhone)
- Targets: ~3B (iPhone), ~7B (MacBook)
- Knowledge distillation: Qwen2.5-14B teacher → 3B iPhone; Qwen2.5-32B teacher → 7B MacBook; offline logits
- Custom kernels from the start — no relying on framework defaults
- iPhone target: model must fit within ~4 GB RAM; CoreML export path planned

## Stack

- **Training**: PyTorch + CUDA + Triton (NVIDIA)
- **Inference**: MLX (Apple Silicon / Metal) — Mac; CoreML (planned) — iPhone
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

## Research

At the start of each conversation, read `research/index.md` to load context on prior research.

## Non-goals

- Training on MPS
- Maximizing benchmark scores over efficiency
