# Stack

## Training: PyTorch + CUDA (NVIDIA)

- Full PyTorch ecosystem: autograd, `torch.compile`, Triton custom kernels
- `scaled_dot_product_attention` (fused CUDA kernel via PyTorch)
- AdamW optimizer (Muon planned — see below)
- Checkpoints saved to local disk + HuggingFace Hub

## Inference: MLX (Apple Silicon / Metal)

- Apple's ML framework, native Metal kernels, lazy evaluation, unified memory
- Custom Metal kernel: fused SwiGLU (one pass instead of two)
- `mx.fast.rms_norm`, `mx.fast.scaled_dot_product_attention` for other hot paths
- int4 block-wise quantization via `mlx.nn.quantize` (dequant-on-the-fly)
- KV cache: prefill once, O(1) decode per token

## Bridge: convert.py

PyTorch checkpoint → MLX `.npz`. Weight tying is made explicit (head.weight saved separately). Parity tested at `atol=1e-6`.

## Why the split?

PyTorch MPS (Apple GPU backend) has silent CPU fallbacks, no `torch.compile`, no int4 quantization, no FlashAttention. Training on MPS is explicitly out of scope. MLX is purpose-built for Apple Silicon inference and gives clean access to Metal kernels.

## Muon optimizer (planned)

Newton-Schulz orthogonalised momentum for 2D weight matrices in transformer blocks. Reference implementation is a ~40-line snippet (no stable PyPI package). Deferred until training baseline is established.
