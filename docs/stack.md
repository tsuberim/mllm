# Stack

## Training: PyTorch + CUDA (NVIDIA)

- Full PyTorch ecosystem: autograd, `torch.compile`, Triton custom kernels
- `scaled_dot_product_attention` (fused CUDA kernel via PyTorch)
- AdamW optimizer (Muon planned — see below)
- Checkpoints saved to local disk + HuggingFace Hub

## Inference: MLX (Mac) + CoreML (iPhone, planned)

- Apple's ML framework, native Metal kernels, lazy evaluation, unified memory
- Custom Metal kernel: fused SwiGLU (one pass instead of two)
- `mx.fast.rms_norm`, `mx.fast.scaled_dot_product_attention` for other hot paths
- int4 block-wise quantization via `mlx.nn.quantize` (dequant-on-the-fly)
- KV cache: prefill once, O(1) decode per token
- iPhone target: model must fit within ~4 GB; int4 base model is 188 MB — well within budget
- CoreML export path: convert MLX weights → CoreML via `coremltools`; enables on-device iOS deployment

## Bridge: convert.py

PyTorch checkpoint → MLX `.npz`. Weight tying is made explicit (head.weight saved separately). Parity tested at `atol=1e-6`.

## Why the split?

PyTorch MPS (Apple GPU backend) has silent CPU fallbacks, no `torch.compile`, no int4 quantization, no FlashAttention. Training on MPS is explicitly out of scope. MLX is purpose-built for Apple Silicon inference and gives clean access to Metal kernels.

## Muon optimizer

Newton-Schulz orthogonalised momentum for 2D weight matrices in transformer blocks.
Implemented directly (~50 lines, no external package needed).

- **Muon** (`lr=0.02`): all `Linear.weight` inside transformer blocks (qkv, proj, w1/w2/w3).
- **AdamW** (`lr=3e-4`): 1D params (RMSNorm weights) and embeddings/head.

Split is forced by the algorithm: Newton-Schulz orthogonalization only makes sense for 2D matrices.
