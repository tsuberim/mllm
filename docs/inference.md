# Inference

## Quick start

```sh
./sample.sh "prompt here" 200   # convert latest ckpt_medium.pt + generate
python bench.py                 # benchmark base model (base_random.npz)
```

## Pipeline

1. **Convert**: `python convert.py ckpt_medium.pt weights_medium.npz`
2. **Infer**: `python infer.py --model medium --weights weights_medium.npz --prompt "..."`
3. **Bench**: `python bench.py --model base --weights base_random.npz --bits 4`

## Benchmarks (base model, ~117M params, M4 MPS)

| Setup | TPS | Peak mem |
|-------|-----|----------|
| fp32, no KV cache | 38.7 | 1536 MB |
| fp32 + KV cache | 242.6 | 802 MB |
| int4 + KV cache | 625.3 | 188 MB |

## KV cache

Prefill: process full prompt once, build K/V cache per layer.
Decode: pass only the new token each step — O(1) attention instead of O(n²).
Positional embeddings use the correct offset derived from cache size.

## Quantization

`mlx.nn.quantize(model, group_size=64, bits=4)` converts all `nn.Linear` layers to block-wise int4. Weights are stored as packed uint32; dequantization happens on-the-fly inside the Metal matmul kernel (no separate dequant pass). Run with `--bits 4` or `--bits 8`.

## Custom kernels

**Fused SwiGLU** (`infer.py`): reads gate and up tensors once, writes `silu(gate)*up` in a single Metal kernel. Avoids one global memory round-trip vs separate silu + multiply ops.

Parity tested against PyTorch CPU at `atol=1e-6` (max observed diff ~5e-7, from `exp` rounding differences between CPU and Metal math libraries).

### Planned

- LUT dequantization: precompute 16-entry float table per block, replace multiply-add with register lookup. Used in llama.cpp Metal kernels, potentially faster on Apple Silicon.
- Tiled int4 matmul: load int4 tiles into threadgroup memory, dequantize in-register, multiply-accumulate. 3-5× speedup vs naive matmul.
- Separate prefill/decode attention kernels: prefill is compute-bound, decode is memory-bound — different optimal tiling strategies.
