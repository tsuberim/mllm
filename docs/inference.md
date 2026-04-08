# Inference

## Quick start

```sh
./sample.sh "prompt here" 200   # convert latest ckpt_medium.pt + generate
python bench.py                 # benchmark base model (base_random.npz)
python bench.py --bits 4        # int4 quantized
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
| int4 + KV cache | ~580 | 188 MB |

Note: int4 TPS varies ~10% run-to-run due to system scheduling; 580 is a representative figure.

## KV cache

Prefill: process full prompt once, build K/V cache per layer.
Decode: pass only the new token each step — O(1) attention instead of O(n²).
Positional embeddings use the correct offset derived from cache size.

## Positional encoding: RoPE

Rotary Position Embeddings (Su et al., 2023) applied to Q and K in each attention layer.
Implemented as a fused Metal kernel (`rope2`) that processes both Q and K in one dispatch per layer.

Key implementation details:
- Module-level frequency cache — all layers share one `[block_size, head_dim]` cos/sin allocation.
- Fresh slice per step — slices are NOT cached; stale MLX lazy-node accumulation over a 500-token decode sequence caused ~18% TPS regression until this was identified and fixed.
- Net cost: ~6% TPS overhead vs learned pos_emb, in exchange for better context-length extrapolation.

## Quantization

`mlx.nn.quantize(model, group_size=64, bits=4)` converts all `nn.Linear` layers to block-wise int4. Weights are stored as packed uint32; dequantization happens on-the-fly inside the Metal matmul kernel (no separate dequant pass). Run with `--bits 4` or `--bits 8`.

## Custom kernels

**Fused SwiGLU** (`infer.py`): reads gate and up tensors once, writes `silu(gate)*up` in a single Metal kernel. Avoids one global memory round-trip vs separate silu + multiply ops.

**Fused Q+K RoPE** (`infer.py`): applies rotary embeddings to both Q and K in one Metal dispatch per layer. Uses `simd_sum` for intra-warp reduction. Halves the dispatch count vs separate Q and K kernels.

Parity tested against PyTorch CPU at `atol=1e-6` (max observed diff ~5e-7, from `exp` rounding differences between CPU and Metal math libraries).

### Planned

- LUT dequantization: precompute 16-entry float table per block, replace multiply-add with register lookup.
- Separate prefill/decode attention kernels: prefill is compute-bound, decode is memory-bound — different optimal tiling strategies.
- Flash attention for prefill: fused QK^T matmul + softmax + V matmul to reduce HBM traffic.
- Custom int4 GEMV: naive simdgroup-reduction kernel benchmarked at parity with MLX's built-in (~490 TPS both ways). MLX's `quantized_matmul` is already close to bandwidth-optimal. A threadgroup-tiled kernel that maximises DRAM burst width could still help; deferred.
