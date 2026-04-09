# Inference

## Quick start

```sh
./sample.sh "prompt here" 200   # convert latest ckpt_iphone.pt + generate
python bench.py                 # benchmark iphone model
python bench.py --bits 4        # int4 quantized
```

## Pipeline

1. **Convert**: `python convert.py ckpt_iphone.pt weights_iphone.npz`
2. **Infer**: `python infer.py --model iphone --weights weights_iphone.npz --prompt "..."`
3. **Bench**: `python bench.py --model experiment --weights checkpoints/weights.npz --bits 4`

## Benchmarks (experiment model, ~21M params, M4 MPS)

| Setup | TPS | Peak mem |
|-------|-----|----------|
| fp32 + KV cache + GQA | 946.8 ± 38.3 | 131 MB |
| fp16 + KV cache + GQA | 1003.0 ± 72.5 | 66 MB |
| int4 + KV cache + GQA | 999.1 ± 24.1 | 24 MB |

Note: figures are 3-run averages (`python bench.py --runs 3`). At 21M params the model is overhead-bound (kernel launch, not memory bandwidth), so fp16/int4 TPS is similar — memory savings are the real gain. At 3B+ params, int4 gives ~3× TPS vs fp32 as bandwidth dominates.

## KV cache

Prefill: process full prompt once, build K/V cache per layer.
Decode: pass only the new token each step — O(1) attention instead of O(n²).
Positional embeddings use the correct offset derived from cache size.

## Attention

`mx.fast.scaled_dot_product_attention(q, k, v, scale=s, mask="causal")` replaces the manual `q @ k^T → softmax → @ v` path. MLX selects `sdpa_vector` for decode (T=1, head_dim=64) — online softmax, no intermediate `[B,H,T,T]` score matrix. ~8–11% TPS gain vs manual attention.

## KV cache

Pre-allocated `[1, H, max_T, D]` buffer per layer, updated via slice assignment each decode step. Eliminates the per-step `mx.concatenate` that previously allocated a new tensor and copied the full cache every token. +22% TPS vs growing-concatenate at n_tokens=500; gain scales with context length (2× at 1500 tokens). Memory footprint is now fixed at model load time rather than growing during generation.

## Positional encoding: RoPE

Rotary Position Embeddings (Su et al., 2023) applied to Q and K in each attention layer.
Implemented as a fused Metal kernel (`rope2`) that processes both Q and K in one dispatch per layer.

Key implementation details:
- Module-level frequency cache — all layers share one `[block_size, head_dim]` cos/sin allocation.
- Fresh slice per step — slices are NOT cached; stale MLX lazy-node accumulation over a 500-token decode sequence caused ~18% TPS regression until this was identified and fixed.
- Net cost: ~6% TPS overhead vs learned pos_emb, in exchange for better context-length extrapolation.

## Quantization

`--bits 16`: weights cast to float16 at load time; halves model memory with no quality loss. KV cache dtype follows model weights automatically.

`--bits 4/8`: `mlx.nn.quantize(model, group_size=64, bits=N)` converts all `nn.Linear` layers to block-wise int4/8. Weights stored as packed uint32; dequantization happens on-the-fly inside the Metal matmul kernel. Required to fit the 3B iphone model in 4 GB RAM.

## Custom kernels

**Fused SwiGLU** (`infer.py`): reads gate and up tensors once, writes `silu(gate)*up` in a single Metal kernel. Avoids one global memory round-trip vs separate silu + multiply ops.

**Fused Q+K RoPE** (`infer.py`): applies rotary embeddings to both Q and K in one Metal dispatch per layer. Uses `simd_sum` for intra-warp reduction. Halves the dispatch count vs separate Q and K kernels.

Parity tested against PyTorch CPU at `atol=1e-6` (max observed diff ~5e-7, from `exp` rounding differences between CPU and Metal math libraries).

### Planned

- ~~Pre-allocated KV cache~~ — done, +22% TPS.
- ~~`mx.compile()` on MLP per block~~ — done, +5% TPS, 18× lower variance. Full-model compilation blocked by Python-offset KVCache (changing slice indices retrace graph each step); MLP is the compilable sweet spot.
- ~~GQA (n_kv_heads=4)~~ — done, +24% TPS (242.6 → 301.8 fp32), 3× smaller KV cache; rope2 kernel extended to handle GQA in one dispatch.
- ~~Float16 inference~~ — done (`--bits 16`); halves peak memory (131 → 66 MB); TPS parity with int4 at experiment scale (both overhead-bound); at iphone (3B) scale bandwidth dominates and int4 gives ~3× over fp32.
- Dedicated decode GEMV kernel: at experiment scale already overhead-bound; would matter at iphone (3B) scale once benchmarked on real weights.
- Custom int4 GEMV: naive simdgroup-reduction kernel benchmarked at parity with MLX's built-in. MLX's `quantized_matmul` is already close to bandwidth-optimal. Deferred.
- LUT dequantization: cache-unfriendly on Apple unified memory — confirmed slower than scale+bias on M-series. Will not implement.
