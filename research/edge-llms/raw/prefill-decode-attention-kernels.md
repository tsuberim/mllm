# Prefill vs Decode Attention Kernels: Design and Optimization

## Sources
- FlashInfer: https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
- FlashInfer paper: https://arxiv.org/html/2501.01005v1
- FlashAttention-2 GitHub: https://github.com/dao-ailab/flash-attention
- Metal FlashAttention: https://github.com/philipturner/metal-flash-attention
- FlashAttention-4: https://arxiv.org/html/2603.05451v1

---

## Why Prefill and Decode Are Different Problems

### Prefill (processing the prompt)
- Input: all T prompt tokens simultaneously
- Attention: each token attends to all previous tokens (causal mask)
- Computation: full T×T attention matrix (with causal masking → T²/2 elements)
- Regime: **compute-bound** when T is large (arithmetic intensity grows with T)
- Optimization goal: maximize FLOP/s (maximize parallelism, use large tiles)

### Decode (autoregressive generation)
- Input: single new token (Q is shape [1, head_dim])
- Attention: new token attends to full KV cache (all previous T tokens)
- Computation: single row of attention matrix (dot product of Q with all K)
- Regime: **memory-bound** (loading KV cache dominates; only 1 compute per load)
- Optimization goal: maximize memory bandwidth utilization (vectorize loads, minimize round-trips)

---

## FlashAttention Design (Standard)

**Core idea**: Tile the attention computation to avoid materializing the full N×N attention matrix in HBM. Keep tiles in SRAM.

**Algorithm**:
1. Divide Q into row tiles Q_i, K/V into column tiles K_j/V_j
2. For each (i, j) tile pair: compute Q_i·K_j^T → softmax over j → accumulate V_j contribution
3. Use online softmax normalization to handle varying max values across tiles

**Memory savings**: O(N) instead of O(N²) for attention matrix
**Compute**: same FLOPs as standard attention

### FlashAttention prefill optimization
- Large tiles: occupy SRAM with as much of K/V as possible
- Fuse softmax into the tile loop
- Causal mask: skip upper triangle blocks entirely (saves ~50% FLOPs for causal)

### FlashAttention decode optimization (Flash-Decoding)
For decode (Q shape [1, d]):
- Standard tiling doesn't apply (no rows to tile over)
- Split-K: parallelize across K/V sequence dimension
  - Different thread blocks handle different chunks of the KV cache
  - Merge partial softmax results afterward
- Result: multiple thread blocks all loading KV cache in parallel → better bandwidth utilization

---

## FlashInfer (2024): Separate Optimized Kernels

**Source**: arxiv:2501.01005

FlashInfer provides separate kernels for prefill, decode, and append (incremental prefill).

### Decode kernel design
- **Split-K trick**: partitions KV sequence across thread blocks
- **Versatile tile size selection**: unlike FlashAttention's fixed tile, FlashInfer selects optimal tiles for decode's different computation pattern
- **Result**: outperforms FlashAttention kernels for decode because FA uses suboptimal tile size for single-query computation

### Batched decode with variable-length KV caches
- **Dynamic load-balanced scheduler**: assigns KV cache chunks to thread blocks adaptively
- Avoids load imbalance from variable sequence lengths in a batch
- Outperforms standard FA by "large margin" in skewed sequence length distributions

### Performance numbers (from FlashInfer)
- 2.15x improvement in attention kernel efficiency vs FlashAttention
- 1.97x end-to-end throughput enhancement

---

## Metal Implementation Strategy (for mllm)

### Current state
`mx.fast.scaled_dot_product_attention`: A Metal kernel provided by MLX that implements attention with optional causal masking. It handles both prefill and decode cases generically but may not be optimal for either.

### Custom decode kernel opportunity

For mllm decode (batch=1, Q=[1, 12, 64]):

```msl
// Pseudocode for decode attention kernel
kernel void decode_attention(
    device const half* Q,           // [1, 12, 64]
    device const half* K_cache,     // [T, 12, 64] or [T, 4, 64] with GQA
    device const half* V_cache,     // [T, 12, 64] or [T, 4, 64] with GQA
    device half* output,            // [1, 12, 64]
    constant int& seq_len,          // current KV cache length T
    uint head_idx [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Each threadgroup handles one head
    // Threads within threadgroup handle different KV positions in parallel
    
    // Step 1: Compute Q·K^T for all K positions
    // Load K_cache[0..T, head_idx, 0..64] in chunks
    // Dot product Q[0, head_idx, :] with each K[t, head_idx, :]
    
    // Step 2: Softmax (online, via split-K with simd_sum reductions)
    
    // Step 3: Weighted sum of V
}
```

**Key optimizations for this kernel**:
1. Each threadgroup handles one head (head-level parallelism)
2. Within a threadgroup, threads parallelize over sequence positions
3. Use `simd_sum()` for the reduction steps
4. Load K and V in vectorized chunks (float4 or float8 loads)
5. No causal mask needed for decode (all T positions are valid)

### Custom prefill kernel (longer term)

For prefill, the existing `mx.fast.scaled_dot_product_attention` is reasonable. If longer contexts are targeted (>1024 tokens), consider integrating Metal FlashAttention (philipturner/metal-flash-attention) which already implements tiled MFA with 86% ALU utilization on M1 Max.

MFA is a single-head implementation; for multi-head, call it per head or extend to multi-head with a loop.

---

## GQA + Custom Kernel Interaction

When implementing GQA (n_kv_heads=4, n_q_heads=12):

In decode attention, the KV cache has 4 heads instead of 12. Each KV head is shared by 3 Q heads. The decode kernel needs to:
1. For Q head i, use KV head (i // 3)
2. Load the smaller KV cache (3x smaller than MHA)
3. Broadcast the single KV head across the 3 Q heads sharing it

This can be implemented as:
```python
# In MLX
k = mx.repeat(self.k_cache, repeats=n_q_per_kv_group, axis=2)  # expand KV heads
v = mx.repeat(self.v_cache, repeats=n_q_per_kv_group, axis=2)
scores = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
```

Or in the custom Metal kernel, index the KV head as `head_idx // heads_per_kv_group`.

---

## Expected Impact of Separate Prefill/Decode Kernels

For mllm:
- **Decode** (current bottleneck at 625 TPS): a dedicated GEMV-style decode kernel could improve utilization from ~36% to ~60-70% of memory bandwidth → **1.5-2x speedup** on decode
- **Prefill**: tiled attention (MFA-style) would improve prefill speed, which affects time-to-first-token. Less impactful for user experience than decode speed.

**Implementation effort**: Moderate. The decode kernel is ~100 lines of MSL. The prefill kernel can reuse MFA (external library).

**Priority**: After GQA and RoPE, before LUT dequantization (which is likely not beneficial on Metal GPU).
