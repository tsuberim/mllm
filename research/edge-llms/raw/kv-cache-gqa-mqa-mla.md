# KV Cache Optimizations: GQA, MQA, MLA, Paged Attention, Sliding Window

## Sources
- GQA paper: https://arxiv.org/abs/2305.13245 (EMNLP 2023)
- DeepSeek-V2 (MLA): https://arxiv.org/abs/2405.04434
- PagedAttention/vLLM: https://arxiv.org/abs/2309.06180
- Mistral sliding window: https://arxiv.org/abs/2310.06825

---

## Multi-Query Attention (MQA)

All query heads share a single key-value head. Drastically reduces KV cache size (by n_heads factor) and memory bandwidth during decode. Trade-off: quality degradation, especially on long contexts.

**KV cache size**: reduced by factor of n_heads vs MHA.

---

## Grouped-Query Attention (GQA)

**Paper**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., EMNLP 2023)

Generalizes MQA: G groups, each group shares one KV head. G=1 → MQA, G=n_heads → MHA.

**Key results**:
- "Uptrained GQA achieves quality close to multi-head attention with comparable speed to MQA"
- Can uptrain existing MHA checkpoint using only 5% of original pre-training compute
- Memory reduction: n_heads / G factor for KV cache
- Adopted by Llama 2, Llama 3, Qwen2, Mistral as standard
- SmolLM2-135M uses GQA: 9 query heads, 3 KV heads (3:1 ratio → 3x KV cache reduction)

**Quality trade-off**: GQA-8 (8 groups) essentially matches MHA quality; G=1 (MQA) shows noticeable degradation on some tasks.

**Inference relevance**: During decode, reducing KV heads reduces memory bandwidth consumed loading KV cache. At small batch sizes (single user), this is the dominant bottleneck on Apple Silicon.

---

## Multi-Head Latent Attention (MLA)

**Paper**: DeepSeek-V2 (arxiv:2405.04434, May 2024)

Instead of caching K and V separately, MLA caches a low-rank latent vector c_KV and decompresses to K,V at attention time.

**Mechanism**:
1. Compress: c_KV = W_DKV * h  (low-rank projection down)
2. Cache c_KV instead of full K,V
3. Decompress: K,V = W_UK * c_KV, W_UV * c_KV (at attention time)

**Results**:
- KV cache reduced by **93.3%** vs standard MHA
- Maximum generation throughput **5.76x** higher than DeepSeek-67B
- Quality **better than MHA** (not just equal) — likely due to larger effective model for same parameter budget
- Compute overhead: decompress ops at attention time (small relative to other ops)

**Feasibility at 117M scale**: High benefit if memory-bound. Implementation complexity is moderate — need to change attention layer, add projection matrices. KV cache for 117M with MHA at 1024 context is already small (~48MB at fp16), so the benefit is less dramatic than at large scale. However, if doing long context or batched inference, MLA would help.

---

## PagedAttention

**Paper**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (arxiv:2309.06180)

Inspired by OS virtual memory paging. KV cache partitioned into fixed-size blocks. Eliminates fragmentation.

**Results**:
- Near-zero KV cache waste (<4% vs 60-80% in naive systems)
- 2-4x throughput improvement for batched serving
- Enables KV cache sharing across requests (e.g., system prompt reuse)

**Relevance to mllm**: Paged attention is a serving optimization for multi-request scenarios. For single-user inference on Mac, the existing KV cache (O(1) decode) is already optimal. PagedAttention would matter if mllm is extended to serve multiple users.

---

## Sliding Window Attention (SWA)

**Paper**: Mistral 7B (arxiv:2310.06825)

Each attention layer only attends to the last `window_size` tokens rather than full context.

**Key details**:
- Mistral uses window_size=4096
- KV cache is bounded at window_size * n_layers (rotating buffer)
- Higher layers can still access information from beyond the window via layer stacking (each layer sees `window_size` tokens from the layer below, which itself saw more)
- Combined with GQA: 8x smaller KV cache than Llama-2 at 8K tokens (4x from GQA + 2x from capped context)

**Quality**: Works well for tasks not requiring very long-range dependencies. Not suitable if the model needs to reference tokens far back in context.

**Relevance to mllm**: The current block_size is 1024 and the model is 117M. SWA would allow extending context without linear KV cache growth. Worth considering for a future "medium" or "large" variant but likely overkill for 117M at 1024 context.

---

## Summary Table

| Method | KV Cache Reduction | Quality vs MHA | Impl Complexity | Best Use Case |
|--------|-------------------|----------------|-----------------|---------------|
| MQA    | n_heads × smaller | Moderate loss  | Low             | Very tight memory |
| GQA    | Tunable (2-8x)    | Near-equal     | Low             | Best balance |
| MLA    | 93.3%             | Better         | Moderate        | Long context / large scale |
| SWA    | Bounded by window | Context-dependent | Low           | Long context |
| Paged  | Fragmentation fix | No change      | High            | Multi-user serving |

## Recommendation for mllm

GQA is the highest-priority change. Switch from vanilla MHA (12 heads) to GQA with n_kv_heads=4 (3:1 ratio). This reduces KV cache by 3x and reduces memory bandwidth during decode by 3x. Quality loss at 117M scale is minimal based on SmolLM2's success with similar ratios.

**Key implementation note**: `mx.fast.scaled_dot_product_attention` natively supports GQA and MQA — k and v tensors with fewer heads than q are handled internally without pre-tiling. The API signature is:
```python
# q: [B, n_q_heads, T_q, head_dim]
# k: [B, n_kv_heads, T_kv, head_dim]  ← smaller, no expansion needed
# v: [B, n_kv_heads, T_kv, head_dim]
out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
```
This means GQA requires only: (1) change attention weight shapes for K and V projections, (2) separate KV cache from Q cache in size. Zero changes to the attention kernel itself.
