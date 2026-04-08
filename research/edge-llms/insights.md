# Edge LLM Inference: Synthesized Insights for mllm

Research date: April 2026. Focus: 117M-param GPT-style decoder targeting MLX (Mac M-series) and CoreML (iPhone).

Current baseline: 625 TPS on M4 with int4 block-wise quantization + KV cache.

---

## Theoretical Ceiling Analysis

**mllm at decode time is deeply memory-bound.** The roofline model on M4 (120 GB/s):

```
Model weight size (int4): ~70MB
Memory bandwidth limit: 120 GB/s / 70MB = ~1,714 TPS theoretical maximum
Current: 625 TPS → 36% of theoretical maximum
```

Three paths to close this gap:
1. Reduce overhead per step (kernel launch, Python dispatch, Metal dispatch)
2. Reduce effective bytes per step (fuse more ops, reduce KV cache bandwidth)
3. Increase parallelism (batching, but breaks single-user latency model)

---

## Changes Ranked by Expected Impact

### Tier 1: High impact, low-to-moderate effort

**1. Grouped-Query Attention (GQA): n_kv_heads=4**
- Estimated TPS gain: **20-40%** (reduces KV cache bandwidth 3x, which contributes ~5-10% of total bandwidth)
- Quality impact: minimal. SmolLM2-135M uses 9q/3kv at same scale. Quality loss vs MHA is <1%.
- Implementation: change attention weight shapes for K/V projections; `mx.fast.scaled_dot_product_attention` natively supports GQA (passes k/v with fewer heads than q, handles broadcast internally — no pre-tiling or repeat needed)
- Also: KV cache size: 37.7MB → 12.6MB (1024 context, fp16). Much better for iPhone's memory budget.
- Source: GQA paper (arxiv:2305.13245); SmolLM2 config (arxiv:2502.02737)

**2. RoPE Positional Embeddings**
- Quality gain: ~1.8% better validation loss at 125M scale (from EleutherAI benchmark)
- Training efficiency: ~30% faster convergence (same quality in fewer steps)
- Inference cost: 1-3% overhead (absorbed by other costs, effectively free)
- Context extension: enables 1K → 4K+ tokens with RoPE scaling (impossible with learned PE)
- Implementation: use `mx.fast.rope()` (fused Metal kernel, already in MLX)
- Requires re-training from scratch (breaking change)
- Source: EleutherAI RoPE blog; RoFormer (arxiv:2104.09864)

**3. `mx.compile()` on inference forward pass**
- Estimated TPS gain: **10-30%** (reduces Python/Metal dispatch overhead)
- Zero quality impact
- Implementation: wrap `model.generate_token()` call with `@mx.compile`
- Zero model changes required
- Source: MLX documentation; expected based on framework overhead at 117M scale

### Tier 2: Moderate impact, moderate effort

**4. Dedicated Decode Attention Kernel (Metal)**
- Estimated TPS gain: **30-60%** on attention specifically; **10-20%** end-to-end
- The decode phase is a GEMV (matrix-vector): Q=[1, 12, 64] vs KV=[T, 12, 64]. A dedicated kernel can vectorize KV loads, use SIMD reductions, and skip causal mask check (always no-op at decode).
- Implementation: ~100 lines of MSL, integrate via `mx.fast.metal_kernel()`
- Prerequisite: implement GQA first so KV cache is already 3x smaller
- Reference: FlashInfer design (arxiv:2501.01005); Metal FlashAttention internals

**5. AWQ Quantization (Activation-aware)**
- Quality gain: ~0.5-2 perplexity points improvement over current RTN int4
- No change in inference speed or memory (same int4 format)
- Small models benefit more from calibration-based quantization (each parameter counts more)
- Implementation: run AWQ calibration on ~100 tokens of representative data; fold scaling into adjacent layers; re-export as int4
- Source: AWQ paper (arxiv:2306.00978, MLSys 2024 Best Paper)

**6. KV Cache Quantization (int8)**
- After GQA (KV cache = 12.6MB): quantizing KV to int8 → 6.3MB
- Estimated TPS gain: **5-15%** (reduces KV bandwidth by 2x)
- Quality: very minor degradation for cached K,V (they're already smoothed by the attention distribution)
- Implementation: store KV in int8 with per-token scale; dequantize at attention time
- Source: TurboQuant MLX (github.com/sharpner/turboquant-mlx); general KV compression literature

### Tier 3: Lower priority or higher effort

**7. Muon Optimizer (Training)**
- Training efficiency gain: **~30-35%** fewer steps to converge vs AdamW
- Zero inference impact (only affects training)
- Overhead during training: <0.1% FLOP overhead for 117M model
- Implementation: now in `torch.optim.Muon` (PyTorch 2.11). Apply to 2D weight matrices; use AdamW for embeddings and heads.
- Source: Keller Jordan blog; arxiv:2502.16982

**8. CoreML Export (iPhone)**
- Not a TPS optimization for MLX — a separate target
- Enables iPhone deployment (117M at int4 ≈ 70MB; fits comfortably in iPhone memory)
- Workflow: PyTorch → `torch.jit.trace` → `ct.convert()` → int4 block-wise quant (block_size=32) → stateful KV cache (iOS 18+)
- Expected: >100 TPS on iPhone 16 Pro (much smaller than Llama 3.1 8B's 33 TPS on Mac)
- Key challenge: custom attention implementation for in-place KV state update
- Source: Apple CoreML Llama blog; HuggingFace Mistral CoreML blog

**9. simdgroup_matrix (SIMD Group Matrix Tiles, Metal 3)**
- For a custom tiled prefill attention or matmul kernel, use `simdgroup_matrix` (Metal 3, M2+) for hardware-accelerated 8×8 FP16 matrix ops
- Not exposed via MLX Python API — requires custom MSL kernel
- M2 GPU: 6.8 TFLOPS FP16 (M2 Pro). At 86% ALU utilization (Metal FlashAttention's number): ~5.8 TFLOPS effective
- Priority: only worth implementing if prefill speed becomes a bottleneck (it currently is not for 117M)

**10. Speculative Decoding**
- Not recommended at 117M scale now
- The target model is too small to have a meaningfully cheaper draft
- Self-speculative (EAGLE-style): draft head would be ~17% of model compute, needs >120% acceptance to break even
- Revisit if model grows to 500M+ parameters
- Source: EAGLE paper (arxiv:2401.15077); Kangaroo (OpenReview 2024)

---

## What NOT to Do (Counterintuitive Findings)

**Don't use LUT-based dequantization on Metal GPU**: LUT approaches (K-quant in llama.cpp, LUT-GEMM) that outperform on CUDA are actually slower on Apple Silicon Metal. The GPU's cache doesn't handle random LUT access patterns well. llama.cpp's Q4_K_M (LUT-based) is noticeably slower than Q4_0 (scale+bias) on M-series. MLX's scale+bias format is already better-suited.

**Don't use SmoothQuant yet**: W8A8 is only hardware-accelerated on M4/A17 Pro Neural Engine, not Metal GPU. For the GPU inference path (which is what matters for 117M LLM), W4A16 (current MLX format) is better than W8A8.

**Don't switch to Mamba/linear attention**: At 117M and 1024 token context, transformer attention is only ~16% of compute. The O(N²) scaling is not yet a problem. Mamba would add implementation complexity for minimal benefit.

**Don't implement PagedAttention**: PagedAttention solves memory fragmentation for multi-request serving. Single-user Mac inference is already well-handled by the current KV cache implementation.

**Don't implement K-quant (codebook) LUT formats**: See point 1. Wrong for Apple Silicon.

---

## Architecture Recommendations

### Short term (next 3 months)
1. Add `mx.compile()` to inference loop (zero model changes, 10-30% gain)
2. Implement GQA with n_kv_heads=4 — requires architecture change, re-training
3. Switch to RoPE — requires re-training, combine with GQA in same re-train

### Medium term (3-6 months)
4. Run AWQ calibration on int4 weights
5. Implement dedicated decode attention Metal kernel (GEMV-style)
6. Quantize KV cache to int8
7. Use Muon for training (switch optimizer when next training run starts)

### Long term (6-12 months)
8. CoreML export path for iPhone (iOS 18+ stateful KV cache)
9. Extend context to 4K via RoPE scaling + continued training
10. Consider EAGLE-style speculative decoding if model grows to 500M+

---

## Combined Expected TPS Impact

Starting from 625 TPS (M4 base, 120 GB/s):

| Change | Expected Factor | Cumulative TPS |
|--------|----------------|----------------|
| Baseline | — | 625 |
| + mx.compile() | 1.15× | 719 |
| + GQA-4 (decode phase KV BW) | 1.15× | 827 |
| + Dedicated decode kernel | 1.20× | 992 |
| + KV cache int8 | 1.10× | 1,091 |

**Target: >1,000 TPS on M4 base** is achievable with these changes.
Upper bound is ~1,714 TPS (memory bandwidth ceiling at int4 for 117M). Achieving 60-70% of ceiling (1,000-1,200 TPS) is realistic.

---

## CoreML iPhone Path: Key Facts

- 117M model at int4 = ~70MB → fits in 500MB budget easily on any modern iPhone
- iOS 18 + macOS 15: stateful KV cache (non-negotiable for usable perf; 13x speedup over pass-as-I/O)
- Minimum deployment target: iOS 18 (limits to iPhone XS and newer, effectively iPhone XR/11+)
- Conversion path: PyTorch → torch.jit.trace → coremltools → int4 block-wise (block_size=32)
- Custom attention needed: in-place KV state update, 4D tensor layout (B, C, 1, S)
- ANE (Neural Engine) vs GPU: GPU path is likely better for this model due to autoregressive nature; ANE needs additional adaptation
- Expected iPhone 16 Pro TPS: 200-500+ (model is 50x smaller than Mistral 7B which gets 33 TPS at 8B params)

---

## Open Questions

1. **GQA quality at 117M with MHA pre-training**: Can we uptrain the existing MHA checkpoint to GQA (using 5% training compute as in the GQA paper), or do we need full re-training? Given the simultaneous need to add RoPE, a full re-train is likely unavoidable.

2. **Optimal n_kv_heads at 117M**: Is 4 KV heads (3:1 ratio) or 3 KV heads (4:1 ratio) the better tradeoff at this scale? SmolLM2 uses 3:1; Llama-3 uses 4:1 (8Q/2KV for 8B). Ablation needed.

3. **AWQ calibration data for mllm**: What calibration data is appropriate? The model trains on TinyStories/WebText. Calibration should use held-out samples from the same distribution.

4. **ANE vs Metal for 117M on iPhone**: Which is faster? ANE has W8A8 hardware on A17 Pro/M4 but needs layout changes. The Metal GPU path is simpler to target first.

5. **Prefill vs Decode TPS**: What fraction of user-perceived latency is prefill (time-to-first-token) vs decode? For a 117M model with typical prompts of 100-500 tokens, prefill may be nearly instant — making decode speed the sole focus.

6. **context extension**: What context length can the model handle with RoPE scaling before quality degrades significantly? SmolLM2 extended from 2K to 8K (4x). With 10× (1K → 10K), quality likely degrades. Starting at 1K and extending to 4K seems safe.
