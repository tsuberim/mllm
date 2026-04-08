# Emerging Small Model Architectures: Phi-3, Gemma, SmolLM, Gemma-3n

## Sources
- SmolLM2 paper: https://arxiv.org/html/2502.02737v1
- Phi-3.5 overview: https://medium.com/aimonks/phi-3-5-microsofts-efficient-multilingual-and-secure-open-source-slms-5ed7d36738aa
- Gemma-3n edge model: https://smythos.com/developers/ai-models/gemma-3n-googles-edge-first-model-built-to-do-more-with-less/
- SmolLM2 HuggingFace: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- LLM architecture gallery: https://sebastianraschka.com/llm-architecture-gallery/

---

## SmolLM2 (HuggingFace, 2024)

### 135M configuration
```
Parameters:    135M
Hidden dim:    576
Layers:        30
Q heads:       9
KV heads:      3  (GQA, 3:1 ratio)
Context:       2048 → 8192 (with RoPE extension)
Vocab size:    49,152
Activation:    SwiGLU
Position:      RoPE (θ=10,000 base → 130,000 extended)
Norm:          RMSNorm
Bias:          No (bias-free)
Training:      2T tokens
Optimizer:     AdamW, β=(0.9, 0.95), WSD schedule
Peak LR:       5×10⁻⁴
```

### Key architectural decisions
- **More layers, narrower**: 30 layers × 576 hidden vs alternatives like 12 × 768
  - Benefit: deeper reasoning paths, better gradient flow
  - Cost: more sequential dependency (harder to parallelize layers)
  - For inference: more layer-sequential memory accesses
- **3:1 GQA ratio**: 3x KV cache reduction, minimal quality loss at this scale
- **Bias-free**: simplifies architecture, marginal speedup
- **RoPE over learned positional**: enables context extension without retraining
- **Vocabulary 49,152**: BPE tokenizer optimized for multilingual data

### Training strategy
- Single-stage training for 135M (simpler than multi-stage used for larger variants)
- High-quality curated data: DCLM filtered with FineWeb-Edu classifier, Stack-Edu, InfiMM-WebMath, FineMath, Cosmopedia
- 2T tokens total — notable: same model quality at 2T tokens with better data vs much more data with lower quality

### What mllm can learn
1. The 30-layer × 576 vs 12-layer × 768 tradeoff: for equivalent parameters, more depth tends to be better for reasoning
2. 3:1 GQA is the proven configuration for <1B models
3. RoPE + WSD LR schedule is the reliable recipe

---

## Phi-3 (Microsoft, 2024)

### Phi-3-Mini (3.8B)
```
Parameters:    3.8B
Architecture:  Llama-style decoder
Attention:     GQA (32 Q heads, 32 KV heads = effectively MHA, no reduction)
Context:       4K → 128K (LongRoPE)
Training:      3.3T tokens
Data:          High-quality synthetic + filtered web
```

Note: Phi-3-Mini uses 32/32 heads (no GQA reduction). Microsoft prioritized quality over memory efficiency here, relying on high-quality data to compensate for model size.

### Phi-3.5-MoE
- Mixture of Experts: 16 experts, 2 active per token
- Activates only ~3.8B params per step (vs 42B total)
- Efficient inference: same compute as Mini but more capacity
- Not directly relevant for mllm (MoE adds routing complexity)

### Key lesson from Phi-3
Data quality > model size. Phi-3-Mini (3.8B) outperforms models trained on much more data at 7B. This is the most important lesson for mllm: invest in training data quality before expanding model size.

---

## Gemma-3 (Google, 2025)

### Gemma-3 270M
```
Parameters:    270M
Architecture:  Gemma-3 decoder
Attention:     MQA (single KV head) — aggressive for size
Context:       8K
Size on disk:  529MB
Hardware:      Runs on Raspberry Pi 5 with 4GB RAM
Activation:    GeGLU (similar to SwiGLU, uses GELU gate)
```

**MQA at 270M**: Gemma-3 makes the aggressive choice of MQA (single KV head) even at 270M. This gives maximum memory/bandwidth efficiency but with quality trade-off. For a 270M model, the quality bar is already low, so the trade-off is acceptable.

**GeGLU vs SwiGLU**: GELU(x) ≈ xΦ(x) vs SiLU(x) = x·sigmoid(x). Both are gated, smooth non-linearities. SwiGLU is slightly faster (sigmoid simpler than GELU approximation) but quality difference is minimal.

### Gemma-3n (Google, 2025)
Per-Layer Embeddings (PLE): different embedding table per layer rather than shared input/output embeddings. Allows layer-specific token representations. Increases model expressivity with small parameter cost.

---

## Architecture Efficiency Patterns (Across All Models)

### What every modern small efficient model has in common
1. **RoPE positional embeddings** (not learned absolute)
2. **SwiGLU or GeGLU activation** in MLP (not ReLU, not GELU)
3. **RMSNorm** (not LayerNorm with bias)
4. **No biases** in linear layers
5. **Pre-norm** (normalize before attention/MLP, not after)
6. **Weight tying** for embedding/unembedding (saves vocab_size × hidden_dim parameters — at 50K × 768 = 38M params, significant for 117M model)

### What varies
- **GQA ratio**: 3:1 (SmolLM2), 4:1 (Llama-3), 8:1 (Mistral), MQA (Gemma-3 small)
- **Layer count vs width tradeoff**: deeper+narrower vs shallower+wider (no clear winner; deeper tends to better reasoning)
- **Vocab size**: 32K (Llama 2), 49K (SmolLM2), 50K (GPT-2/mllm), 256K (Gemma) — larger vocab = better tokenization efficiency but larger embedding tables

### mllm's current alignment with best practices
- SwiGLU: ✓ (already implemented)
- RMSNorm: ✓ (already implemented)
- No biases: ✓ (already implemented)
- Pre-norm: ✓ (already implemented)
- Weight tying: ✓ (already implemented)
- RoPE: ✗ (uses learned embeddings — should migrate)
- GQA: ✗ (uses MHA — should migrate to GQA-4)

mllm is 4/6 on best practices. The two missing items (RoPE, GQA) are the highest-value changes.

---

## Parameter Count Analysis: mllm vs SmolLM2-135M

### mllm (117M, 12 layers × 768 hidden)
- Embedding: 50,257 × 768 = 38.6M (but weight-tied with output head)
- 12 × attention (MHA): 4 × 768² = 2.36M each → 28.3M total
- 12 × MLP (SwiGLU, 4× expansion): 3 × 768 × 3072 = 7.08M each → 85M total
- Norms etc: small
- Total ≈ 117M (weight-tied), 155M (unembedded)

### SmolLM2-135M (30 layers × 576 hidden)
- 30 × attention (GQA): smaller per-layer but 2.5x more layers
- Different depth-vs-width tradeoff

**Key question**: Is 12-layer × 768 or 30-layer × 576 better at ~135M? The SmolLM2 configuration was chosen after ablations — the deeper-narrower configuration is likely better for language modeling quality at this scale. Consider for mllm's next major version.

---

## Recommendation

**For current 117M model (short term)**: Add GQA and RoPE. Aligns with best practices, achieves concrete speedups.

**For next major architecture revision**: Consider 30-layer × 576-dim with GQA-4 and RoPE. This is exactly SmolLM2-135M's configuration and has been validated on 2T tokens. Re-training from scratch anyway if changing PE scheme.
