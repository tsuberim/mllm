# Architecture Choices for Edge Efficiency: GQA, MQA, Linear Attention, Mamba, RWKV

## Sources
- SmolLM2 paper: https://arxiv.org/html/2502.02737v1
- Mamba overview: https://thegradient.pub/mamba-explained/
- Mamba vs transformers tradeoffs: https://goombalab.github.io/blog/2025/tradeoffs/
- SmolLM2 HuggingFace: https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct
- GQA paper: https://arxiv.org/abs/2305.13245
- Mistral paper: https://arxiv.org/abs/2310.06825

---

## Modern Small Model Architecture Choices

### SmolLM2-135M Configuration
```
vocab_size: 49,152
hidden_dim: 576
n_layers: 30
n_heads: 9 (query)
n_kv_heads: 3 (GQA — 3:1 ratio)
context_length: 2048 → 8000 (with RoPE scaling)
activation: SwiGLU
position: RoPE (θ=10,000 → 130,000 for extended context)
optimizer: AdamW with Warmup-Stable-Decay schedule
training: 2T tokens (135M)
```

Note: SmolLM2 uses 30 layers but narrower hidden dim (576) vs mllm's 12 layers at 768 hidden dim. Different efficiency tradeoff: more layers × narrower is sometimes better for sequential reasoning.

### Key architectural decisions in modern SLMs

| Model | Attention | Position | Activation | Context | Size |
|-------|-----------|----------|------------|---------|------|
| SmolLM2-135M | GQA (9q/3kv) | RoPE | SwiGLU | 2K→8K | 135M |
| Gemma-3 270M | MQA | RoPE | GeGLU | 8K | 270M |
| Phi-3 Mini | GQA | RoPE | SwiGLU | 4K | 3.8B |
| mllm (current) | MHA | Learned | SwiGLU | 1K | 117M |

---

## GQA vs MQA at Small Scale

At 117M scale with 12 heads, the tradeoff is:
- **MHA** (current): 12 Q heads, 12 KV heads → KV cache = 2 × n_layer × head_dim × seq_len × n_heads × bytes
- **GQA-4** (4 KV heads): 3x smaller KV cache, 3x less bandwidth loading KV during decode
- **MQA** (1 KV head): 12x smaller KV cache, but quality degradation at small scale

For 117M, 1024 context, fp16: KV cache = 2 × 12 × 64 × 1024 × 12 × 2 = 37.7MB
With GQA-4: 12.6MB. Marginal absolute saving but meaningful bandwidth reduction during decode.

**Recommendation**: GQA with n_kv_heads=4 (12/4 = 3x ratio). Higher priority than MQA because quality at 117M is precious.

---

## Mamba / Linear Attention Alternatives

### Mamba (State Space Model)
- Linear O(N) complexity vs O(N²) transformer
- Constant-size hidden state (no growing KV cache)
- 5x faster than transformers at equal parameter count on long sequences
- Fixed-size recurrent state: no memory growth with context length

**Performance comparison**:
- Mamba processes sequences in O(N) time, transformers O(N²)
- At short contexts (< 1K tokens), transformers are often comparable or faster due to parallelism
- At long contexts (> 8K), Mamba's advantage grows rapidly
- Mamba training is still somewhat specialized (selective scan is CUDA-heavy); Metal support is immature

**Quality limitations** (from 2025 tradeoffs blog):
- Mamba-style SSMs excel at compression/streaming tasks
- Transformers still win on tasks requiring precise retrieval from earlier context
- Hybrid models (Mamba + attention layers) achieve competitive quality with 60% fewer FLOPs on long sequences

### RWKV-style
- Similar linear time/memory to Mamba
- Recurrent formulation: constant inference memory
- Competitive quality on language modeling at small scales
- Less popular in production, fewer Apple Silicon optimizations

### Linear Attention
- Approximates softmax attention with kernel trick
- O(Nd²) instead of O(N²d) where d is head dim
- Quality gap vs standard attention at small scales is noticeable
- Not commonly used in practice for language generation (more for classification)

---

## Analysis for mllm

**Current context**: 117M parameters, 1024 token block size, targeting Mac + iPhone.

At 1024 tokens, the KV cache is small (~37MB full precision, ~18MB int8). The N² attention cost is:
- N=1024, D=768, H=12 heads: attention computation = 12 × 1024² × 64 = ~805M ops per layer = ~9.6B ops total for attention
- Linear layers: 12 layers × ~6 ops per token × 768² = ~51B ops
- **Attention is ~16% of compute at 1024 context** — not dominant yet

**At decode time (N=1 new token)**: attention reduces to 1 × 1024 × 64 per head — trivially cheap compared to linear layers. The bottleneck is loading weights for the linear layers.

**Conclusion on alternatives**: Mamba/linear attention would only benefit mllm significantly if:
1. Targeting much longer contexts (>4K tokens) where attention cost grows
2. Memory is extremely tight (iPhone with <2GB available for model)

At 117M / 1024 tokens, the transformer architecture is not the bottleneck. Stick with transformer but adopt GQA + RoPE for alignment with modern SLMs.

---

## Architectural Insights from Phi-3 / Gemma

### Phi-3 Mini (3.8B, Microsoft)
- Uses "high-quality data" philosophy: trains on filtered, curriculum-designed synthetic data
- Aggressive context extension: 4K→128K via LongRoPE
- GQA with 32 query heads / 32 KV heads for Mini (MHA effectively, scaled up)
- Key lesson: at <4B params, data quality dominates architecture choices

### Gemma-3 270M
- Uses MQA (single KV head) to be extremely memory efficient
- Fits in 529MB — usable on Raspberry Pi 5
- Shows that aggressive MQA is viable at very small scales where quality is already limited
- GeGLU activation (vs SwiGLU) — similar family, slightly different gate function

### Key lessons for mllm
1. Data quality and quantity matter more than architecture at 117M scale
2. GQA/MQA are standard — use GQA-4 (3:1 ratio, compromise between quality and memory)
3. RoPE + context extension is the standard path for longer contexts
4. Gemma-3's 270M with MQA is the closest comparable model — monitor its quality for reference

---

## Recommended Architecture Changes (Ranked by Impact)

1. **GQA (n_kv_heads=4)**: ~3x KV bandwidth reduction at decode, minimal quality loss
2. **RoPE**: ~1.8% quality improvement, context extendability, standard tooling
3. **Extended context (via RoPE scaling)**: 1K → 4K with continued pretraining
4. **No Mamba/linear attention yet**: insufficient benefit at current scale and context length
