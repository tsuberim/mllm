# Offline Logit Generation Pipeline

## Goal
Generate and store top-K (or importance-sampled) teacher logits at scale cheaply, for offline distillation training.

## Recommended tool: vLLM

vLLM is the fastest open-source LLM inference engine for NVIDIA GPUs. For logit generation:

**Status of logits support in vLLM:**
- Issue #8926 (requesting full logit export) was closed as completed in January 2025
- Issue #11397 contains implementation details
- vLLM supports log-prob extraction natively: `SamplingParams(logprobs=K)` returns top-K log probabilities per token position
- The `logits_processor` API gives access to the full (vocab_size,) logit tensor per step

**Practical approach:**
```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B", dtype="float16")
params = SamplingParams(max_tokens=0, prompt_logprobs=64)  # 64 top-K logprobs
outputs = llm.generate(prompts, params)
# outputs[i].prompt_logprobs: List[Dict[token_id -> logprob]]
```

Store to Parquet per the existing project plan: `(input_ids, token_ids, log_probs)` columns.

## Alternative: SGLang

SGLang (from LMSYS) achieves comparable or higher throughput than vLLM on many workloads and supports similar logprob extraction. Benchmarks show ~1150 tokens/sec on some workloads; vLLM achieves 3000+ tokens/sec for smaller models.

## Expected throughput

For Qwen2.5 models on a single NVIDIA A100 80GB (using vLLM):
- Qwen2.5-1.5B: ~10,000–15,000 tokens/sec (estimated; fp16, batch size ≥ 64)
- Qwen2.5-3B: ~5,000–8,000 tokens/sec (estimated)

These are rough estimates from vLLM throughput benchmarks on comparable-size models (DeepSeek-R1-Distill-Qwen-7B achieves ~3,362 tokens/sec on A100). Smaller models scale up roughly inversely with parameter count at constant batch size.

**Time to generate 5B tokens on one A100:**
- At 10,000 tok/s with Qwen2.5-1.5B: ~139 hours — too slow for a single run
- At 10,000 tok/s with batch size maximization: likely 3–5× faster → ~30–50 hours
- Renting 4× A100 on a cloud spot instance: ~8–12 hours for 5B tokens
- More practical: generate 1B tokens (~3 hours on 4× A100), ~$10–30 on spot instances

## Storage format

Per the sparse logit sampling paper ([arXiv:2503.16870](https://arxiv.org/abs/2503.16870)), importance sampling beats top-K for the same K:

```
# Generate using importance sampling (random sampling KD)
# Instead of always taking top-64 by probability,
# sample 64 tokens proportional to p_i (without replacement)
```

vLLM's `prompt_logprobs` returns top-K, not importance-sampled. To do importance sampling:
1. Use vLLM's `logits_processor` to intercept the full logit tensor
2. Sample K tokens proportionally, store (token_id, log_prob) pairs

This requires a custom logits processor. See: https://github.com/akhilkedia/RandomSamplingKD

## Storage cost

Per existing research notes (design-space paper):
- Top-64 fp16 sparse: ~2 bytes × 2 (token_id + logprob) × 64 × tokens = ~256 bytes/token
- At 1B tokens: ~256 GB — **too large**
- Practical storage: store token_id as int16 (up to 65536 vocab; Qwen uses 151K vocab → need int32) and logprob as fp16
  - int32 (4 bytes) + fp16 (2 bytes) = 6 bytes per pair × 64 pairs × 1B tokens = 384 GB — still large
  - With K=16: ~96 GB per 1B tokens — manageable with compression (Parquet gzip: 3–5× compression on repetitive integer arrays → ~20–30 GB per 1B tokens)
  - With K=32: ~192 GB uncompressed, ~40–60 GB compressed

**Practical recommendation**: K=32, fp16 logprobs, int32 token_ids, gzip Parquet. Budget ~50 GB per 1B tokens. For 1B-token training run: one SSD is sufficient.

## Cloud cost estimate

Generating 1B tokens of Qwen2.5-1.5B logits:
- 1× A100 80GB spot: ~$1–2/hr on Lambda Labs or RunPod
- At ~10K tok/s: ~28 hours → ~$30–60
- At optimized batching (30K tok/s possible with vLLM on small model): ~9 hours → ~$10–20

This is a one-time cost. The generated dataset can be reused for multiple training runs.

## Pipeline design

```
1. Download corpus (FineWeb-Edu or SmolLM-Corpus)
2. Tokenize with teacher tokenizer (Qwen2.5 tiktoken)
3. Run vLLM on corpus in batches, save prompt_logprobs to Parquet shards
   - Shard size: ~1M examples per file
   - Format: {input_ids: List[int], token_logprobs: List[List[(int, float)]]}
4. (Optional) apply importance sampling post-processing
5. Upload to object storage (GCS/S3) for use during training
```

**Critical note**: Store the teacher's tokenization of the corpus alongside the logits. If you later change the student's tokenizer, you'll need GOLD-trainer-style cross-tokenizer alignment — non-trivial.

## Sources
- [vLLM GitHub issue #8926](https://github.com/vllm-project/vllm/issues/8926) — logit export status
- [vLLM logits processors docs](https://docs.vllm.ai/en/latest/design/logits_processors/)
- [Sparse Logit Sampling / RandomSamplingKD (arXiv:2503.16870)](https://arxiv.org/abs/2503.16870)
- [verl async on-policy KD recipe](https://verl.readthedocs.io/en/latest/advance/async-on-policy-distill.html)
- [Qwen2.5 speed benchmark](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
