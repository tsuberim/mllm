# Knowledge Distillation Methods — Practical Guide for Merlin

**Scope**: Training a 1B-parameter student from a Qwen2.5-3B teacher. Offline logit generation on NVIDIA GPU; training on CUDA/PyTorch. Inference on MLX/Apple Silicon.

---

## 1. Which loss to use and why

### Short answer
Use **forward KL** for pretraining-scale offline distillation. Use **reverse KL or JSD** for fine-tuning / instruction tuning.

### Long answer

The forward KL vs reverse KL debate is more nuanced than the literature often presents. Key evidence:

**ICLR 2025 blogpost** ([source](https://iclr-blogposts.github.io/2025/blog/llm-knowledge-distil/)): At token level, the performance difference between forward and reverse KL is negligible (ROUGE1 0.5404 vs 0.5291 for 7B→1.5B distillation). Reverse KL may even converge slower in simple settings.

**Where reverse KL matters**: sequence-level reasoning tasks (GSM8K, MATH) where mode-averaging produces wrong answers. For pretraining on general text, forward KL's mode-covering behavior is *desirable* — the teacher distribution genuinely is multimodal and we want the student to cover it.

**Concrete recommendation by phase:**

| Phase | Loss | Rationale |
|---|---|---|
| Pretraining (offline logits) | Forward KL (α=0.9) + NLL (α=0.1) | Mode-covering is correct for general language modeling; forward KL is more stable with sparse stored logits |
| SFT / instruction tuning (online) | JSD (beta=0.5) or Reverse KL | Single-mode responses; prevent averaging over valid completions |
| Reasoning fine-tuning (optional) | Reverse KL or Entropy-Aware OPD | Mode-seeking critical when there's one correct answer |

**DistiLLM's skew-KL alternative**: Skew Reverse KL with α=0.1 prevents gradient instability when student/teacher distributions are far apart (early training). Formula:
```
D_SRKL^(0.1)(p, q) = D_KL(q, 0.9*p + 0.1*q)
```
This is a drop-in improvement over standard reverse KL. Worth using if doing online fine-tuning.

### What to avoid
- **MSE loss on logits**: the design-space paper found significant degradation vs KL ([arXiv:2410.16215](https://arxiv.org/abs/2410.16215))
- **Pure distillation (λ=1.0)**: omit NLL at your peril — even a 10% NLL term materially helps
- **Forward KL with uncorrected sparse logits**: top-K truncation introduces bias; see storage section below

---

## 2. Sequence-level vs token-level distillation

**Token-level** (standard for offline pretraining): compute KL divergence at every token position independently, using stored logits. This is what the design-space paper and DistillKit use. Simple, efficient, no sampling required.

**Sequence-level** (GKD, MiniLLM): student generates full sequences, teacher scores them. Addresses train/inference distribution mismatch. Better for fine-tuning tasks where the student generates text autoregressively.

**For pretraining from scratch**: token-level is the only practical option. Sequence-level requires running inference on the student model during training, which is 100–1000× slower than teacher scoring.

**For SFT after pretraining**: sequence-level via GKD is feasible and recommended if the teacher fits in memory (Qwen2.5-1.5B + 1B student = ~3.5 GB — fits easily).

---

## 3. Temperature scaling

Use **T=1.0 or T=2.0** for pretraining. The design-space paper found the performance difference between T=0.5 and T=2.0 is negligible (~same 2.5% improvement). Don't over-tune this.

**Always apply T² rescaling to the distillation loss:**
```python
loss_distill = (T**2) * F.kl_div(student_logprobs, teacher_probs, reduction='batchmean')
```

**When generating offline logits**: decide upfront whether to store raw logits (apply T at training time) or pre-softened logits (apply T at generation time). Storing raw logits is more flexible. Storing log-softmax (with T=1) is simpler and slightly more space-efficient.

Adaptive temperature methods (NormKD, WTTM) showed no significant additional benefit in pretraining experiments — not worth the implementation complexity.

---

## 4. Mixing ratio: distillation + NLL

**Use 90% distillation loss + 10% NLL** as the default:
```python
loss = 0.9 * loss_distill + 0.1 * loss_nll
```

Evidence: design-space paper found α=0.9 optimal; α=1.0 (pure distillation) slightly underperforms. The WSD scheduler for α (gradually increasing distillation contribution during training) gave the best results (+8.0% improvement), but the static 0.9/0.1 split is a good baseline.

The NLL term is not just regularization — it was found to be crucial for maintaining performance on hard benchmarks (MMLU, C-Eval) where pure KLD distillation falls short.

---

## 5. Teacher size and the capacity gap

**Recommendations: Qwen2.5-14B → 3B iPhone student; Qwen2.5-32B → 7B MacBook student.**

Evidence:
- The "Law of Capacity Gap" paper ([ACL 2025](https://aclanthology.org/2025.acl-long.1097.pdf)) finds optimal teacher scales ~linearly with student size, suggesting 3–4× the student size
- The Distillation Scaling Laws paper ([arXiv:2502.08606, ICML 2025](https://arxiv.org/abs/2502.08606)) confirms distillation beats supervised learning when the teacher already exists; at 3B–7B scale the gains are reliable and meaningful

**iPhone (3B student)**: Qwen2.5-14B — 4.7:1 ratio, squarely in the optimal 3–4× range. Qwen2.5-7B at 2.3× is too small; leaves performance on the table.

**MacBook (7B student)**: Qwen2.5-32B — 4.6:1 ratio, optimal. Consistent teacher family across both targets — same tokenizer, same training distribution.

**Cost (one-time, spot A100 ~$1.50/hr):**
- 14B teacher: ~$90–180/1B tokens
- 32B teacher: ~$200–400/1B tokens

---

## 6. Online vs offline — decision for this project

**Use offline distillation for pretraining.** Rationale:
- Qwen2.5-1.5B is already trained; we get its knowledge for free (Distillation Scaling Laws finding)
- Online logits from a fully-trained teacher are higher quality than on-the-fly logits from a teacher being trained concurrently (design-space paper: online early-checkpoint logits were −20.9% vs −offline +1.6%)
- No large GPU cluster required; logit generation can be done in a separate, cheaper batch job
- Storage cost is manageable: ~50 GB per 1B tokens (see below)

**Use online GKD for instruction fine-tuning phase** after pretraining. The teacher fits alongside the student, and sequence-level on-policy training is important for instruction-following quality.

---

## 7. Data generation pipeline

### Step 1: Corpus preparation
Use FineWeb-Edu or SmolLM-Corpus (already in scope). Tokenize with Qwen2.5's tiktoken tokenizer. Chunk into sequences of 2048 tokens (matching teacher context window).

### Step 2: Logit generation with vLLM

vLLM supports prompt log-prob extraction natively (feature completed January 2025, issue #8926 closed):

```python
from vllm import LLM, SamplingParams
llm = LLM(model="Qwen/Qwen2.5-1.5B", dtype="float16")
params = SamplingParams(max_tokens=0, prompt_logprobs=32)
outputs = llm.generate(text_chunks, params)
```

`prompt_logprobs=32` returns top-32 logprobs per token position. For importance sampling (per ACL 2025 sparse logit paper), use the custom RandomSamplingKD implementation at https://github.com/akhilkedia/RandomSamplingKD.

### Step 3: Storage format

```
# Parquet schema per shard (~1M examples):
{
  "input_ids":    List[int32],          # tokenized text
  "top_token_ids": List[List[int32]],   # shape: [seq_len, K]  
  "top_logprobs":  List[List[float16]]  # shape: [seq_len, K]
}
```

Use gzip compression: 3–5× reduction on integer arrays. Budget ~50 GB per 1B tokens with K=32.

### Step 4: Storage cost vs quality tradeoff

| K | Storage/1B tokens (compressed) | Bias | Notes |
|---|---|---|---|
| 16 | ~25 GB | Higher | Acceptable with importance sampling |
| 32 | ~50 GB | Low | Recommended baseline |
| 64 | ~100 GB | Very low | Diminishing returns |

With importance sampling (vs top-K), K=16 is nearly equivalent to top-K K=64 per the sparse logit sampling paper.

### Step 5: Cloud cost estimate
- 1B tokens of Qwen2.5-1.5B logits on 1× A100 80GB spot (~$1.50/hr on RunPod/Lambda Labs): ~6–12 hours → **~$10–20**
- 5B tokens: ~$50–100, ~30–60 hours
- Parallel on 4× A100: divide by 4

This is a one-time cost per teacher model.

---

## 8. Training recipe

### Loss implementation
```python
def distillation_loss(student_logits, stored_logprobs, stored_token_ids,
                      hard_labels, T=1.0, alpha=0.9):
    # Reconstruct sparse teacher distribution
    # student_logits: (B, L, V)
    # stored_logprobs: (B, L, K) — pre-computed teacher log-probs at temperature T
    # stored_token_ids: (B, L, K) — corresponding token ids
    
    # Forward KL: KL(teacher || student)
    student_logprobs = F.log_softmax(student_logits / T, dim=-1)
    
    # Gather student log-probs at teacher's top-K positions
    student_at_topk = student_logprobs.gather(-1, stored_token_ids)
    teacher_probs = stored_logprobs.exp()  # convert log-probs to probs
    
    # KL divergence (only over top-K positions; normalise teacher probs)
    teacher_probs = teacher_probs / teacher_probs.sum(-1, keepdim=True)
    kl_loss = (teacher_probs * (stored_logprobs - student_at_topk)).sum(-1).mean()
    kl_loss = kl_loss * (T ** 2)  # T² scaling
    
    # NLL on hard labels
    nll_loss = F.cross_entropy(student_logits.view(-1, V), hard_labels.view(-1))
    
    return alpha * kl_loss + (1 - alpha) * nll_loss
```

### Hyperparameters
- **alpha**: 0.9 (distillation) / 0.1 (NLL)
- **T**: 1.0 (start here; T=2.0 if training is unstable early on)
- **K**: 32 (sparse logits)
- **Batch size**: maximize for training stability; standard LM training sizes apply
- **LR schedule**: WSD (Warmup-Stable-Decay) — same as SmolLM2 training
- **Weight decay**: standard 0.1

### What to ablate
1. Teacher size: Qwen2.5-0.5B vs 1.5B — may reveal capacity gap effect
2. K: 16 vs 32 — check if quality drops with importance-sampled K=16
3. T: 1.0 vs 2.0 — low priority; design-space paper shows low sensitivity
4. alpha: 0.9 vs 0.8 — secondary concern

---

## 9. Apple Silicon / MLX inference considerations

### Training is on CUDA — this is fine
All distillation training runs on NVIDIA GPU with PyTorch. MLX is the inference runtime only. No distillation-specific changes are needed for MLX.

### Weight conversion
The standard PyTorch → MLX weight conversion path applies. Distillation does not add new weight types or architectures that would break conversion. The student model after distillation is a standard transformer.

### No MLX-native distillation needed
MLX does support fine-tuning (LoRA via mlx-lm), and in principle a distillation loop could be implemented in MLX. However:
- MLX training is significantly slower than CUDA for full pretraining
- Teacher model (Qwen2.5-1.5B) is large enough that running it on-device during training would be slow
- Offline distillation completely decouples teacher inference from student training — teacher never runs on Apple Silicon during the training phase

**Inference quality note**: distillation typically improves calibration (less over-confident outputs), which translates directly to better sampling quality at inference time on MLX. No special handling needed.

### iPhone target (future)
The 1B student, after distillation, will be quantized to int4/int8 for CoreML export. Distillation is known to improve robustness to post-training quantization — the softer distributions learned from the teacher help the student tolerate precision reduction better than models trained with hard labels alone. This is a useful secondary benefit.

---

## 10. Summary of findings by topic

| Topic | Verdict | Confidence |
|---|---|---|
| Best loss for pretraining | Forward KL + 10% NLL | High (design-space paper, ICLR 2025 blog) |
| Best loss for SFT | JSD or Reverse KL | High (GKD, MiniLLM, DistiLLM) |
| Sequence vs token level | Token-level for pretraining; sequence for SFT | High |
| Temperature | T=1.0–2.0, insensitive; always apply T² scaling | High |
| NLL mixing ratio | 90/10 (distillation/NLL) | High (design-space paper) |
| Best teacher size | Qwen2.5-3B (for 1B student) | High (capacity gap well-resolved at this ratio) |
| Online vs offline | Offline for pretraining, online for SFT | High |
| Logit generation tool | vLLM with `prompt_logprobs=32` | High |
| Storage format | K=32 fp16 logprobs + int32 token_ids, Parquet gzip | High |
| Apple Silicon impact | None — distillation is training-time only | High |

---

## Sources index

- [Pre-training Distillation Design Space (arXiv:2410.16215)](https://arxiv.org/abs/2410.16215) — mixing ratio, temperature, online vs offline
- [Distillation Scaling Laws (arXiv:2502.08606, ICML 2025)](https://arxiv.org/abs/2502.08606) — when distillation beats supervised training
- [Law of Capacity Gap (ACL 2025)](https://aclanthology.org/2025.acl-long.1097.pdf) — optimal teacher/student size ratio
- [MiniLLM (ICLR 2024, arXiv:2306.08543)](https://arxiv.org/abs/2306.08543) — reverse KL for language generation
- [GKD (ICLR 2024, arXiv:2306.13649)](https://arxiv.org/abs/2306.13649) — generalized JSD, on-policy fine-tuning
- [DistiLLM (ICML 2024, arXiv:2402.03898)](https://arxiv.org/html/2402.03898) — skew-KL, replay buffer, training efficiency
- [Sparse Logit Sampling / RandomSamplingKD (ACL 2025 Oral, arXiv:2503.16870)](https://arxiv.org/abs/2503.16870) — importance sampling beats top-K
- [ICLR 2025 blogpost: Forward KL vs Reverse KL](https://iclr-blogposts.github.io/2025/blog/llm-knowledge-distil/) — token-level differences minimal
- [On-policy distillation survey (arXiv:2604.00626)](https://arxiv.org/html/2604.00626) — comprehensive survey, entropy-aware OPD
- [vLLM issue #8926](https://github.com/vllm-project/vllm/issues/8926) — logit export status
