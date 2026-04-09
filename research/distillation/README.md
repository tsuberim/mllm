# Distillation Datasets Research

Research into publicly available datasets that include full teacher output distributions (logits, token probabilities, soft labels) for knowledge distillation with distribution-matching losses (KL divergence, JSD, etc.).

## What exists with full distributions

Almost nothing is publicly released at scale. The landscape is stark:

| Dataset | Teacher | Format | Size | Public? | License |
|---|---|---|---|---|---|
| [arcee-ai/LLama-405B-Logits](https://huggingface.co/datasets/arcee-ai/LLama-405B-Logits) | Llama 3.1 405B | Top-K sparse (Parquet) | 10k examples, 11.5 GB | Yes | Apache-2.0 |
| MiniPLM refined corpus | Qwen1.5-1.8B | Text only (binary shards) | 50B tokens | Yes | MIT |
| Arcee internal (Virtuoso) | DeepSeek-V3 | Top-K compressed | 1.1B–5B+ tokens | No | — |
| NeMo-Aligner examples | Nemotron-3/4 | JSONL top-K | Small (oasst1 scale) | No | — |

The Arcee 405B dataset is the only public logit dataset of note. Everything else is either methodology (no data released) or tools to generate your own.

"Distillation datasets" on HuggingFace are overwhelmingly text-output datasets — GPT-4 generated responses, chain-of-thought traces, etc. — not distribution-level data. The OpenThoughts-114k dataset even has a discussion thread asking whether "distillation" requires logit outputs (it doesn't, in their usage).

## Formats used

### 1. Top-K sparse (most common for offline distillation)
Store `(token_id, logit_or_logprob)` pairs for the top-K tokens at each position. Vocab not stored.
- Arcee: `token_ids` + `top_values` columns in Parquet
- NeMo: JSONL with top-K in descending order (order matters — training diverges otherwise)
- Typical K: 16–128 in practice; 100 is a common default

### 2. Full vocabulary (unusable at scale)
Float32 logits for every token in vocab (~128K for Llama 3):
- Storage: ~586 GB per billion tokens (float32)
- After top-p=0.95 + top-k=100 truncation: ~15 TB per 100B tokens (150 MB/B tokens)
- Practical only for tiny datasets or with heavy compression

### 3. Importance-sampled sparse (proposed, not yet common)
Random Sampling KD (ACL 2025, Sparse Logit Sampling paper): sample tokens proportional to probability rather than always taking top-K. Unbiased estimate vs biased top-K. Not yet packaged into a dataset format anyone uses.

### 4. Teacher-weighted text corpus (MiniPLM style)
Teacher log-probabilities used to filter/re-weight training examples; only the resulting text is released, not the probabilities themselves. Distribution signal is implicit in data selection, not explicit in stored values.

### 5. Online / on-policy (GKD, GOLD, DistiLLM)
Teacher is called live during training; nothing is pre-stored. Logits flow directly into the loss. This is the dominant approach in the academic literature. Requires teacher in memory during training.

## Practical storage tradeoffs

| Format | Storage per 1B tokens | Quality | Bias |
|---|---|---|---|
| Full vocab fp32 | ~586 GB | Lossless | None |
| Top-100 fp16 | ~3–5 GB | Near-lossless | Biased (top-K truncation) |
| Top-16 fp16 | ~0.5–1 GB | Lossy | More biased |
| Importance-sampled K=16 | ~0.5 GB | Competitive | Unbiased |
| Text-only (MiniPLM) | ~4 GB (tokenized text) | Indirect signal | N/A |

The ACL 2025 sparse logit sampling paper is the most important recent result here: top-K is biased and importance sampling matches quality with lower K. This matters for dataset design.

## Gaps — what doesn't exist

1. **A large-scale public logit dataset for pretraining** — nothing comparable to FineWeb or the Pile exists with teacher logits attached. Arcee's internal 5B-token DeepSeek-V3 logit dataset is the closest thing and it's not public.

2. **A logit dataset matched to a modern small-model vocabulary** — the Arcee dataset uses Llama 3.1's 128K tokenizer. If this project uses a different tokenizer, the logits don't transfer without cross-tokenizer alignment (what GOLD Trainer does).

3. **Importance-sampled logit datasets** — the theoretically superior format from the ACL 2025 paper has no corresponding public dataset.

4. **Pretraining-scale logit datasets at all** — the one public dataset (Arcee) is 10k instruction examples. Pretraining needs billions of tokens.

5. **Any logit dataset from open-weights reasoning models** — DeepSeek-R1, Qwen3, etc. These would be high-value teachers for a 1B model.

## Recommendations for this project

### Verdict on using existing data
The Arcee Llama-405B dataset is too small (10k examples, instruction-tuned) for pretraining from scratch. It could be useful for fine-tuning/instruction-tuning a pretrained base, but not for the initial pretraining phase.

### Recommended approach
Generate your own logit dataset offline using a suitable teacher.

**Candidate teachers (open weights, manageable on decent hardware):**
- Qwen2.5-14B → 3B iPhone student (4.7:1 ratio, optimal per capacity gap law)
- Qwen2.5-32B → 7B MacBook student (4.6:1 ratio, optimal per capacity gap law)
- Consistent teacher family: same tokenizer and training distribution across both targets

**Practical pipeline:**
1. Take a text corpus (FineWeb-Edu, SmolLM-Corpus, or similar)
2. Run teacher inference offline, store top-K logits per token (use K=32–64 with importance sampling per the ACL 2025 paper rather than naive top-K)
3. Store as Parquet or JSONL: `(input_ids, token_ids, log_probs)` columns
4. Train with KL divergence loss on stored logits + small NLL term (90/10 ratio per design-space paper)

**Storage estimate for a training run:**
- 5B tokens × 64 (token_id + logprob pairs) × 4 bytes × 2 ≈ ~2.5 GB — very manageable
- At 1B tokens: ~500 MB. Completely feasible locally.

**What to use for the loss:**
- Avoid forward KL (mode-averaging); prefer reverse KL or JSD(β≈0.5)
- Add 10% NLL/cross-entropy term to prevent distribution collapse
- Do NOT use top-K cached logits with standard KL without the bias correction from importance sampling

**Teacher capacity gap warning:**
The design-space paper found that very large teachers don't always beat small ones when there's a big capacity gap. For a 1B student, a 1B–3B teacher may outperform a 7B+ teacher in practice. Worth ablating.

### On-policy vs offline
On-policy (GKD-style) gives better results but requires keeping the teacher in memory during training. For a 1B student training on an NVIDIA GPU, a 1B teacher fits easily. For training on CPU/limited RAM, offline is the only viable path.

## Methods and training recipe

- [methods.md](methods.md) — practical guide: which loss to use, training recipe, data generation pipeline, hyperparameter choices, teacher size selection

## Sources consulted
- `raw/arcee-llama-405b-logits.md`
- `raw/minillm.md`
- `raw/gkd.md`
- `raw/miniplm.md`
- `raw/pretraining-distillation-design-space.md`
- `raw/sparse-logit-sampling.md`
- `raw/nvidia-nemo-kd.md`
- `raw/distillkit.md`
- `raw/trl-gold-trainer.md`
- `raw/loss-functions-kl-jsd-skew.md`
- `raw/online-vs-offline-distillation.md`
- `raw/temperature-scaling-nll-mixing.md`
- `raw/distillation-scaling-laws.md`
- `raw/logit-generation-pipeline.md`
