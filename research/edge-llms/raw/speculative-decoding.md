# Speculative Decoding: Draft Models, Self-Speculative, EAGLE

## Sources
- EAGLE paper: https://arxiv.org/pdf/2401.15077 (ICML 2024)
- EAGLE-3: https://arxiv.org/html/2503.01840v1
- Kangaroo (self-speculative): https://openreview.net/forum?id=lT3oc04mDp
- Draft & Verify: https://aclanthology.org/2024.acl-long.607/
- Speculative decoding survey: https://aclanthology.org/2024.findings-acl.456.pdf
- Google retrospective: https://research.google/blog/looking-back-at-speculative-decoding/

---

## Speculative Decoding: Core Idea

Instead of generating tokens one at a time (each requiring a full forward pass), a cheap "draft" model proposes K tokens speculatively, then the target model verifies all K in a single parallel pass. If the draft token matches what the target would produce, it's accepted for free. If it doesn't match, generation restarts from the divergence point.

**Speedup condition**: Draft model must be much cheaper than target, and acceptance rate must be high enough that average tokens accepted per target pass > 1.

---

## Standard Draft Model Approach

Use a small separate model (e.g., 7B target + 70M draft). Draft proposes 4-8 tokens per step.

**Limitations for mllm at 117M**:
- The target model is already tiny (117M). Finding a meaningfully cheaper draft model is hard — would need ~10-15M parameter model.
- A 10x smaller model would be ~12M parameters. At that size, quality drops dramatically and acceptance rate falls.
- The speedup from draft+verify only exceeds 1x if acceptance rate * K > overhead. With a weak draft, acceptance rate is low.
- Memory: need to load both models. For iPhone with 4GB constraint, this is expensive.

**Verdict**: Traditional draft-model speculative decoding is unlikely to help significantly at 117M scale.

---

## Self-Speculative Decoding (No Separate Model)

### Draft & Verify (ACL 2024)
- Uses the target model itself with some layers skipped for drafting
- Early exit at a middle layer produces draft tokens
- Full model verifies
- No additional parameters, no extra memory footprint
- Achieved **1.99x speedup** on Llama-2 variants
- Plug-and-play — no additional training required

**Relevance**: At 117M / 12 layers, the model is already thin. Skipping 4-6 layers for drafting leaves a very weak predictor. Acceptance rate likely too low to be beneficial. The 1.99x speedup on Llama-2 (32 layers) does not transfer to 12-layer models.

### Kangaroo (NeurIPS 2024)
- Trains a lightweight adapter on top of an early-exit sub-network
- The adapter shares the target model's LM head (weight tying)
- Double early-exiting: exit early for draft, then full model for verification
- **2.04x speedup**, outperforms Medusa-1 with 88.7% fewer additional parameters
- Requires training the adapter (not purely plug-and-play)

**Relevance to mllm**: Kangaroo at 117M would require training a small sub-network adapter. With only 12 layers, the early-exit point would be around layer 4-6. This is worth experimenting with but not high priority.

---

## EAGLE (ICML 2024, EMNLP 2024, NeurIPS 2025)

**Repo**: github.com/SafeAILab/EAGLE

### How it works
EAGLE operates at the feature (hidden state) level, not token level. A lightweight autoregressive head predicts the next hidden state vector from the current hidden state + token embedding. The target model verifies by comparing against its own hidden state.

**Key innovation**: Drafting at feature level is more predictable than token level (features are smoother). Higher acceptance rate than token-level drafts.

### EAGLE variants
- **EAGLE-1** (ICML 2024): Feature-level draft head. Speedup 2.7-3.5x on LLaMA-2-Chat 70B
- **EAGLE-2** (EMNLP 2024): Dynamic draft trees — branches speculative paths, target verifies whole tree in one pass
- **EAGLE-3** (NeurIPS 2025): Tri-layer feature fusion (early + middle + late layers simultaneously). **3.0-6.5x speedup**, 20-40% better than EAGLE-2

### Requirements
- Needs a small trained draft head (~1-2 transformer layers on top of feature extraction)
- Training the draft head requires a fraction of original training compute
- At inference: parallel tree verification in target model

**Relevance to mllm**: EAGLE at 117M scale faces the same challenge — the draft head would be a non-trivial fraction of the total model. At 12 layers deep, the feature space may not be predictable enough. However, EAGLE-3's tri-layer fusion could work. The key question: is the draft head's compute < 1/N where N is the average accepted tokens?

**Rough estimate for 117M**:
- Full model: 12 transformer layers
- EAGLE draft head: ~2 transformer layers
- Draft head compute: ~17% of full model
- For speedup: need average acceptance > 1.2 tokens per full-model pass
- With a good draft head on quality training data, acceptance rate of 70-80% over 4 tokens → 2.8-3.2 tokens accepted → ~2.5x net speedup minus draft overhead ≈ **1.8-2x net speedup**

This is potentially worth implementing but requires training a draft head. Not a zero-effort win.

---

## Key Insight from Research

> "The latency of the draft model is a far stronger determinant of speculative decoding speedup than the draft model's language modeling accuracy."
> — from comprehensive 2024 speculative decoding survey

Implication: the draft model (or head) must be extremely cheap computationally, even if less accurate. For mllm, this suggests:
1. Draft head: 1 or 2 small MLP layers, not a full transformer layer
2. Direct token prediction (not feature-level) from compressed hidden state
3. Accept lower accuracy in exchange for <5% draft overhead

---

## Recommendation for mllm

Low priority now. The model is too small for traditional speculative decoding to give large speedups. The effort required (training a draft head, implementing tree verification) is significant. 

**If pursuing**: The best approach would be:
1. Train a tiny EAGLE-style draft head (2-layer MLP) on top of layer 8's hidden state
2. Target 4 draft tokens per step
3. Expect ~1.5-2x speedup if acceptance rate >60%

More impactful optimizations (GQA, RoPE, AWQ) should come first.
