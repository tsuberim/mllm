# Pre-training Distillation for Large Language Models: A Design Space Exploration

**URL:** https://arxiv.org/html/2410.16215v1  
**Published:** 2024 (arxiv preprint)  
**License:** N/A (research paper)

## Teacher models used
- GLM-4-9B (primary experiments, 100B token run)
- 32B model (scaling experiments)

## Distribution data included
No public logit dataset released. The paper documents the process of generating and storing logits offline, with detailed analysis of storage costs:

- Full float32 logits for 100B tokens → **~58.6 PB** of storage
- After top-p (p=0.95) + top-k (k=100) truncation → **~15 TB** (~4,000× reduction)
- This gives a reference: **full vocab logits ≈ 586 GB per billion tokens** (fp32); top-100 sparse ≈ ~150 MB/billion tokens

## Format used
Two-stage truncation: first top-p=0.95, then top-k=100. Stored as sparse (index, value) pairs.

## Key findings
- Larger student LLMs benefit more from pre-training distillation
- Larger teachers don't always beat smaller teachers due to capacity gaps
- KL divergence and NLL perform similarly; MSE causes significant degradation
- Offline logits from fully pre-trained teachers outperform online logits from partially-trained teachers
- Optimal loss mix: ~90% distillation loss + 10% language modeling loss
- Temperature 0.5–2.0 gives similar results (low sensitivity)

## Key notes
- The 15 TB number for 100B tokens at k=100 is a critical practical data point
- No dataset released; this is primarily a methodology paper
