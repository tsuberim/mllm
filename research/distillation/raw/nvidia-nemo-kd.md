# NVIDIA NeMo-Aligner: SFT with Knowledge Distillation

**URL:** https://docs.nvidia.com/nemo-framework/user-guide/24.12/modelalignment/knowledge-distillation.html  
**Blog:** https://developer.nvidia.com/blog/data-efficient-knowledge-distillation-for-supervised-fine-tuning-with-nvidia-nemo-aligner/  
**License:** Apache-2.0 (NeMo framework)

## Teacher models used
Nemotron-3-8B-Chat-4k-SFT (demo), Nemotron-4 340B SFT (experiments). Framework supports arbitrary models.

## Distribution data included
No pre-built dataset with logits provided. The framework includes scripts to generate and store teacher logits offline from any dataset.

**Format:** JSONL with chunking  
- Files named `train_with_logits_CHUNK_ID.jsonl`
- Each line contains the training example augmented with top-K teacher logits in **descending order**
- Top-K value: `k=4` in tutorial, `k=100` recommended in practice
- Must be in descending order — wrong order causes training divergence

**Base dataset used in docs:** OpenAssistant (oasst1) instruction pairs.

## Storage
No published size estimates, but loaded in chunks to manage memory. Chunk-based loading with `start_from_idx` / `end_at_idx` parameters.

## Key notes
- This is a production-quality implementation for offline logit distillation — scripts are open source in NeMo-Aligner repo
- The descending-order requirement is a gotcha: generate with any other script, must sort by logit value before saving
- On-policy distillation also available in NeMo-RL (newer, separate framework)
- No public pre-generated logit datasets from Nvidia; the tooling is there but you run it yourself
