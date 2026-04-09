# Arcee AI DistillKit

**GitHub:** https://github.com/arcee-ai/DistillKit  
**Technical paper:** https://www.arcee.ai/blog/distillkit-v0-1-by-arcee-ai  
**License:** Apache-2.0

## Overview
Open-source toolkit for offline logit-based LLM distillation. Evolved from simple scripts (Aug 2024) into a full framework. Used internally by Arcee to produce SuperNova, Medius, and Virtuoso model series.

## Teacher models used (Arcee internal runs)
- Llama 3.1 405B → SuperNova 70B (dataset released: arcee-ai/LLama-405B-Logits, 10k examples)
- DeepSeek-V3 670B (8-bit) → Virtuoso-Lite 10B (~1.1B tokens logits, NOT released)
- DeepSeek-V3 → Virtuoso-Medium-v2 32B (5B+ tokens logits, NOT released)

## Logit storage format
Two approaches:
1. **Simple top-K**: store (token_id, logit_value) pairs for top-k tokens per position
2. **Advanced compression** (DistillKit v0.1+):
   - Polynomial approximation of the logit distribution curve
   - Error-diffusion quantization of residuals to preserve quality
   - Bit-level packing with arbitrary bit widths (1–64 bits)
   - Config parameters: `d: 128256` (vocab size), `k: 128` (top-k), `exact_k: 16`

## Training datasets used with DistillKit
- OpenHermes-2.5 (200k examples)
- WebInstruct-Sub (250k examples)
- FineTome (100k examples)

## Key notes
- The only publicly released logit dataset from Arcee is the 10k-example Llama-405B one; billion-token logit datasets are held internally
- The compression scheme (polynomial + quantization) is a significant engineering contribution not matched elsewhere
- DistillKit supports both logit-based and hidden-states-based distillation
- INTELLECT-1 (the decentralized training project) also used the 405B logit dataset
