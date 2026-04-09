# arcee-ai/LLama-405B-Logits

**URL:** https://huggingface.co/datasets/arcee-ai/LLama-405B-Logits  
**License:** Apache-2.0

## Teacher model
Llama 3.1 405B (Meta, instruction-tuned)

## Distribution data included
Top-K sparse logits stored per token. Columns:
- `token_ids` — indices of top-k tokens
- `top_values` — corresponding logit scores
- `normalized_logits` — softmax-normalized probabilities (not full 128K vocab)
- `input_ids` — tokenized input
- `attention_mask` — standard mask

**Not** full vocabulary; sparse top-k format to control storage.

## Dataset size
- 10,000 training examples (instruction-tuning scale, not pretraining)
- 11.5 GB on disk (Parquet)
- Sequence lengths: ~624–6,400 tokens

## Key notes
- Built for Arcee AI's DistillKit (open-source) and used to train their SuperNova 70B and the INTELLECT-1 model
- Arcee's later Virtuoso models used 1.1B–5B+ tokens of DeepSeek-V3 logits internally, but those logit datasets have NOT been released publicly
- Prior to this dataset, Arcee's pipeline was capped at ~100M tokens from Llama-405B logits
- Format is compatible with DistillKit's offline distillation pipeline
- The DistillKit technical paper describes advanced compression: polynomial approximation of the logit distribution curve, error-diffusion quantization of residuals, and bit-level packing (1–64 bits) — but it's unclear whether the released dataset uses this compression or the simpler top-k format
