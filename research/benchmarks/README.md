# Small Edge LM Benchmark Comparison

> April 2026 snapshot. Models in the 1B–4B range targeting Apple Silicon / mobile.

## Accuracy Benchmarks

Scores are accuracy (%) unless noted. Base models unless column says "IT" (instruction-tuned).
Shot counts follow each model's official eval protocol.

| Model | Params | MMLU | ARC-C | HellaSwag | WinoGrande | TruthfulQA | GSM8K | HumanEval | BoolQ |
|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B IT | 1.23B | 49.3 | 59.4 | 41.2 | — | — | 44.4 | — | — |
| Llama 3.2 3B IT | 3.21B | 63.4 | 78.6 | 69.8 | — | — | 77.7 | — | — |
| Phi-3-mini-4k IT | 3.8B | 70.9 | 86.3 | 75.3 | 71.6 | 64.7 | 85.7 | 57.3 | 78.1 |
| Phi-3.5-mini IT | 3.8B | 69.0 | 84.6 | 69.4 | 68.5 | 64.0 | 86.2 | 62.8 | 78.0 |
| Phi-4-mini IT | 3.8B | 67.3 | 83.7 | 69.1 | 67.0 | 66.4 | 88.6 | — | 81.2 |
| Gemma 3 1B (base) | 1.0B | — | 38.4 | 62.3 | 58.2 | — | 38.4 | 36.0 | 63.2 |
| Gemma 3 1B IT | 1.0B | 38.8 | — | — | — | — | 62.8 | 41.5 | — |
| Qwen2.5 0.5B | 0.5B | 47.5 | 35.6 | 52.1 | 56.3 | 40.2 | 41.6 | 30.5 | — |
| Qwen2.5 1.5B | 1.5B | 60.9 | 54.7 | 67.9 | 65.0 | 46.6 | 68.5 | 37.2 | — |
| Qwen2.5 3B | 3.0B | 65.6 | 56.5 | 74.6 | 71.1 | 48.9 | 79.1 | 42.1 | — |
| SmolLM2 1.7B | 1.7B | ~19.4† | 60.5 | 68.7 | 59.4 | — | 31.0 | — | — |
| OpenELM 3B | 3.0B | 24.8 | 47.7 | 76.9 | 68.0 | 38.8 | — | — | — |
| MobileLLM 1B | 1.01B | — | 39.0 | 61.4 | 62.3 | — | — | — | 66.7 |

† SmolLM2 reports MMLU-Pro (MCF), which is harder than MMLU — not directly comparable.

## Architecture Highlights

| Model | Params | Context | GQA | Sliding Window | Notes |
|---|---|---|---|---|---|
| Llama 3.2 1B | 1.23B | 128K | Yes | No | Shared embeddings; pruned+distilled from Llama 3.1 8B |
| Llama 3.2 3B | 3.21B | 128K | Yes | No | Shared embeddings; 9T training tokens |
| Phi-3-mini | 3.8B | 4K / 128K (LongRope) | No (MHA, 32 heads) | No | 4.9T tokens; synthetic + filtered web |
| Phi-3.5-mini | 3.8B | 128K | Not disclosed | No | Updated multilingual data mix; 3.4T tokens |
| Phi-4-mini | 3.8B | 128K | Yes | No | 200K vocab; 5T tokens |
| Gemma 3 1B | 1.0B | 32K | Yes | Yes (local 1024-tok, 5:1 local:global) | 2T tokens; no vision |
| Qwen2.5 0.5B | 0.5B | 32K | Yes (14Q/2KV) | No | Tied embeddings |
| Qwen2.5 1.5B | 1.5B | 32K | Yes (12Q/2KV) | No | Tied embeddings |
| Qwen2.5 3B | 3.0B | 32K | Yes (16Q/2KV) | No | Strong math/code |
| SmolLM2 1.7B | 1.7B | n/a | Not disclosed | No | 11T training tokens |
| OpenELM 3B | 3.0B | ~2K | Not disclosed | No | Layer-wise non-uniform scaling; 1.8T tokens |
| MobileLLM 1B | 1.01B | 2K | Yes (20Q/5KV) | No | 54 layers (deep+thin); 1T tokens |

## Inference on Apple Silicon

### Generation throughput (tok/s) — interactive use

| Model | Quant | Chip | RAM | Gen tok/s | Backend |
|---|---|---|---|---|---|
| Llama 3.2 1B | 4-bit | M4 Max | 128GB | ~100–120 | MLX |
| Llama 3.2 3B | 4-bit | M4 Max | 128GB | ~40–80 | MLX |
| Llama 3.2 3B | 4-bit | M1 Pro | 16GB | 25–35 | Ollama/GGUF |
| Phi-3-mini | Q4_K_M | modern Mac | — | 30–50 | llama.cpp/Ollama |
| Phi-4-mini | Q4_K_M | modern Mac | — | 30–50 | llama.cpp/Ollama |
| Gemma 3 4B | 8-bit | M4 Max | 64GB | 100.5 | Ollama/MLX |
| Qwen2.5 1.5B | Q4_K_M | M1 | 8GB | 35–45 | Ollama |
| Qwen2.5 7B | 4-bit | M1 Max | 64GB | 63.7 | MLX |

High tok/s numbers on M4 Max often reflect prefill; pure generation (the interactive-use bottleneck) is bandwidth-limited.

### Memory footprint (approximate)

| Model | Params | Q4 size | fp16 size |
|---|---|---|---|
| Llama 3.2 1B | 1.23B | ~0.7 GB | ~2.5 GB |
| Llama 3.2 3B | 3.21B | ~1.8 GB | ~6.4 GB |
| Phi-3/3.5/4-mini | 3.8B | ~2.2 GB | ~7.6 GB |
| Gemma 3 1B | 1.0B | ~0.6 GB | ~2.0 GB |
| Qwen2.5 0.5B | 0.5B | ~0.35 GB | ~1.0 GB |
| Qwen2.5 1.5B | 1.5B | ~0.9 GB | ~3.0 GB |
| Qwen2.5 3B | 3.0B | ~1.7 GB | ~6.0 GB |
| SmolLM2 1.7B | 1.7B | ~1.0 GB | ~3.4 GB |

Q4 sizes estimated at ~0.55 bytes/param (Q4_K_M); actual files vary ±10–15%.

## Key Takeaways

**Accuracy per parameter:**
- **Phi-4-mini (3.8B)** leads on math (GSM8K 88.6) and reasoning; Phi-3-mini leads on HumanEval (57.3) in the 3B class.
- **Qwen2.5 3B** is the strongest true-3B model: MMLU 65.6, GSM8K 79.1, HumanEval 42.1.
- **Gemma 3 1B IT** punches above weight on math (GSM8K 62.8) for a 1B model; sliding-window arch suits constrained inference.
- **SmolLM2 1.7B** is the best sub-2B model on commonsense (HellaSwag 68.7, ARC 60.5); 11T token training is the edge.
- **OpenELM 3B** scores poorly (MMLU 24.8) — only 1.8T tokens, older data mix. Research artifact, not a practical baseline.
- **MobileLLM 1B** is an architecture study (deep+thin) rather than a general-purpose model.

**Inference:**
- Q4 of every model here fits in 8 GB unified memory with headroom.
- M4 Max saturates at 80–120 tok/s for 1–3B models; M2/M3 16–32 GB delivers 25–60 tok/s at Q4.
- MLX is 20–90% faster than llama.cpp on M3+; all models have mlx-community Q4 conversions.

**Competitive target:** to matter in this space, a custom model needs to beat SmolLM2 1.7B or Qwen2.5 1.5B on at least one relevant benchmark while matching or improving inference speed on M-series. Phi-3-mini's MMLU/GSM8K at 3.8B is the ceiling to aim at if we scale up.
