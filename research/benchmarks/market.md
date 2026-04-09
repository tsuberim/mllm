# Market Share & Use Cases

> Proxies only — no actual sales data exists for open-weight models.
> Data collected April 2026.

## Popularity Proxies

| Model | HF downloads/mo (instruct) | HF likes | Ollama pulls (all-time) |
|---|---|---|---|
| Qwen2.5 1.5B | 9.9M | 2,065 | 27M (family) |
| Qwen2.5 3B | 7.8M | 1,180 | ^ |
| Llama 3.2 3B | 5.9M | 4,300 | 64.4M (1B+3B combined) |
| Qwen2.5 0.5B | 5.7M | 839 | ^ |
| Llama 3.2 1B | 4.2M | 2,768 | ^ |
| Gemma 3 4B IT | 1.6M | 1,006 | 35.2M (family) |
| Gemma 3 1B IT | 1.1M | 675 | ^ |
| Phi-4-mini | 929K | 710 | 1.1M |
| Phi-3-mini (4k+128k) | ~930K | 3,106 | 17.1M |
| Phi-3.5-mini | 636K | 970 | 845K |
| SmolLM2 1.7B | 123K | ~400 | 3.2M |
| OpenELM 3B | ~3K | 339 | not on Ollama |
| MobileLLM 1B | 147 | ~10 | not on Ollama |

### Caveats

- HF downloads count automated pulls (CI, fine-tuning jobs) equally with human downloads; likes are a better engagement signal
- Ollama pulls are all-time cumulative, not 30-day — not directly comparable to HF numbers; but Ollama's audience (local inference on Mac/Linux/Windows) is the most relevant to this project
- OpenELM-1.1B-Instruct (1.5M downloads, 74 likes) is almost certainly an automated pipeline artifact — discard
- MobileLLM and OpenELM have no Ollama presence — research artifacts, not deployed models
- No developer survey (Stack Overflow, a16z) breaks down usage at this granularity

### Key Reads

- Qwen2.5 dominates HuggingFace: the 1.5B instruct alone outpaces Llama 3.2 1B 2:1
- Llama 3.2 dominates Ollama — the local inference audience
- Phi-3-mini accumulated 17.1M Ollama pulls despite being released earlier; strong mindshare in the coding/agentic niche

## Use Cases

| Model | Primary use case | Notes |
|---|---|---|
| Llama 3.2 1B/3B | General-purpose edge assistant | Pruned+distilled from 8B; Meta's explicit target is on-device chat and summarization |
| Phi-3/3.5/4-mini | Coding + STEM reasoning | Trained on synthetic "textbook" data; strong HumanEval/GSM8K; popular in agentic pipelines |
| Qwen2.5 0.5B–3B | Multilingual assistant + coding | Broadest language coverage; strong math; widely used as fine-tune base |
| Gemma 3 1B | On-device assistant (Android/mobile) | Google's explicit target; sliding-window arch reduces KV cache |
| SmolLM2 1.7B | Embedded / resource-constrained inference | HF's own model; targets browser, edge devices, CI eval harnesses |
| OpenELM | Architecture research | Layer-wise scaling study; Apple internal; not for production |
| MobileLLM 1B | Architecture research | Deep+thin study (depth > width for sub-1B); not a chat model |

## Competitive Position for This Project

Direct competitors (same goal — useful on-device assistant): **Llama 3.2** and **Gemma 3 1B**.
Strongest accuracy competitor: **Qwen2.5 1.5B**.
Ceiling to aim at on reasoning/coding: **Phi-4-mini** (3.8B).
