# Benchmark Selection

## Use

| Benchmark | Role |
|---|---|
| HellaSwag | Language understanding floor |
| ARC-Challenge | Reasoning ceiling at small scale |
| GSM8K | Multi-step reasoning, practical proxy |
| HumanEval + MBPP | Code utility (MBPP less memorization-prone) |
| IFEval | Instruction following — most practically relevant for a chat model |
| MT-Bench | Holistic assistant quality (LLM-judged multi-turn) |
| Tok/s @ Q4 on M-series | First-class project goal, not a footnote |
| Peak RAM @ Q4 | iPhone constraint: must stay under 4 GB |

Report MMLU for external comparability but don't optimize for it — small-model scores are suspiciously high (likely contamination) and it measures factual memorization more than reasoning.

## Skip

- **BoolQ** — saturated even at 1B, no signal
- **TruthfulQA** as optimization target — methodology contested, doesn't correlate with actual helpfulness
- **WinoGrande** — fine, but less differentiation at 1–3B scale

## Open Question

Primary use case (unresolved): general chat assistant vs. coding assistant. This shifts weight between IFEval/MT-Bench and HumanEval/MBPP.
