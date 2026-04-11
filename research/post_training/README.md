# Post-Training Research

Research synthesized April 2026. Sources: Tulu 3, Qwen3, DeepSeek-R1, SAIL-RL, Agent-RLVR, OpenRLHF, veRL, TRL docs.

## Phase Order

**SFT (traces + instruction mix) → RL (GRPO, verifiable rewards) → Thinking fusion SFT**

Key insight: thinking fine-tune comes *after* RL, not before. SAIL-RL (2025) found that SFT-introduced thinking produces "pseudo aha moments" — shallow narrative that poisons the RL gradient. Qwen3's own recipe (Stage 3) puts thinking fusion *after* reasoning RL. Merlin follows this.

---

## Phase 1+2: SFT (Merged)

Instruction following is not a separate phase — it's a 70/30 mix baked into SFT:
- 70% validated agentic traces (bash/filesystem, Docker-verified)
- 30% code-flavored instruction-following data

**Why merged:** data close to the target distribution outperforms generic instruction data. Two separate SFT runs doubles cost for minimal gain. Run separately only if evals show regression.

### Loss

Response-only (prompts masked). For agentic traces, completion tokens carry the signal; prompt loss adds noise for short-completion tasks.

Ablation worth running: "Instruction Modelling" (NeurIPS 2024) — loss over both instruction and response. May help when system prompts are long relative to outputs.

### Packing

TRL `SFTTrainer` with `packing=True`, `packing_strategy="bfd"` (best-fit decreasing). Use **per-sample loss normalization** — normalize by completion tokens per sample, not total chunk tokens. Without this, short traces are underweighted. TRL v1.0 standardizes this but verify it's enabled.

### LoRA vs Full Fine-Tune

Full fine-tune at bf16: ~24GB total (weights + AdamW optimizer states). Fits on single H100.
- Full fine-tune: if trace corpus ≤100K examples
- LoRA (rank=64, α=128, q/k/v/o + MLP): if iterating fast or overfitting risk

### Datasets

| Source | Samples | License | Notes |
|---|---|---|---|
| Synthetic Qwen2.5-Coder-32B traces | ~20–50K | Qianwen License v1.1 (permits training) | Docker-validated only |
| glaiveai/glaive-function-calling-v2 | 113K | Apache 2.0 | |
| NousResearch/hermes-function-calling-v1 | — | Apache 2.0 | |
| allenai/tulu-3-sft-mixture (Persona-IF subset) | ~30K | ODC-BY-1.0 | Best instruction-following subset |
| FLAN v2 (ai2-adapt-dev, code/bash filtered) | ~20K | Apache 2.0 | Filter to technical tasks only |
| Evol CodeAlpaca | 107K | Apache 2.0 | |

**Do not use:** WizardLM, Orca, Open-Orca — GPT-4 generated, OpenAI ToS prohibits use in publicly released models.

### Hyperparameters

| Param | Value |
|---|---|
| LR | 1e-5 to 2e-5 (full), 3e-4 (LoRA) |
| Warmup | 3% of steps, cosine |
| Epochs | 2–3 |
| Batch (global) | 32–64 |
| Max seq len | 6144 |
| Weight decay | 0.01 |

### Catastrophic Forgetting

Mix in 5–10% general Python code (TheStack samples) to preserve pre-training code quality.

### Compute

~20–50K traces, avg 1500 tokens response: 30–75M response tokens.
Full fine-tune, 2–3 epochs: **2–6 H100-hours** (~$4–24).

### Eval

- Task pass@1 on held-out Docker sandbox: target ≥60%
- Special token format compliance (regex check)
- HumanEval / MBPP regression (must hold)

---

## Phase 3: RL on Verifiable Rewards

### Algorithm: GRPO (G=4–8)

- PPO: requires critic model, doubles memory, adds instability. Not worth it for binary rewards.
- REINFORCE++: good for dense rewards; fails for binary pass/fail (zero variance from single sample → zero gradient).
- **GRPO with G=4–8:** group relative advantage estimates. For binary rewards, comparing within-group rollouts is essential — a single pass/fail sample gives no gradient signal.

### Reward Signal

```
+1.0  task completes (output + filesystem state correct)
 0.0  task fails, error, timeout
+0.1  correct tool selected on failed task (optional shaping, reduces sparsity)
```

The reward function must strip `<think>` blocks before evaluating bash output — otherwise model learns to hide valid bash inside thinking.

### Sparse Reward Mitigations

1. **Curriculum:** single-step tasks first → multi-step only after >50% pass@1
2. **Process shaping:** reward each individual tool call (non-error result = small reward), not just final outcome
3. **Agent-RLVR guidance:** when all G rollouts fail, provide a hint and re-attempt; include guided trajectories in RL batch

### Framework

| Scale | Framework |
|---|---|
| Single H100 | TRL GRPOTrainer |
| Multi-GPU / async rollouts | veRL (Modal example: `modal.com/docs/examples/grpo_verl`) |

Reward function: `RewardManager` subclass with Modal Sandbox subprocess calls. Use pre-warmed containers to keep overhead <5 sec/task.

### Task Distribution

Target 500–2000 unique tasks. Diversity > quantity for RL generalization.

| Source | License |
|---|---|
| feasibility/tasks.py (53 tasks, extend) | Your data |
| InterCode-Bash | Apache 2.0 |
| SWE-bench Lite (code subset) | MIT |
| Custom Qwen-generated tasks | Permissive |

### Hyperparameters

| Param | Value |
|---|---|
| LR | 5e-7 to 1e-6 |
| KL penalty (β) | 0.01–0.04 |
| Group size (G) | 4–8 |
| Rollout max tokens | 2048 |
| Reward clip | ±5 |
| Entropy bonus | 0.001–0.01 |
| ε (PPO clip) | 0.2 |
| Gradient clip | 1.0 |
| Rollout temp | 0.9–1.0 |

### Compute

500–2000 steps, GRPO G=4, 3B model: **8–48 H100-hours** (~$16–192). Most expensive phase. Checkpoint every 50 steps.

### Eval

- Task pass@1 on 50 held-out tasks: target >70% (vs. 47% feasibility baseline)
- Pass@4 (best-of-4) to assess distribution quality
- KL from SFT policy: monitor per step, keep <0.1 nats
- Reward trajectory: must rise; if flat at 0 after 200 steps, curriculum is wrong

### Risks

- **Reward hacking:** multiple test cases per task, randomize expected outputs
- **Training instability:** gradient clipping + LR warmup (50 steps)
- **Rollout collapse (all identical):** entropy bonus + temp > 1.0
- **3B capacity ceiling:** RL can't discover task types absent from SFT traces

---

## Phase 4: Thinking Fusion SFT

Runs *after* RL. Re-introduces `<think>` tokens using traces from the RL-trained model.

### Mechanism

50/50 token mix of thinking vs non-thinking examples:
- Thinking: sampled from RL model at high temp, rejection-filtered for correctness + coherent reasoning
- Non-thinking: Phase 1 agentic traces (clean, no `<think>` tokens)

Chat template includes a think/no-think flag per turn (e.g., `<|no_think|>` in your protocol). Model learns *when not to think* by seeing the distribution of contexts where thinking is absent.

### Think Block Quality Filter

Reject traces where:
- Think block is pure narrative ("I will now run bash...")
- Think block references tools that don't exist
- Think block is circular / repetitive
- Thinking didn't change the final answer (model would have gotten it right anyway)

Only keep traces where thinking demonstrably led to a better decision.

### Hyperparameters

| Param | Value |
|---|---|
| LR | 5e-6 (conservative) |
| Think/no-think ratio | 50/50 by token count |
| Max think block | 1024 tokens (hard cap) |
| Epochs | 1–2 |

### Compute

30K thinking traces, avg 3000 tokens: **4–8 H100-hours** (~$8–24).

### Eval

- Thinking activation rate: target <30% of tasks trigger `<think>` (most are simple enough to skip)
- Pass@1 on hard tasks: expect 10–20% improvement over non-thinking baseline
- `<|no_think|>` compliance: hard requirement, 100%
- Context efficiency: mean tokens per task should not exceed 2× non-thinking baseline

### Abort Criterion

If thinking pass rate doesn't improve >10% over non-thinking baseline after 20K fusion SFT examples, skip this phase. 3B may not have enough capacity for useful self-generated reasoning.

---

## Compute Summary

| Phase | H100-hours | Est. Cost |
|---|---|---|
| SFT (traces + instruction) | 2–6 | $4–24 |
| RL (GRPO) | 8–48 | $16–192 |
| Thinking fusion SFT | 4–8 | $8–32 |
| **Total** | **14–62** | **$28–248** |

RL dominates. Use spot/preemptible H100s on Modal; checkpoint frequently.

---

## Open Questions

1. **Qwen trace license:** Qianwen License v1.1 permits training other models with outputs. Verify before public release.
2. **Tokenizer version:** all phases must use the same tokenizer. If retrained after trace gen, re-tokenize Phase 1 SFT data.
3. **Does thinking help at 3B?** DeepSeek-R1-Distill-Qwen-1.5B shows reasoning via distillation, but RL-derived thinking from scratch at 3B is less proven. Treat Phase 4 as an experiment with a clear abort criterion.

---

## Key Sources

- Tülu 3: <https://arxiv.org/abs/2411.15124>
- Qwen3 Technical Report: <https://arxiv.org/abs/2505.09388>
- DeepSeek-R1: <https://arxiv.org/abs/2501.12948>
- SAIL-RL: <https://arxiv.org/abs/2511.02280>
- Agent-RLVR: <https://arxiv.org/abs/2506.11425>
- REINFORCE++: <https://arxiv.org/abs/2501.03262>
- veRL: <https://github.com/verl-project/verl>
- TRL GRPO: <https://huggingface.co/docs/trl/main/en/grpo_trainer>
- Instruction Modelling (NeurIPS 2024): <https://arxiv.org/abs/2408.10642>
- SFT Memorizes, RL Generalizes: <https://github.com/LeslieTrue/SFTvsRL>
