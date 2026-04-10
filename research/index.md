# Research Index

## Eval Pipeline

- [eval.py](../eval.py) — staged accuracy + perf pipeline with early exit: coherence → HumanEval → MBPP → perf → agent harness (opt-in, post-SFT)

## Benchmarks

- [benchmarks/README.md](benchmarks/README.md) — accuracy, architecture, and Apple Silicon inference comparison across 13 models in the 0.5B–4B range
- [benchmarks/selection.md](benchmarks/selection.md) — which benchmarks to use and why, what to skip, and the unresolved use-case question
- [benchmarks/market.md](benchmarks/market.md) — HuggingFace downloads, Ollama pull counts, and use cases per model as proxies for adoption

## Edge LLMs

- [edge-llms/insights.md](edge-llms/insights.md) — synthesized findings on inference optimization for a 117M-param GPT on MLX/CoreML, including current 625 TPS baseline
- [edge-llms/conclusions.md](edge-llms/conclusions.md) — key takeaways and actionable next steps scoped to this project
- [edge-llms/sources.md](edge-llms/sources.md) — bibliography of all sources consulted for edge inference research

## Distillation

- [distillation/README.md](distillation/README.md) — survey of logit/distribution datasets for KD, formats, storage costs, gaps, and recommendations for a 117M model
- [distillation/methods.md](distillation/methods.md) — practical training guide: loss selection, temperature, mixing ratios, teacher size, offline pipeline, Apple Silicon notes
- [distillation/raw/](distillation/raw/) — raw notes per source (arcee dataset, MiniLLM, GKD, MiniPLM, design-space paper, sparse logit sampling, NeMo, DistillKit, GOLD trainer, loss functions, scaling laws, pipeline)

## Feasibility

- [feasibility/harness.py](feasibility/harness.py) — agentic eval harness: runs tasks against a 3B model under 9.5K context, measures correctness, TPS, memory
- [feasibility/tasks.py](feasibility/tasks.py) — task suite: search, analysis, transform, scripting, multi-step, tree, semantic, iterative, pipeline, parallel batch, dataset cleaning (53 tasks total)

## Training Ideas

- **Grad-norm-based source weighting** — sample training sources proportional to per-source EMA of gradient norm (already computed for grad clipping, zero extra cost). Start weights=1 (uniform), converge toward the importance-sampling distribution over time. Theoretically minimises gradient variance (Katharopoulos & Fleuret, ICML 2018). Hypothesis: reduces Python over-representation effect without manual tuning. Test: compare val loss curve and convergence rate vs. fixed source weights on experiment config.

## Metal

- [metal/README.md](metal/README.md) — index and overview of the Metal/MLX deep-dive research series
- [metal/01_metal_execution_model.md](metal/01_metal_execution_model.md) — Metal GPU object hierarchy, command encoding, threadgroup/SIMD execution model
- [metal/02_chip_families.md](metal/02_chip_families.md) — GPU microarchitecture specs across M1–M4 generations relevant to kernel tuning
- [metal/03_mlx_kernel_optimization.md](metal/03_mlx_kernel_optimization.md) — MLX-specific kernel optimization techniques for transformer inference on Apple Silicon
- [metal/04_coreml_iphone.md](metal/04_coreml_iphone.md) — PyTorch → CoreML export path and on-device iPhone deployment constraints
- [metal/05_conclusions.md](metal/05_conclusions.md) — current 625 TPS baseline, theoretical ceiling (~2051 TPS), and gap analysis
