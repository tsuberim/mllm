# Raw Sources

All sources consulted during research. Organized by topic.

## Metal Architecture

- https://dougallj.github.io/applegpu/docs.html — Apple G13 GPU reverse-engineered architecture (ISA, ALU design, register file layout)
- https://github.com/philipturner/metal-benchmarks — Philip Turner's metal-benchmarks repo (GPU microarchitecture constants, measured throughput, chip clock speeds)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf — Apple Metal Feature Set Tables (threadgroup sizes, feature availability by chip family)
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf — MSL Specification v4 (data types, built-ins, address spaces, instruction accuracy)
- https://developer.apple.com/videos/play/wwdc2022/10159/ — Scale Compute Workloads Across Apple GPUs (WWDC22): occupancy, SIMD-group best practices, threadgroup sizing
- https://developer.apple.com/videos/play/wwdc2016/606/ — Advanced Metal Shader Optimization (WWDC16): SIMD barriers, single-SIMD threadgroup recommendation
- https://developer.apple.com/videos/play/tech-talks/111375/ — Explore GPU Advancements in M3 and A17 Pro: Dynamic Caching, hardware ray tracing, mesh shading
- https://alyssarosenzweig.ca/blog/asahi-gpu-part-3.html — Asahi GPU Part III (Alyssa Rosenzweig): reverse-engineering M1 GPU pipeline
- https://chipsandcheese.com/p/a-brief-look-at-apples-m2-pro-igpu — Chips & Cheese M2 Pro iGPU: L2 cache sizes, measured bandwidth
- https://chipsandcheese.com/p/igpu-cache-setups-compared-including-m1 — Chips & Cheese M1 GPU cache hierarchy
- https://forum.beyond3d.com/threads/dynamic-register-allocation-in-gpus.63981/ — Dynamic register allocation discussion (context for Dynamic Caching)
- https://developer.apple.com/documentation/metal/mtlgpufamily — MTLGPUFamily Apple7/8/9 (Apple Developer Docs)

## Chip Families & Specs

- https://arxiv.org/html/2502.05317v1 — Apple vs Oranges: Evaluating Apple Silicon M-Series SoCs for HPC (arxiv 2502.05317); measured GPU throughput, roofline data
- https://maderix.substack.com/p/inside-the-m4-apple-neural-engine — Inside the M4 ANE Part 1: ANE architecture reverse-engineering
- https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615 — Inside the M4 ANE Part 2: benchmarks, 95 ms dispatch overhead, 32 MB SRAM
- https://arxiv.org/html/2603.06728v1 — Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference (arxiv 2603.06728)
- https://machinelearning.apple.com/research/neural-engine-transformers — Deploying Transformers on the Apple Neural Engine (Apple ML Research)
- https://machinelearning.apple.com/research/exploring-llms-mlx-m5 — Exploring LLMs with MLX and Neural Accelerators in the M5 GPU (Apple ML Research)
- https://arxiv.org/html/2601.19139 — Native LLM and MLLM Inference at Scale on Apple Silicon (arxiv 2601.19139)
- https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176 — Disaggregated Inference: NPU prefill + GPU decode (SqueezeBits)
- https://tzakharko.github.io/apple-neural-accelerators-benchmark/ — Apple Neural Accelerators Benchmark (A19/M5)
- https://www.michaelstinkerings.org/apple-m5-gpu-roofline-analysis/ — M5 GPU Roofline Analysis (Michael Stinkerings)
- https://www.notebookcheck.net/Apple-M4-8-core-GPU-Benchmarks-and-Specs.910076.0.html — M4 GPU specs (NotebookCheck)
- https://support.apple.com/en-us/121553 — MacBook Pro M4 Pro/Max Tech Specs (Apple Support)
- https://en.wikipedia.org/wiki/Apple_M1 — Apple M1 Wikipedia
- https://en.wikipedia.org/wiki/Apple_M2 — Apple M2 Wikipedia
- https://en.wikipedia.org/wiki/Apple_M3 — Apple M3 Wikipedia
- https://en.wikipedia.org/wiki/Apple_M4 — Apple M4 Wikipedia
- https://en.wikipedia.org/wiki/Apple_A16 — Apple A16 Wikipedia
- https://en.wikipedia.org/wiki/Apple_A17 — Apple A17 Wikipedia
- https://en.wikipedia.org/wiki/Apple_A18 — Apple A18 Wikipedia

## MLX Kernel Internals

- https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html — MLX Custom Metal Kernels Documentation
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.fast.metal_kernel.html — MLX fast.metal_kernel API reference
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/gemm.h — MLX Steel GEMM kernel source
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/sdpa_vector.h — MLX sdpa_vector decode attention kernel
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h — MLX Steel attention (prefill FlashAttention)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h — MLX quantized matmul kernels (qmv, qmm patterns, int4 dequant)
- https://github.com/ml-explore/mlx/pull/1833 — MLX use_optimal_threadgroups PR
- https://github.com/ml-explore/mlx/issues/2693 — MLX Metal 4 Tensor API + M5 Neural Accelerators issue
- https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-metal.metal — llama.cpp ggml-metal.metal (Q4_0 dequant, SwiGLU kernels)
- https://hazyresearch.stanford.edu/blog/2024-11-28-tk-mlx — ThunderMittens for Apple Metal (Hazy Research): simdgroup_matrix register layout

## CoreML & iPhone

- https://apple.github.io/coremltools/docs-guides/source/stateful-models.html — coremltools stateful models documentation (MLState API, KV cache example)
- https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html — coremltools PyTorch conversion guide
- https://apple.github.io/coremltools/docs-guides/source/opt-overview.html — coremltools optimization overview (palettization vs linear quantization)
- https://github.com/smpanaro/more-ane-transformers — more-ane-transformers (GPT-2 benchmarks on M1/M2 ANE, evals/SPEED.md)
- https://huggingface.co/blog/swift-coreml-llm — HuggingFace swift-coreml-llm (conversion gotchas, type mismatch issues)
- https://developer.apple.com/videos/play/wwdc2024/10159/ — WWDC24: Bring your models to Apple Silicon (SDPA fusion, int4, stateful KV cache, Mistral-7B demo)
- https://developer.apple.com/videos/play/wwdc2024/10161/ — WWDC24: Deploy ML/AI models on-device (MLComputePlan API, 1.6× speedup with state)
- https://machinelearning.apple.com/research/neural-engine-transformers — Apple ml-ane-transformers research article (Conv2d substitution, einsum patterns, ANE principles)
