# Sources Bibliography

Research for mllm edge inference optimization. April 2026.

---

## Papers

### Attention Mechanisms
- **GQA** — Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints." EMNLP 2023. https://arxiv.org/abs/2305.13245
- **MLA / DeepSeek-V2** — DeepSeek team (2024). "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model." https://arxiv.org/abs/2405.04434
- **PagedAttention** — Kwon et al. (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023. https://arxiv.org/abs/2309.06180
- **Mistral 7B / SWA** — Jiang et al. (2023). "Mistral 7B." https://arxiv.org/abs/2310.06825
- **Weighted GQA** — (2024). "Weighted Grouped Query Attention in Transformers." https://arxiv.org/abs/2407.10855
- **MLA Adaptation** — (2025). "Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs." https://arxiv.org/abs/2502.14837

### Quantization
- **AWQ** — Lin et al. (2023). "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024 Best Paper. https://arxiv.org/abs/2306.00978
- **GPTQ** — Frantar et al. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." https://arxiv.org/abs/2210.17323
- **SmoothQuant** — Xiao et al. (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." ICML 2023. https://arxiv.org/abs/2211.10438
- **LUT-GEMM** — (2022/2024). "LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models." ICLR 2024. https://arxiv.org/abs/2206.09557
- **FLUTE Fast LUT Matmul** — (2024). "Fast Matrix Multiplications for Lookup Table-Quantized LLMs." EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.724/
- **LUT Tensor Core** — (2025). "LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference." ISCA 2025. https://dl.acm.org/doi/10.1145/3695053.3731057
- **TurboQuant** — Google (2025). "TurboQuant." https://arxiv.org/abs/2504.19874

### Attention Kernels
- **FlashAttention-2** — Dao (2023). Fast and memory-efficient exact attention. https://github.com/dao-ailab/flash-attention
- **FlashAttention-4** — (2025). "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling." https://arxiv.org/html/2603.05451v1
- **FlashInfer** — (2025). "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving." https://arxiv.org/html/2501.01005v1

### Positional Embeddings
- **RoPE / RoFormer** — Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." https://arxiv.org/abs/2104.09864
- **RoPE Phase Modulation** — (2025). "Rotary Positional Embeddings as Phase Modulation: Theoretical Bounds on the RoPE Base for Long-Context Transformers." https://arxiv.org/html/2602.10959

### Speculative Decoding
- **EAGLE-1** — Li et al. (2024). "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." ICML 2024. https://arxiv.org/pdf/2401.15077
- **EAGLE-3** — (2025). "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test." https://arxiv.org/html/2503.01840v1
- **Kangaroo** — (2024). "Kangaroo: Lossless Self-Speculative Decoding for Accelerating LLMs via Double Early Exiting." NeurIPS 2024. https://openreview.net/forum?id=lT3oc04mDp
- **Draft & Verify** — (2024). "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding." ACL 2024. https://aclanthology.org/2024.acl-long.607/
- **Speculative Decoding Survey** — (2024). "A Comprehensive Survey of Speculative Decoding." ACL 2024 Findings. https://aclanthology.org/2024.findings-acl.456.pdf

### Small Model Architectures
- **SmolLM2** — Allal et al. (2025). "SmolLM2: When Smol Goes Big — Data-Centric Training of a Small Language Model." https://arxiv.org/html/2502.02737v1
- **Mixture of Attention Spans** — (2024). "Mixture of Attention Spans: Optimizing LLM Inference Efficiency with Heterogeneous Sliding-Window Lengths." https://arxiv.org/abs/2406.14909

### Optimizers
- **Muon** — Jordan (2024). Keller Jordan blog. https://kellerjordan.github.io/posts/muon/
- **Muon Scalable** — Liu & Su (2025). "Muon is Scalable for LLM Training." https://arxiv.org/pdf/2502.16982
- **Muon Convergence** — (2025). "Improved Convergence Rates of Muon Optimizer for Nonconvex Optimization." https://arxiv.org/html/2601.19400

### Apple Silicon / MLX
- **MLX on M5** — Apple (2025). "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU." https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- **MLX Benchmarks** — (2024). "Benchmarking On-Device Machine Learning on Apple Silicon with MLX." https://arxiv.org/html/2510.18921v1
- **Native LLM Apple Silicon** — Barrios et al. (2025). "Native LLM and MLLM Inference at Scale on Apple Silicon." https://arxiv.org/html/2601.19139v2
- **Comparative Study** — (2025). "Production-Grade Local LLM Inference on Apple Silicon: A Comparative Study of MLX, MLC-LLM, Ollama, llama.cpp, and PyTorch MPS." https://arxiv.org/pdf/2511.05502

### CoreML
- **CoreML Llama 3.1** — Apple (2024). "On Device Llama 3.1 with Core ML." https://machinelearning.apple.com/research/core-ml-on-device-llama
- **Mistral CoreML** — HuggingFace (2024). "WWDC 24: Running Mistral 7B with Core ML." https://huggingface.co/blog/mistral-coreml
- **Neural Engine Transformers** — Apple (2022). Apple machine learning research on ANE transformer optimization.

### Metal / GPU Kernels
- **Metal FlashAttention 2.0** — Draw Things (2024). https://medium.com/engineering-draw-things/metal-flashattention-2-0-pushing-forward-on-device-inference-training-on-apple-silicon-fe8aac1ab23c
- **ML Flash Attention** — Philip Turner (2024). https://github.com/philipturner/metal-flash-attention
- **Mamba** — Gu & Dao (2023). The Gradient explainer. https://thegradient.pub/mamba-explained/
- **SSM vs Transformer Tradeoffs** — (2025). https://goombalab.github.io/blog/2025/tradeoffs/

---

## Key GitHub Repositories

| Repo | Relevance |
|------|-----------|
| https://github.com/ml-explore/mlx | MLX framework — primary inference backend |
| https://github.com/ggml-org/llama.cpp | Metal kernel reference implementation |
| https://github.com/mit-han-lab/llm-awq | AWQ quantization reference |
| https://github.com/mit-han-lab/smoothquant | SmoothQuant reference |
| https://github.com/philipturner/metal-flash-attention | Metal FlashAttention |
| https://github.com/at2005/FlashMetal | FlashAttention for Metal (alternative) |
| https://github.com/SafeAILab/EAGLE | EAGLE speculative decoding |
| https://github.com/KellerJordan/Muon | Muon optimizer |
| https://github.com/apple/coremltools | CoreML export tools |
| https://github.com/huggingface/swift-transformers | CoreML LLM export examples |
| https://github.com/sharpner/turboquant-mlx | TurboQuant KV cache compression |
| https://github.com/flashinfer-ai/flashinfer | FlashInfer attention kernels |
| https://github.com/TristanBilot/mlx-benchmark | MLX operation benchmarks |

---

## Documentation

- MLX Custom Metal Kernels: https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html
- CoreML Quantization Overview: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html
- CoreML Quantization Algorithms: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html
- MLX quantized_matmul API: https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantized_matmul.html
- ExecuTorch CoreML Backend: https://docs.pytorch.org/executorch/0.7/backends-coreml.html
- PyTorch Muon: https://docs.pytorch.org/docs/stable/generated/torch.optim.Muon.html

---

## Blog Posts and Technical Write-ups

- EleutherAI RoPE blog: https://blog.eleuther.ai/rotary-embeddings/
- Keller Jordan on Muon: https://kellerjordan.github.io/posts/muon/
- Andreas Kunar on Apple Silicon LLM memory: https://medium.com/@andreask_75652/thoughts-on-apple-silicon-performance-for-local-llms-3ef0a50e08bd
- Sebastian Raschka MLA tutorial: https://sebastianraschka.com/llms-from-scratch/ch04/05_mla/
- FlashInfer blog: https://flashinfer.ai/2024/02/02/introduce-flashinfer.html
- Google speculative decoding retrospective: https://research.google/blog/looking-back-at-speculative-decoding/
- Nubank Muon for pretraining: https://building.nubank.com/muon-for-improved-foundation-model-pretraining-data-efficiency/
- SmolLM2 inside: https://huggingface.co/blog/Kseniase/insidesmol
- MLX vs llama.cpp comparison: https://groundy.com/articles/mlx-vs-llamacpp-on-apple-silicon-which-runtime-to-use-for-local-llm-inference/
- Gemma-3n edge model: https://smythos.com/developers/ai-models/gemma-3n-googles-edge-first-model-built-to-do-more-with-less/
