# Quantization on Apple Silicon: AWQ, GPTQ, SmoothQuant, MLX int4

## Sources
- AWQ paper: https://arxiv.org/abs/2306.00978 (MLSys 2024 Best Paper)
- GPTQ paper: https://arxiv.org/abs/2210.17323
- SmoothQuant: https://arxiv.org/abs/2211.10438 (ICML 2023)
- CoreML quantization: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html
- MLX quantization issue: https://github.com/ml-explore/mlx/issues/71

---

## GPTQ (Generative Pre-trained Transformer Quantization)

**Paper**: arxiv:2210.17323 (Oct 2022)

**Approach**: One-shot weight-only quantization using approximate second-order (Hessian) information. Quantizes weights column-by-column within each layer, immediately compensating remaining weights for the introduced error.

**Performance**:
- Quantizes 175B models in ~4 GPU hours
- 3-4 bit with minimal perplexity degradation
- Doubles compression gains vs previous one-shot methods
- Primarily for W4A16 (4-bit weights, 16-bit activations)

**Apple Silicon relevance**: GPTQ requires calibration data (~128 samples). CoreML Tools implements GPTQ via `LayerwiseCompressor`. MLX does not natively implement GPTQ but models can be GPTQ-quantized before MLX loading. Weight-only quantization (W4A16) is exactly what MLX uses — weights stored at int4, dequantized for compute.

---

## AWQ (Activation-aware Weight Quantization)

**Paper**: arxiv:2306.00978 (Jun 2023, MLSys 2024 Best Paper)
**Authors**: Ji Lin, Jiaming Tang, et al. (MIT)

**Key insight**: Not all weights are equally important. Only ~1% of weight channels correspond to salient activations. Protecting these 1% channels (by scaling them up before quantization) dramatically reduces quantization error.

**Mechanism**:
- No backpropagation required
- Uses activation statistics to identify salient channels
- Applies equivalent scaling transformation: scale W channel up → descale in activation side → net effect: salient weights get finer quantization grid
- All weights remain same bit-width (no mixed precision) → hardware friendly

**Performance**:
- Better than GPTQ on instruction-following and multi-modal tasks
- TinyChat framework: **>3x speedup** over HuggingFace FP16 on desktop and mobile GPUs
- Enables 70B Llama-2 on mobile GPUs
- Smaller accuracy drop than GPTQ at same bit-width

**Apple Silicon relevance**: AWQ's hardware-friendly approach (no mixed precision) maps well to Metal kernels. The scaling transformation can be absorbed into adjacent layers (scale folding) — no runtime overhead. For mllm at 117M, AWQ quantization could improve quality over simple round-to-nearest (RTN) int4 used by `mlx.nn.quantize`.

**Implementation**: MIT LLM-AWQ repo (github.com/mit-han-lab/llm-awq). Calibration requires ~100-200 tokens of representative data.

---

## SmoothQuant

**Paper**: arxiv:2211.10438 (Nov 2022, ICML 2023)

**Target**: W8A8 (both weights AND activations at int8). Unlike GPTQ/AWQ which are weight-only.

**Problem**: Activations have outliers that make int8 quantization lossy. Weights are easier to quantize.

**Solution**: Mathematically equivalent migration of quantization difficulty from activations to weights via per-channel scaling: X̃ = X / s, W̃ = W * s. Scale s chosen to balance smoothness.

**Results**:
- 1.56x speedup, 2x memory reduction for LLMs
- Enables W8A8 for OPT, BLOOM, Llama, etc.
- Negligible accuracy loss

**Apple Silicon relevance**: M4 and A17 Pro support int8 activation quantization on the Neural Engine. However, the Metal GPU path doesn't expose fused W8A8 int8 ops in the same way CUDA does. SmoothQuant is most useful if targeting ANE or if Apple adds W8A8 Metal support. For the GPU path, W4A16 (AWQ/GPTQ-style) is currently better supported.

---

## MLX Native Quantization (Current State)

MLX implements group-wise min-max quantization (W4A16) with:
- `mlx.nn.quantize(model, group_size=64, bits=4)` — what mllm currently uses
- This is simple round-to-nearest (RTN) within each group
- Group size 64 means 64 weights share one scale+bias pair
- Metal kernel: `mlx.core.quantized_matmul` / internal `qmm_t` / `qmm_n` kernels
- Weights stored as packed 4-bit integers, scales/biases in float16

**Implementation details** (from MLX Metal source):
- `mlx/backend/metal/kernels/fp_quantized.h` handles template-specialized dequantization
- Packed bit manipulation: bit-shifting and masking to unpack 4-bit values
- Scale + bias: `w_hat = scale * quantized_val + bias`
- Group boundary: every 64 weights gets new scale/bias

**Limitation of current RTN**: RTN does not minimize quantization error globally — just rounds each weight independently. AWQ-style calibration could reduce perplexity degradation, especially for small models where each parameter matters more.

---

## CoreML Quantization

**Source**: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html

Supported algorithms:
1. **RTN (Round-to-Nearest)**: Post-training, data-free. `linear_quantize_weights` API.
2. **GPTQ**: Layer-wise, requires calibration data (~128 samples). `LayerwiseCompressor` API.
3. **QAT**: Quantization-aware training. Highest quality but requires fine-tuning.
4. **Activation quantization**: Per-tensor or per-channel, requires calibration dataset.

**Hardware support**:
- Weights: int4 and int8 supported
- Activations: int8 supported
- Neural Engine (ANE): M3 and earlier → int8 only; M4/A17 Pro → W8A8 int8 hardware support
- GPU (Metal): W4A16 is primary path for LLM inference

**Block-wise int4** (from CoreML Mistral export):
```python
op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int4",
    granularity="per_block",
    block_size=32,
)
```
This matches the approach used for MLX — weights stored as int4, compute in float16.

---

## Comparison for mllm

| Method | Quality | Data Needed | Complexity | Speedup |
|--------|---------|-------------|------------|---------|
| RTN (current MLX) | Good | None | Trivial | Baseline |
| GPTQ | Better | ~128 samples | Medium | Same memory |
| AWQ | Best | ~128 samples | Medium | Same + folding |
| W8A8 (SmoothQuant) | Good | ~128 samples | High | 1.5x (if hardware supports) |

**Recommendation**: Implement AWQ calibration on top of the existing int4 MLX path. The scaling factors can be computed offline and folded into adjacent linear layers. At 117M scale, the perplexity improvement from AWQ over RTN may be meaningful (small models are more sensitive to quantization error per parameter). Expected improvement: ~0.5-2 perplexity points on language modeling benchmarks.
