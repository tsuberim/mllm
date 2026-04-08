# MLX Framework Benchmarks and Internals

## Sources
- Apple M5 MLX paper: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Native LLM Apple Silicon paper: https://arxiv.org/html/2601.19139v2
- MLX benchmark (Tristan Bilot): https://github.com/TristanBilot/mlx-benchmark
- MLX transformers benchmark: https://github.com/aukejw/mlx_transformers_benchmark
- Comparative study: https://alphaxiv.org/overview/2511.05502
- SwiftLM: https://github.com/SharpAI/SwiftLM
- TurboQuant MLX: https://github.com/sharpner/turboquant-mlx

---

## MLX Architecture Overview

MLX is Apple's open-source array framework for machine learning on Apple Silicon (released Dec 2023). Key design principles:

- **Lazy evaluation**: operations are not executed until results are needed (or `mx.eval()` is called). Enables graph optimization.
- **Unified memory**: arrays live in shared CPU/GPU memory — no copies between devices
- **Composable function transforms**: `mx.grad()`, `mx.vmap()`, `mx.compile()`
- **Metal backend**: JIT-compiled Metal kernels for GPU execution
- **Python + C++ + Swift**: Python frontend, C++ core, Swift bindings

---

## Performance Numbers

### Token generation throughput (decode, various hardware)

**vllm-mlx (2025, from arxiv:2601.19139)**:
- Qwen3-0.6B: **525.5 t/s** (M-series, unspecified chip)
- Llama-3.2-1B: **461.9 t/s**
- Llama-3.2-3B: **203.6 t/s**
- Qwen3-4B: **159.0 t/s**
- Qwen3-8B: **93.3 t/s**
- Qwen3-30B-A3B (MoE): **109.7 t/s**

**Concurrency scaling (Qwen3-0.6B)**:
- 1 request: 441 t/s
- 16 concurrent requests: 1,642 t/s (3.7x throughput from batching)

**Comparison to other frameworks**:
- vllm-mlx: 1.21-1.87x faster than llama.cpp (across tested models)
- MLX leads by 20-87% for models <14B (compute-bound range)
- Gap closes to near-zero at 27B+ (memory bandwidth saturation)

### M5 vs M4 speedups (Apple MLX paper, 2025)

**Prefill speedup (M5 vs M4)**:
- Qwen 1.7B: **3.57x** faster TTFT
- Qwen 8B (BF16): **3.62x**
- Qwen 14B (4-bit): **4.06x**
- Qwen 30B MoE (4-bit): **3.52x**

**Generation speedup (M5 vs M4)**:
- All models: **1.19-1.27x** (limited by memory bandwidth, not compute)
- M5 has 28% more memory bandwidth (153 vs 120 GB/s) → explains ~1.25x factor

**Key insight**: Prefill is compute-bound (M5's Neural Accelerators give 3.5-4x speedup). Decode is memory-bound (only 1.2x speedup from 28% more bandwidth). This is the roofline model in action.

---

## MLX Metal Accelerator Integration (M5)

MLX leverages Metal 4's Tensor Operations and Metal Performance Primitives to access M5's dedicated Neural Accelerators for matrix-multiply operations.

```
Tensor Operations (Metal 4) → Neural Accelerator hardware → Matrix-multiply at peak TFLOPS
```

This is not available for earlier chips (M1-M4) in the same way — those use the standard Metal GPU shader cores with `simdgroup_matrix` operations.

---

## Quantization in MLX

### Native MLX quantization API
```python
import mlx.core as mx
import mlx.nn as nn

model = MyModel(config)
nn.quantize(model, group_size=64, bits=4)  # in-place int4 quantization
```

**Internal format**: 4-bit weights packed 2 per byte, stored as uint8. Per-group (64 weights) scale and bias in float16.

### Conversion utilities
```bash
mlx_lm.convert --hf-path meta-llama/Llama-3-8B --mlx-path ./llama3-mlx -q  # converts + quantizes in seconds
```

### Memory reduction
- fp16 model → int4 (group_size=64): ~4x reduction
- 117M fp16: ~234MB → int4: ~62MB (plus 234MB/64 × 2 × 2 = 14.6MB for scales = ~77MB total)

---

## TurboQuant MLX (2025)

**Source**: github.com/sharpner/turboquant-mlx (POC of Google's TurboQuant paper arxiv:2504.19874)

A KV cache quantization approach:
- PolarQuant: randomized Hadamard rotation + Lloyd-Max quantization applied to KV cache
- 4.6x KV cache compression at 98% of FP16 quality/speed
- Fused Metal kernels for the quantize/dequantize operation

**For mllm**: KV cache quantization is a meaningful target. Current KV cache at 1024 context, 12 layers, 12 heads, fp16 = 37.7MB. With TurboQuant-style compression (4.6x): ~8.2MB. This reduces KV cache bandwidth by 4.6x during decode.

More relevant after GQA is implemented (GQA reduces KV size 3x first, then quantizing it further reduces by 4.6x → ~2.7MB total KV cache).

---

## MLX `mx.compile()` for Inference

MLX's `mx.compile()` traces the computation graph and reuses compiled kernels for repeated calls with the same shapes. For inference:

```python
@mx.compile
def generate_token(x, cache):
    return model(x, cache=cache)
```

This avoids Python overhead and Metal kernel recompilation on every token. Expected benefit: 10-30% throughput improvement for small models where Python overhead is a significant fraction of total time.

---

## MLX vs Other Frameworks: Summary

| Framework | Throughput | TTFT | Memory | Quantization |
|-----------|-----------|------|--------|--------------|
| MLX (vllm-mlx) | Highest | Low | Good | int4, int8, NF4 |
| MLC-LLM | High | Lowest | Good | AWQ, GPTQ, FP8 |
| llama.cpp | Medium | Medium | Good | Q4_0-Q8_0, K-quants |
| Ollama (old, llama.cpp) | Low-Medium | Medium | OK | GGUF formats |
| Ollama (new, MLX) | High | Low | Good | MLX int4 |
| PyTorch MPS | Low | High | Poor | Limited |

**Note**: Ollama switched to MLX backend for Apple Silicon in 2026, gaining significant speedups (from llama.cpp's ~150 t/s to MLX's ~230+ t/s for comparable models).

---

## Key MLX Limitations

1. **No distributed inference**: MLX doesn't support multi-device inference (relevant for Mac Pro with two M-chips)
2. **No batched training on MPS**: Not a goal (training is CUDA/Triton based)
3. **Python overhead for small models**: At 117M, Python dispatch time is non-negligible. `mx.compile()` helps.
4. **No streaming prefetch API**: Metal's `simdgroup_async_copy` not exposed; can't overlap load/compute in Python API
5. **quantized_matmul only supports specific group sizes**: group_size must be in {32, 64, 128, 256}
6. **No Flash Attention Metal kernel**: Uses `mx.fast.scaled_dot_product_attention` which is a general implementation, not the fully-tiled MFA

---

## Practical Recommendations for mllm

1. **Use `mx.compile()` on the inference forward pass**: Reduce Python overhead. Should give 10-30% TPS improvement.
2. **Profile with Metal Instruments**: Identify which layers/ops are actually bottlenecking. Assumption: linear layers dominate, but verification needed.
3. **Enable GPU residency**: Use Metal residency sets (macOS 15+) to pin model weights and avoid eviction latency.
4. **Quantize KV cache**: After GQA, quantize KV cache to int8 (2x reduction in KV bandwidth).
5. **Monitor MLX releases**: Apple is actively improving (Metal 4, Neural Accelerator support). Some optimizations come for free with framework updates.
