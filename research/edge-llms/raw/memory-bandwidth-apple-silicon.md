# Memory Bandwidth Optimization on Apple Silicon

## Sources
- Apple M5 MLX paper: https://machinelearning.apple.com/research/exploring-llms-mlx-m5
- Andreas Kunar medium analysis: https://medium.com/@andreask_75652/thoughts-on-apple-silicon-performance-for-local-llms-3ef0a50e08bd
- llama.cpp M-series discussion: https://github.com/ggml-org/llama.cpp/discussions/4167
- Native LLM inference paper: https://arxiv.org/html/2601.19139v2
- MLX vs llama.cpp analysis: https://groundy.com/articles/mlx-vs-llamacpp-on-apple-silicon-which-runtime-to-use-for-local-llm-inference/

---

## Apple Silicon Memory Architecture

### Unified Memory
CPU, GPU, and Neural Engine share a **single physical memory pool** with a single high-bandwidth LPDDR bus. No PCIe transfer overhead between CPU and GPU.

**Consequence**: Model weights loaded once into memory are accessible to both CPU and GPU without copying. GPU can access model weights at full memory bandwidth from the start.

### Memory Bandwidth by Chip (as of 2025)

| Chip | Memory BW | Max RAM |
|------|-----------|---------|
| M4 (base) | 120 GB/s | 32 GB |
| M4 Pro | 273 GB/s | 64 GB |
| M4 Max | 546 GB/s | 128 GB |
| M5 (base) | 153 GB/s | 32 GB (+28% vs M4) |
| M5 Max | ~650 GB/s | 128 GB (est.) |

**Comparative**: NVIDIA RTX 4090 has ~1000 GB/s GDDR6X bandwidth but discrete GPU (PCIe transfer overhead for model loading). For LLM inference where model weights must be loaded each step, the PCIe bottleneck hurts NVIDIA on smaller models in single-user scenarios.

---

## Compute vs Memory Bound Analysis

### Theory: Arithmetic Intensity
For a linear layer computing y = Wx:
- Weights: M × N parameters
- Operations: 2MN FLOPs (multiply + add)
- Memory access: MN × sizeof(weight) bytes
- Arithmetic intensity = 2MN / (MN × bytes) = 2 / bytes_per_param

**At int4 (0.5 bytes/param)**: intensity = 4 FLOP/byte
**At fp16 (2 bytes/param)**: intensity = 1 FLOP/byte

**Roofline analysis for M4**:
- Peak FP16 compute: ~38 TFLOPS (GPU)
- Memory bandwidth: 120 GB/s
- Compute/bandwidth ratio: 38T / 120G = ~317 FLOP/byte

→ Any operation with arithmetic intensity < 317 is **memory-bound** on M4.
→ At int4: intensity = 4 FLOP/byte → **deeply memory-bound** (4 vs 317)
→ The model is not compute-limited; we're bottlenecked by how fast we can read weights

### What this means for 117M model

**Decode phase** (generating one token at a time):
- Load all 117M weights (at int4: ~70MB)
- Perform ~117M × 2 = 234M FLOPs of compute
- Arithmetic intensity ≈ 234M / 70MB = 3.3 FLOP/byte → **memory-bound**
- Theoretical max TPS = 120 GB/s / 70 MB = **1,714 TPS** (memory bandwidth limit at int4)
- Current: 625 TPS → we're at ~36% of theoretical memory bandwidth limit at int4

**Prefill phase** (processing prompt of length T):
- Load weights once, process T tokens in parallel
- Arithmetic intensity grows linearly with T (each weight participates in T computations)
- At T=1024: intensity ≈ 3.3 × 1024 = 3,379 FLOP/byte → **compute-bound**
- Theoretical limit = ~38 TFLOPS / (234M × 2 × 1024) = far from bottleneck

**Key insight**: For decode, we need to get faster at memory access, not compute. For prefill, we need to maximize GPU FLOP utilization.

---

## Sources of Inefficiency in Current Stack

### Weight loading
- int4 weights: 70MB per pass (all 12 layers, every token)
- MLX lazy evaluation may cause suboptimal memory access patterns
- Metal GPU cache: L2 cache is ~6-16MB on M4 → entire 70MB model doesn't fit → cold cache every step

### KV cache
- Current KV cache at 1024 tokens, 12 layers, 12 heads, 64 dim, fp16:
  - Size = 12 × 2 × 12 × 64 × 1024 × 2 = 37.7 MB
- With int4 quantization of KV cache: 9.4 MB
- Loading KV cache per step adds to bandwidth pressure

### Memory bandwidth utilization
- Current: 625 TPS × 70MB = 43.75 GB/s effective weight bandwidth
- M4 peak: 120 GB/s
- **Utilization: ~36%** — significant headroom

---

## Strategies to Improve Memory Bandwidth Utilization

### 1. Reduce weight size (quantization)
- int4 already implemented (current)
- int2/NF4 could further reduce to 35MB — 2x reduction but severe quality loss
- **Not recommended** at 117M scale (quality too impactful)

### 2. Prefetch / pipeline weight loading with compute
- Load next layer's weights while computing current layer
- Requires explicit async copy in Metal kernel
- Available via `simdgroup_async_copy` (M2+)
- Estimated benefit: 10-20% TPS improvement by hiding memory latency

### 3. Fuse operations to reduce round-trips
- Current fused SwiGLU is an example: load weights once, apply gate + activation in one kernel pass
- More fusion opportunities: fuse RMSNorm into attention input, fuse residual add
- Each fusion reduces the number of times activations go to/from memory

### 4. Reduce KV cache bandwidth
- GQA (n_kv_heads=4): 3x smaller KV cache → 3x less bandwidth for KV load per decode step
- int8 KV cache: 2x reduction with minimal quality loss
- Current: 37.7MB KV cache at fp16, 12 heads
- With GQA-4 + int8 KV: ~4.7MB → substantial bandwidth savings

### 5. GPU memory wiring (residency sets)
- On macOS 15+: pin model weights as GPU-resident
- Prevents OS from evicting weights between inference calls (relevant for interactive use)
- Can eliminate 10-50ms latency spikes on first token after idle period

### 6. Activation checkpointing (inference)
- For very long sequences in prefill: recompute some activations rather than store all
- Trades compute for memory — beneficial when activation memory exceeds GPU L2

---

## Theoretical TPS Ceiling Analysis

For mllm at decode time (memory-bound regime):

| Config | Weight Size | M4 BW | Theoretical TPS |
|--------|-------------|--------|-----------------|
| fp16 | 234 MB | 120 GB/s | 513 TPS |
| int8 | 117 MB | 120 GB/s | 1,026 TPS |
| int4 (current) | ~70 MB | 120 GB/s | 1,714 TPS |
| int4 + GQA-4 | ~70 MB weight + 4.7MB KV | 120 GB/s | ~1,600 TPS |

Current achievement: 625 TPS (36% efficiency). Headroom exists to 3x improve TPS before hitting bandwidth ceiling, without any architectural changes to the model.

**Why 36% not 100%?**:
1. Metal kernel dispatch overhead
2. Sequential layer execution (layer N+1 can't start until layer N finishes, even if weights are loaded)
3. Attention computation overhead (even for small model)
4. MLX framework overhead (lazy evaluation, Python dispatch)
5. GPU not 100% occupied during compute phases between loads

---

## M4 vs M4 Pro for mllm

The 117M model (70MB int4) fits easily in M4 base's 32GB RAM with massive room to spare. The bottleneck is the 120 GB/s bandwidth.

M4 Pro (273 GB/s) would give **2.3x theoretical speedup** at decode — model would potentially reach 1,400+ TPS. This is the "right" answer for someone who wants maximum TPS from mllm without changing the model architecture.

**Practical recommendation**: For M4 base (120 GB/s), focus on reducing overhead to approach 1,000+ TPS before targeting the bandwidth ceiling. The current 625 TPS suggests significant overhead reduction opportunities in the MLX inference loop.
