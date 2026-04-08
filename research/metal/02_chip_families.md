# Apple Silicon Chip Families: GPU & ML Reference

## GPU Core Microarchitecture (Constant Across All Generations)

| Parameter | Apple7 (M1) | Apple8 (M2, A16) | Apple9 (M3, M4, A17, A18) |
|-----------|-------------|-----------------|--------------------------|
| EUs per GPU core | 16 | 16 | 16 |
| ALUs per EU | 8 | 8 | 8 |
| ALUs per GPU core | 128 | 128 | 128 |
| SIMD width | 32 | 32 | 32 |
| Threadgroup memory | 32 KB | 32 KB | 32 KB (dynamic pool) |
| Max threads/threadgroup | 1024 | 1024 | 1024 |
| Register file per core | ~208 KB | ~208 KB | ~208 KB (dynamically shared) |
| FP32 ops/cycle/core | 256 | 256 | 256 |
| FP16 ops/cycle/core | 256 | 256 | 256 |

FP16 ALU throughput = FP32 (same ALU, each handles 1 op/cycle regardless of precision). FP16 advantage is bandwidth and register density, not extra compute GFLOPS.

---

## M1 Family (Apple7, TSMC N5, 2020–2022)

| Chip | GPU Cores | FP32 TFLOPS | Mem BW (GB/s) | Neural Engine TOPS | Max RAM |
|------|-----------|-------------|---------------|-------------------|---------|
| M1 | 7–8 | 2.62 | 68.3 | 11 | 16 GB |
| M1 Pro | 14–16 | 5.31 | 204.8 | 11 | 32 GB |
| M1 Max | 24–32 | 10.62 | 409.6 | 11 | 64 GB |
| M1 Ultra | 48–64 | 15.9–21.2 | 819.2 | 22 (2× die) | 128 GB |

**Memory:** LPDDR5. M1 base: 4-ch 64-bit at 68 GB/s. M1 Pro: 256-bit. M1 Max: 512-bit. M1 Ultra: 1024-bit.

**M1 Ultra:** Two M1 Max dies via UltraFusion (2.5 TB/s die-to-die interconnect). Logical single device but cross-die ops pay inter-die bandwidth.

**Clock:** ~1.278 GHz (M1), ~1.296 GHz (Pro/Max).

---

## M2 Family (Apple8, TSMC N5P, 2022–2023)

| Chip | GPU Cores | FP32 TFLOPS | Mem BW (GB/s) | Neural Engine TOPS | Max RAM |
|------|-----------|-------------|---------------|-------------------|---------|
| M2 | 8–10 | 3.58 | 100 | 15.8 | 24 GB |
| M2 Pro | 16–19 | 6.8 | ~200 | 15.8 | 32 GB |
| M2 Max | 30–38 | 13.6 | ~400 | 15.8 | 96 GB |
| M2 Ultra | 60–76 | 21.3–27.2 | ~800 | 31.6 (2× die) | 192 GB |

**Memory:** LPDDR5-6400. M2 base: 128-bit at 100 GB/s. Same bus widths as M1 per tier.

**GPU improvements vs M1:** ~35% faster at same power (2 extra cores + higher clock ~1.398 GHz vs 1.296 GHz). Neural Engine: 15.8 TOPS (up from 11, ~40% faster).

---

## M3 Family (Apple9, TSMC N3B, 2023)

| Chip | GPU Cores | FP32 TFLOPS | Mem BW (GB/s) | Neural Engine TOPS | Max RAM |
|------|-----------|-------------|---------------|-------------------|---------|
| M3 | 8–10 | ~3.58 | 102.4 | 18 | 24–64 GB |
| M3 Pro | 14–18 | ~6.5–7.4 | 153.6 | 18 | 36 GB |
| M3 Max | 30–40 | 12.3–16.4 | 307–409.6 | 18 | 128 GB |
| M3 Ultra | 60–80 | ~32.8 | ~819 | 36 (2× die) | 192 GB |

**Memory:** LPDDR5-6400 maintained. 3 nm gives power/area benefits but minimal raw TFLOPS gain over M2 at base level. Clock ~1.40 GHz.

**Neural Engine:** 18 TOPS (up from 15.8 in M2).

### New in Apple9: Dynamic Caching

Apple9's GPU replaces the fixed register file allocation with a **hardware-managed dynamic register cache**.

Pre-M3: entire SIMD-group's register budget was statically allocated at pipeline creation time, based on *peak* register usage. Registers used by only some branches still consumed the full budget.

M3+: the hardware scheduler **allocates on-chip memory (registers, threadgroup, buffer cache) dynamically per-dispatch**. The previously distinct pools (register cache, buffer cache, tile cache) merge into one unified on-chip pool with hardware partitioning.

**Effect on ML workloads:** Higher average GPU occupancy without code changes. A kernel that peaked at 256 registers (384 threads/core on M1) can sustain 1024 threads/core on M3 when most paths use fewer. The `[[kernel, max_total_threads_per_threadgroup(N)]]` annotation gives hints but Dynamic Caching often exceeds the nominal 32 KB threadgroup memory in practice when competing uses are absent.

**Also new in Apple9:** Hardware ray tracing, hardware mesh shaders (both irrelevant for ML).

---

## M4 Family (Apple9, TSMC N3E, 2024–2025)

| Chip | GPU Cores | FP32 TFLOPS | Mem BW (GB/s) | Neural Engine TOPS | Max RAM |
|------|-----------|-------------|---------------|-------------------|---------|
| M4 | 8–10 | ~3.5–4.4 | 120 | 38 | 32 GB |
| M4 Pro | 16–20 | 7.4–9.22 | 273 | 38 | 64 GB |
| M4 Max | 32–40 | 14.7–18.4 | 410–546 | 38 | 128 GB |

**Same Metal GPU family as M3 (Apple9).** No new GPU ISA. Higher clock (~1.58 GHz vs ~1.40 GHz) gives ~13% raw throughput gain.

**TFLOPS note:** Multiple sources give 2.9–4.4 range for 10-core M4. The 4.4 figure is theoretical (160 EUs × 8 ALUs × 2 ops/cycle × 1.47 GHz). The ~2.9 figure is measured under iPad Pro thermal constraints. MacBook/Mac Mini form factors: 3.5–4.4 TFLOPS sustained.

### ARM SME in M4

M4 CPU performance cores expose **ARM SME (Scalable Matrix Extension)** — ARM's standardized matrix ISA, replacing Apple's private AMX.

SME capabilities:
- Outer-product matrix accumulate (FMOPA)
- Scalable tile registers (ZA array)
- Native BF16 and FP16 data
- User-space accessible (unlike AMX's private kernel APIs)

For transformer inference: SME is optimal for **single-token decode** (small-batch GEMM) with ~0.095 ms dispatch overhead vs ANE's ~95 ms. Not confirmed whether A18 CPU cores expose SME to user-space.

**M4 has NO GPU neural accelerator** (unlike M5). GPU matmul uses `simdgroup_matrix` on the regular pipeline. The 38 TOPS comes from the Neural Engine block, separate from the GPU.

### M4 Neural Engine

- 16 cores, 32 MB on-chip SRAM
- 38 TOPS marketed = 19 TFLOPS FP16 × 2 (INT8 convention). **Measured: 19 TFLOPS FP16**
- INT8 does **not** run 2× faster — ANE dequantizes INT8→FP16 before compute. INT8 saves memory bandwidth only.
- 94% utilization on deep chained graphs (32+ ops). ~30% utilization on single ops (dispatch overhead ~95 ms dominates).
- Working set > 32 MB SRAM: ~30% performance degradation (spills to DRAM)
- 1×1 convolution is **3× faster than equivalent matmul** on ANE — convolution datapath is the fast path
- Efficiency: 6.6 TFLOPS/W (vs GPU ~1–2 TFLOPS/W, vs A100 ~0.08 TFLOPS/W)

---

## A-Series (iPhone Targets)

| Chip | Metal Family | GPU Cores | FP32 TFLOPS (est.) | Mem BW (GB/s) | NE TOPS | RAM | Process |
|------|-------------|-----------|-------------------|---------------|---------|-----|---------|
| A16 Bionic | Apple8 | 5 | ~1.6 | 51.2 | 17 | 6 GB | TSMC N4P |
| A17 Pro | Apple9 | 6 | ~2.1 | ~68–77 | 35 | 8 GB | TSMC N3B |
| A18 | Apple9 | 5 | ~1.9 | 60 | 35 | 8 GB | TSMC N3E |
| A18 Pro | Apple9 | 6 | ~2.3 | 60 | 35 | 8 GB LPDDR5X | TSMC N3E |

**A-series vs M-series GPU:** Architecturally identical cores (same ISA, same ALU design). A Metal kernel compiled for Apple9 runs on both A18 Pro and M3/M4 without changes. Scale differs: M4 Max has 40 GPU cores vs A18 Pro's 6. M-series memory bandwidth is 5–8× higher.

**iPhone RAM ceiling:** 8 GB hard limit. At int4: ~7–8B params maximum practical fit.

**iPhone TDP:** 5–7W sustained vs MacBook Pro M4 Max's 40+ W. Sustained GPU decode triggers thermal throttle on iPhone within seconds.

**A17 Pro bandwidth:** Apple stated "50% more than A16" (51.2 × 1.5 = ~76.8 GB/s; some sources say ~68 GB/s). Exact value unconfirmed.

**A18 bandwidth:** Apple stated "17% more than A17 Pro," confirmed at ~60 GB/s. Lower absolute bandwidth than A17 Pro despite faster process — die area tradeoffs.

---

## Neural Engine: Use vs Avoid

### Supported Operations

- Convolution (primary fast path; 1×1 conv is **3× faster than matmul**)
- Matrix multiply (slower than 1×1 conv by 3×)
- Elementwise (add, mul, relu, gelu, sigmoid, tanh)
- Layer norm, softmax, RMS norm
- Limited gather/scatter

### Precision

- **FP16** (primary, fast path)
- **INT8** (dequantized to FP16 at compute time — bandwidth savings only)
- No FP32, no BF16, no INT4 native compute support

### Tensor Layout Requirement

ANE requires **[Batch, Channels, 1, Sequence]** (channels-first). PyTorch/MLX default (B, S, C) requires transposition. All `nn.Linear` layers must be expressed as 1×1 `nn.Conv2d` for ANE dispatch.

### When to Use ANE vs GPU

| Workload | Preferred Backend | Reason |
|----------|------------------|--------|
| Prefill (long prompt, batch > 1) | ANE (via CoreML) | 19 TFLOPS FP16 at 6.6 TFLOPS/W |
| Single-token decode | GPU (MLX) or SME (CPU) | ANE dispatch overhead ~95 ms kills per-token latency |
| Attention with dynamic sequence length | GPU (MLX) | ANE requires fixed shapes |
| Single op in isolation | GPU | 30% ANE utilization; overhead amortizes badly |
| Deep chained graph (32–64 ops) | ANE | 94% utilization, best energy efficiency |
| INT4 weight quantization | GPU (MLX) | ANE only computes in FP16; no INT4 speedup |
| Sequence length > ~1024 | GPU | ANE SDPA exceeds SRAM beyond ~1024 tokens |
| Battery-constrained (iPhone) | ANE | ~80× more efficient per FLOP than A100 |

**Best practice:** Hybrid — prefill on ANE (CoreML), decode on GPU (MLX). Apple's own research confirms this pattern.

---

## Unified Memory Architecture

### Physical Architecture

CPU, GPU, Neural Engine, and SME/AMX share a **single DRAM pool on-package** (not on-die). Same LPDDR5/LPDDR5X chips serve all compute units. No PCIe bus, no GDDR, no HBM.

### Zero-Copy

Metal buffers with `storageMode:.shared` (default on Apple Silicon): **same physical pages mapped into both CPU and GPU address spaces**. No memcpy. CPU↔GPU "transfers" are cache-coherent reads across a shared memory hierarchy.

System Level Cache (SLC) is shared between CPU and GPU: 8 MB (M1), up to 48 MB (M1 Max), 96 MB (M1 Ultra). Activations that fit in SLC avoid DRAM entirely.

### LLM Decode Bandwidth Ceiling

```
tokens/sec ≈ memory_bandwidth_GB_s / model_size_GB
```

Examples at int4:
- **M4 (120 GB/s):** 117M model (58.5 MB) → ~2051 TPS theoretical. Measured: 625 TPS = ~30% utilization.
- **M4 Max (546 GB/s):** 8B model (4 GB) → ~136 TPS theoretical. Measured: ~93 TPS (~68% BW utilization).
- **A18 (60 GB/s):** 117M model (58.5 MB) → ~1026 TPS theoretical ceiling.

**Memory bandwidth dominates decode throughput** more than GPU TFLOPS. M4 Max at 546 GB/s is competitive with A100 SXM5 at 2 TB/s when normalized — you run a 4× larger model locally.

---

## Full Chip Comparison

| Chip | Metal Family | GPU Cores | FP32 TFLOPS | FP16 TFLOPS | Mem BW (GB/s) | NE TOPS | Max RAM | Process |
|------|-------------|-----------|-------------|-------------|---------------|---------|---------|---------|
| M1 | Apple7 | 7–8 | 2.62 | 5.24 | 68.3 | 11 | 16 GB | N5 |
| M1 Pro | Apple7 | 14–16 | 5.31 | 10.6 | 204.8 | 11 | 32 GB | N5 |
| M1 Max | Apple7 | 24–32 | 10.6 | 21.2 | 409.6 | 11 | 64 GB | N5 |
| M1 Ultra | Apple7 | 48–64 | 15.9–21.2 | 31.8–42.4 | 819.2 | 22 | 128 GB | N5 |
| M2 | Apple8 | 8–10 | 3.58 | 7.16 | 100 | 15.8 | 24 GB | N5P |
| M2 Pro | Apple8 | 16–19 | 6.8 | 13.6 | ~200 | 15.8 | 32 GB | N5P |
| M2 Max | Apple8 | 30–38 | 13.6 | 27.2 | ~400 | 15.8 | 96 GB | N5P |
| M2 Ultra | Apple8 | 60–76 | 21.3–27.2 | 42.6–54.4 | ~800 | 31.6 | 192 GB | N5P |
| M3 | Apple9 | 8–10 | ~3.58 | ~7.16 | 102.4 | 18 | 24–64 GB | N3B |
| M3 Pro | Apple9 | 14–18 | ~6.5–7.4 | ~13–14.8 | 153.6 | 18 | 36 GB | N3B |
| M3 Max | Apple9 | 30–40 | 12.3–16.4 | 24.6–32.8 | 307–409 | 18 | 128 GB | N3B |
| M4 | Apple9 | 8–10 | ~3.5–4.4 | ~7–8.8 | 120 | 38 | 32 GB | N3E |
| M4 Pro | Apple9 | 16–20 | 7.4–9.22 | 14.8–18.4 | 273 | 38 | 64 GB | N3E |
| M4 Max | Apple9 | 32–40 | 14.7–18.4 | 29.4–36.8 | 410–546 | 38 | 128 GB | N3E |
| A16 | Apple8 | 5 | ~1.6 | ~3.2 | 51.2 | 17 | 6 GB | N4P |
| A17 Pro | Apple9 | 6 | ~2.1 | ~4.2 | ~68–77 | 35 | 8 GB | N3B |
| A18 | Apple9 | 5 | ~1.9 | ~3.8 | 60 | 35 | 8 GB | N3E |
| A18 Pro | Apple9 | 6 | ~2.3 | ~4.6 | 60 | 35 | 8 GB LPDDR5X | N3E |

---

## Sources

- [Apple Silicon HPC Evaluation arxiv 2502.05317](https://arxiv.org/html/2502.05317v1)
- [Inside the M4 Neural Engine, Part 1 — Maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Inside the M4 Neural Engine, Part 2 — Maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Orion: Characterizing Apple's ANE for LLM arxiv 2603.06728](https://arxiv.org/html/2603.06728v1)
- [Deploying Transformers on Apple Neural Engine — Apple ML Research](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Exploring LLMs with MLX and M5 Neural Accelerators — Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
- [Native LLM Inference at Scale on Apple Silicon arxiv 2601.19139](https://arxiv.org/html/2601.19139)
- [Apple G13 GPU Reference (dougallj)](https://dougallj.github.io/applegpu/docs.html)
- [Philip Turner metal-benchmarks](https://github.com/philipturner/metal-benchmarks)
- [Disaggregated Inference: NPU prefill + GPU decode — SqueezeBits](https://blog.squeezebits.com/disaggregated-inference-on-apple-silicon-npu-prefill-and-gpu-decode-67176)
- [Explore GPU Advancements in M3 and A17 Pro — Apple WWDC Tech Talks](https://developer.apple.com/videos/play/tech-talks/111375/)
- [Metal Feature Set Tables PDF](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Wikipedia: Apple M1/M2/M3/M4/A16/A17/A18](https://en.wikipedia.org)
- [MTLGPUFamily.apple9 — Apple Developer Docs](https://developer.apple.com/documentation/metal/mtlgpufamily/apple9)
