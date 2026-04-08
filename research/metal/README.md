# Metal Research

Deep research on Apple Metal GPU architecture and its application to MLX-based transformer inference.

## Files

| File | Contents |
|------|----------|
| [01_metal_execution_model.md](01_metal_execution_model.md) | Metal object hierarchy, thread/memory model, MSL, occupancy, synchronization, int4 dequant patterns, roofline |
| [02_chip_families.md](02_chip_families.md) | M1→M4, A16→A18 specs table; Neural Engine deep-dive; Dynamic Caching; unified memory; decode bandwidth ceilings |
| [03_mlx_kernel_optimization.md](03_mlx_kernel_optimization.md) | `mx.fast.metal_kernel` internals; tiled GEMM; attention kernel dispatch; fused SwiGLU; KV cache memory; current 625 TPS gap analysis |
| [04_coreml_iphone.md](04_coreml_iphone.md) | PyTorch → CoreML export; stateful KV cache; ANE constraints; int4 palettization; iPhone memory limits; conversion gotchas |
| [05_conclusions.md](05_conclusions.md) | Actionable conclusions: two bugs causing ~70% TPS gap, iPhone ceiling analysis, priority order |

## Key Findings

**Current 625 TPS at int4 is ~30% of theoretical ceiling (~2051 TPS).** Two bugs, not optimizations:

1. **Manual attention** (`q @ k.T + softmax`) bypasses `sdpa_vector` kernel — switching to `mx.fast.scaled_dot_product_attention` enables the optimized decode path
2. **`mx.concatenate` KV cache** copies ~18.87 MB per step at T=512 (~10% of M4's 120 GB/s budget)

Everything else (LUT dequant, tiled matmul, fused RMSNorm) is secondary.

**iPhone target (A18):** 60 GB/s bandwidth, 8 GB RAM, no MLX. CoreML is the only path. Expected ~20–50 TPS for 117M at int4 palettization — hardware ceiling, not an optimization problem.

**Metal kernel portability:** Apple9 Metal family covers M3, M4, A17 Pro, A18, A18 Pro. Shaders compile once and run on all targets.
