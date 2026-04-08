# Metal Execution Model & GPU Architecture

## Object Hierarchy

```
MTLDevice
  └── MTLCommandQueue
        └── MTLCommandBuffer
              └── MTLComputeCommandEncoder
                    └── dispatch to MTLComputePipelineState
                          └── compiled from MTLFunction
                                └── sourced from MTLLibrary
```

**MTLDevice** — abstraction over physical GPU. On Apple Silicon: always one device.

**MTLLibrary** — compiled MSL. AOT (`xcrun metal`) or JIT (`device.makeLibrary(source:)`). JIT costs tens–hundreds of ms. Build once, reuse.

**MTLComputePipelineState** — fully linked kernel object. Two critical queries:
- `threadExecutionWidth` → **32** on all Apple Silicon
- `maxTotalThreadsPerThreadgroup` → typically 1024, lower if kernel uses heavy resources

**Encoder flavors:**
- Compute — ML kernels
- Render — vertex/fragment (irrelevant for ML)
- Blit — memcpy/fill between buffers

**Dispatch methods:**
- `dispatchThreads(_:threadsPerThreadgroup:)` — Metal handles boundary clamping
- `dispatchThreadgroups(_:threadsPerThreadgroup:)` — exact threadgroup count, developer manages boundaries

---

## Thread Hierarchy

```
Grid (3D)
  └── Threadgroup (3D, up to 1024 threads total)
        └── SIMD-group (32 threads, hardware warp)
              └── Thread (smallest unit)
```

**SIMD-group = 32 threads, universal across all Apple Silicon (Apple7/8/9).**

Each SIMD-group has:
- Shared program counter
- 32-bit execution mask (for divergent branches via masking)
- 128 GPRs (32-bit), accessible as 16-bit halves or 64-bit pairs
- 256 uniform registers (shared across all 32 threads, compiler-managed)

Divergent branches both execute — inactive threads run as no-ops.

### Threadgroup Limits (Apple7/8/9 — All Generations)

| Parameter | Value |
|-----------|-------|
| Max threads per threadgroup | 1024 |
| Max threadgroup memory | 32 KB |
| Max threadgroups per grid (per dim) | 65,535 |
| SIMD width | 32 |

---

## Memory Model

### Address Spaces

| Space | MSL Qualifier | Scope | Notes |
|-------|---------------|-------|-------|
| Device memory | `device` | All threads | Unified with CPU on Apple Silicon |
| Threadgroup memory | `threadgroup` | Within threadgroup | 32 KB, fast scratchpad |
| Constant memory | `constant` | Read-only all | ~64 KB, preloaded into uniform regs |
| Thread-local | `thread` | Per-thread stack | Register file / stack |
| Texture | `texture<>` | All threads | Cached, sampled access |

### Memory Hierarchy — Sizes and Latency

| Level | Size | Latency | Notes |
|-------|------|---------|-------|
| Register file (per core) | ~208 KB | 2–3 cycles | Live thread registers |
| Threadgroup / shared | 32 KB per threadgroup | ~10 cycles | Programmer-managed scratchpad |
| L1 data cache (per core) | 8 KB | Fast | Tiny — careful |
| L2 cache | 768 KB (M1), ~3 MB (M2 Pro per cluster) | ~50 ns | Major improvement M1→M2 |
| System Level Cache (SLC) | 8 MB (M1), 8–48 MB (Pro/Max) | ~234 ns | Shared CPU+GPU |
| DRAM (unified) | up to 192 GB | >342 ns | No PCIe overhead |

**SIMD shuffle bandwidth: 256 B/cycle** — fastest inter-thread communication, zero memory traffic.

### Key: float4 is NOT wide SIMD

Apple GPUs have a scalar ISA. `float4` decomposes to 4 scalar ops. For peak throughput: use **8+ independent accumulator chains** (ILP), not vector types. Naive float4 measures ~815 GFLOPS on M5; scalar with ILP measures ~3,849 GFLOPS — a 4.7× gap.

---

## Occupancy and Waves

### Static Allocation (M1/M2, Apple7/Apple8)

Register file per core: **208 KB** shared across all active threads.

| Registers per thread | Max threads per core | SIMD-groups per core |
|---------------------|---------------------|---------------------|
| ≤ 104 (≤ 208 half-words) | 1024 | 32 |
| 112 | 960 | 30 |
| 128 | 832 | 26 |
| 192 | 576 | 18 |
| 256 | 384 (minimum) | 12 |

Occupancy drops in **64-thread increments**. Target: 1K–2K concurrent threads per GPU core.

Threadgroup memory also limits occupancy: full 32 KB per threadgroup constrains concurrent threadgroups.

### Dynamic Caching (M3/A17 Pro, Apple9+)

Critical architectural shift. Pre-M3: register file statically allocated at peak usage. M3+: **register cache** — allocated/deallocated dynamically as the shader executes.

Effects:
- Kernels with conditional paths no longer pay peak register cost for unused branches
- A shader peaking at 256 registers (384 threads/core on M1) can sustain 1024 threads/core on M3 if most paths use fewer
- Threadgroup memory, tile memory, and stack are all dynamically cached — unused SRAM is reallocated elsewhere

**Practical implication:** On M3+, register pressure matters less for occupancy (still matters for latency hiding). On M1/M2, minimizing peak registers directly controls occupancy.

---

## Metal Shading Language (MSL)

### Key Data Types

| Type | Width | Notes |
|------|-------|-------|
| `half` | 16-bit | IEEE 754 binary16, native on all Apple Silicon |
| `float` | 32-bit | IEEE 754 single |
| `bfloat` | 16-bit | bfloat16, MSL 3.0+ |
| `half2`, `float4`, etc. | Vector | Syntactic sugar, decomposed to scalar ops |

**`float ↔ half` conversion is free** (zero ALU cycles, A8 and later). Use `half` aggressively.

**Precision gotcha:** `half result = x * 0.5;` — literal `0.5` is float. Use `0.5h` to stay in half.

### Fast Math (Default ON)

Metal enables fast math by default:
- No NaN/Inf handling
- FMA fusion allowed
- Lower-precision transcendentals
- Up to 50%+ gain

Disable: `MTLCompileOptions.fastMathEnabled = false`. Explicit FMA: `fma(a, b, c)`.

### Instruction Throughput (cycles per SIMD)

| Op | Throughput | Latency |
|----|-----------|---------|
| FADD/FMUL (FP16) | 1 | 2.17 |
| FFMA (FP16) | 1 | 2.18 |
| FADD/FMUL (FP32) | 2 | 2.21 |
| FFMA (FP32) | 2 | 2.21 |
| IMUL32 | 4 | 4.02 |
| RECIP32 | 6 | 6 |
| RSQRT32 | 8 | 8 |

### SIMD-Group Intrinsics

Zero extra memory traffic, operate within 32-thread SIMD-group:

```metal
T simd_sum(T value);                        // sum all 32 lanes
T simd_min(T value);
T simd_max(T value);
T simd_prefix_inclusive_sum(T value);
T simd_broadcast(T value, ushort lane);
T simd_shuffle(T value, ushort lane);       // gather from any lane
T simd_shuffle_down(T value, ushort delta); // shift down
bool simd_all(bool value);
simd_vote simd_ballot(bool value);          // bitmask of true lanes
```

### simdgroup_matrix — Hardware Tensor Cores (Apple7+)

```metal
#include <metal_simdgroup_matrix>

simdgroup_float8x8 A, B, C, D;
simdgroup_multiply_accumulate(D, A, B, C);  // D = A × B + C
simdgroup_load(A, src, stride, offset, transpose);
simdgroup_store(A, dst, stride, offset);
```

Each SIMD-group 8×8 matmul = 512 FMAs. Uses regular FP pipeline but reduces register pressure.
Available from Apple7 (M1). Deprecated in Apple9 in favor of Metal Performance Primitives.

### Compute Kernel Built-ins

```metal
kernel void myKernel(
    uint tid   [[thread_index_in_threadgroup]],
    uint3 gpos [[thread_position_in_grid]],
    uint3 tgpos [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]]
)
```

---

## Dispatching: Grid/Threadgroup Size Rules

- Threadgroup size must be a **multiple of 32** (SIMD width)
- Total threads per threadgroup ≤ 1024
- Read `MTLComputePipelineState.maxTotalThreadsPerThreadgroup` at runtime — don't hardcode 1024

**Memory-bound kernels** (transformer decode): larger threadgroup (256–1024) for better coalescing.

**Compute-bound kernels** (prefill GEMM): threadgroup = tile workers, typically 128–512 threads.

**SIMD-group matrix kernels**: 4–16 SIMD-groups per threadgroup = 128–512 threads.

Patterns:
```metal
// Elementwise: Grid=(N,1,1), TG=(256,1,1)
// Matrix-vector: Grid=(rows,1,1), TG=(32,1,1) — one SIMD per row
// Reduction: Grid=(batch*seq,1,1), TG=(128,1,1)
```

---

## Synchronization

### threadgroup_barrier

```metal
threadgroup_barrier(mem_flags::mem_none);        // execution fence only
threadgroup_barrier(mem_flags::mem_threadgroup); // + flush threadgroup memory
threadgroup_barrier(mem_flags::mem_device);      // + flush device memory
```

All threads in threadgroup must reach this before any proceed. Use minimum necessary flag.

### simdgroup_barrier (Faster)

```metal
simdgroup_barrier(mem_flags::mem_threadgroup);
```

Only synchronizes within 32-thread SIMD-group. Much cheaper. Use when threadgroup = 1 SIMD-group or you can restructure to avoid cross-SIMD sync.

### Atomic Operations

```metal
device atomic_int* counter;
atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);
atomic_fetch_min_explicit(counter, val, memory_order_relaxed);
atomic_compare_exchange_weak_explicit(counter, &expected, desired, ...);
```

**Reduction pattern** (SIMD → threadgroup → device):
```metal
float partial = simd_sum(my_value);
if (simd_lane_id == 31)
    atomic_fetch_add_explicit(&tg_sum, partial, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (tid == 0)
    atomic_fetch_add_explicit(&device_sum, tg_sum, memory_order_relaxed);
```

---

## Int4/Int8 Quantization in Metal

### MLX Packing Format

For int4: 8 values per `uint32` (4 bits × 8 = 32 bits).
```
uint32 = v0 | (v1 << 4) | (v2 << 8) | ... | (v7 << 28)
```

Each group of 64 weights shares one `float scale` + one `float bias` (zero-point).

### Dequantization (MLX pattern — no shifts, prescaled activations)

Instead of shifting each nibble right, MLX prescales the activation vector before the dot product:

```metal
// Prescale activation to fold in divisors for nibble positions:
x_thread[i]   = x[i];           // full scale
x_thread[i+1] = x[i+1] / 16.0; // compensates for (packed & 0x00f0) not shifting
x_thread[i+2] = x[i+2] / 256.0;
x_thread[i+3] = x[i+3] / 4096.0;

// Dot product — no shifts needed:
const device uint16_t* ws = (const device uint16_t*)w;
accum += (x_thread[4*i]   * (ws[i] & 0x000f) +
          x_thread[4*i+1] * (ws[i] & 0x00f0) +
          x_thread[4*i+2] * (ws[i] & 0x0f00) +
          x_thread[4*i+3] * (ws[i] & 0xf000));
```

### LUT Dequantization — Reality Check

MLX does **not** use LUT dequantization. The bitmask + prescaled-activation approach is already near-optimal for Apple Silicon because:
- Accumulation stays in float32 (MLX `AccumType = float` in quantized kernels)
- A separate LUT would require: unpack index → register table lookup → multiply — no net win
- llama.cpp Metal kernels also use bitmask extraction, not LUT

### qmv (Matrix-Vector, Decode) Pattern

```metal
// 2 SIMD-groups per threadgroup, 4 output rows per SIMD-group
for (int k = lane_id; k < K; k += 32) {
    // Load packed int4 weights, dequantize via prescaled-x pattern
    // Multiply by input vector element
    // Accumulate into 4 output accumulators
}
for (int r = 0; r < N_R0; r++)
    accumulators[r] = simd_sum(accumulators[r]);
```

Split-K variant (`qvm_split_k`) for K ≥ 1024: 8–32 threadgroups per output, followed by reduction pass.

---

## Roofline Model

Ridge point = peak GFLOPS / peak GB/s. Below: memory-bound. Above: compute-bound.

| Chip | BW (GB/s) | FP32 GFLOPS | Ridge Point |
|------|-----------|------------|-------------|
| M1 | 67 | 2,617 | 39 FLOP/byte |
| M2 | 100 | 3,579 | 36 FLOP/byte |
| M4 | 120 | ~4,260 | 35 FLOP/byte |
| M4 Max | 546 | ~18,430 | 34 FLOP/byte |

### Transformer Op Arithmetic Intensities

| Operation | AI (FLOP/byte) | Regime (decode, batch=1) |
|-----------|---------------|--------------------------|
| GEMM (prefill, large N) | ~N | Compute-bound |
| Matrix-vector (decode) | ~2 (FP16) | **Memory-bound** |
| Int4 matmul (decode) | ~8 (8× less data) | **Memory-bound** |
| Softmax | ~4–8 | Memory-bound |
| Elementwise (SiLU, SwiGLU) | ~1–2 | Memory-bound |

**Decode is always memory-bound.** Token throughput ≈ memory_bandwidth / model_size_bytes.

Int4 doesn't escape memory-bound territory (ridge ~35 FLOP/byte vs AI ~8), but reads 8× less data → ~8× more tokens/sec.

---

## MLX `mx.fast.metal_kernel` Under the Hood

```python
kernel = mx.fast.metal_kernel(
    name="my_kernel",
    input_names=["inp"],
    output_names=["out"],
    source="""
        uint i = thread_position_in_grid.x;
        out[i] = inp[i] * 2.0f;
    """,
)
result = kernel(
    inputs=[input_array],
    output_shapes=[(N,)],
    output_dtypes=[mx.float32],
    grid=(N, 1, 1),
    threadgroup=(256, 1, 1),
    template=[("T", mx.float32)],
)
```

1. **Signature auto-generated**: MLX wraps source in `[[kernel]] void` signature. Inspects for `inp_shape`, `inp_strides`, `inp_ndim` references and adds them automatically.
2. **Compilation**: JIT via `device.makeLibrary(source:)`. Cached by name + template hash. Build object **once at module level** — each `metal_kernel()` call creates a new MTLLibrary.
3. **Dispatch**: Uses `dispatchThreads(grid, threadsPerThreadgroup)` — Metal handles boundary clamping.
4. **`template`**: each unique `(name, value)` combo JIT-compiles a distinct kernel specialization.
5. **`ensure_row_contiguous=True` (default)**: auto-copies non-contiguous inputs. Set False + use `elem_to_loc()` to handle strides manually.
6. **`init_value=float`**: pre-fills outputs on GPU before kernel runs.
7. **`atomic_outputs=True`**: wraps outputs as `device atomic<T>*`.
8. **`verbose=True`**: prints generated full source at dispatch.

**Limitation:** no control over occupancy hints, max_threads, or threadgroup memory size from Python — only available in raw `.metal` files via `[[kernel, max_total_threads_per_threadgroup(N)]]`.

---

## M5 Neural Accelerators (Apple A19/M5, 2025)

Each GPU core has a **dedicated matrix multiply unit**:
- 1024 FP16 MACs per cycle per GPU core
- ~2048 INT8 OPS per cycle per GPU core
- Peak: ~7.4 FP16 TFLOPS for A19 (5 cores), ~70 TFLOPS est. for M5 Max (40 cores)
- Separate from regular FP pipeline (unlike `simdgroup_matrix` which shares ALUs)

**Access:** Only via Metal Performance Primitives (MPP), not MSL directly (as of Xcode 26.1). MLX integration in progress (issue #2693).

M4 has **no GPU neural accelerator** — relies on Neural Engine (separate block) and `simdgroup_matrix` via regular pipeline.

---

## Sources

- [Apple G13 GPU Architecture (Dougall Johnson)](https://dougallj.github.io/applegpu/docs.html)
- [Metal Benchmarks — Philip Turner](https://github.com/philipturner/metal-benchmarks)
- [Metal Feature Set Tables PDF](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)
- [Metal Shading Language Spec v4](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Scale Compute Workloads Across Apple GPUs — WWDC22](https://developer.apple.com/videos/play/wwdc2022/10159/)
- [Advanced Metal Shader Optimization — WWDC16](https://developer.apple.com/videos/play/wwdc2016/606/)
- [Explore GPU Advancements in M3 and A17 Pro — Apple Tech Talks](https://developer.apple.com/videos/play/tech-talks/111375/)
- [Asahi GPU Part III — Alyssa Rosenzweig](https://alyssarosenzweig.ca/blog/asahi-gpu-part-3.html)
- [MLX Custom Metal Kernels Docs](https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html)
- [Apple HPC Evaluation arxiv 2502.05317](https://arxiv.org/html/2502.05317v1)
- [M5 GPU Roofline — Michael Stinkerings](https://www.michaelstinkerings.org/apple-m5-gpu-roofline-analysis/)
- [Chips and Cheese: M2 Pro iGPU](https://chipsandcheese.com/p/a-brief-look-at-apples-m2-pro-igpu)
- [Apple Neural Accelerators Benchmark (A19/M5)](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)
- [Exploring LLMs with MLX and M5 — Apple ML Research](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
