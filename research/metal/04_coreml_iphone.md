# CoreML & iPhone Inference

On-device LLM deployment path: PyTorch → CoreML → iPhone.

---

## Export Pipeline

### Two Capture Paths

**Path A — `torch.jit.trace` (stable, recommended)**
```python
import torch, coremltools as ct, numpy as np

model.eval()
example_ids = torch.zeros(1, 128, dtype=torch.int32)

traced = torch.jit.trace(model, (example_ids,))

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 512)), dtype=np.int32),
    ],
    outputs=[ct.TensorType(name="logits", dtype=np.float16)],
    minimum_deployment_target=ct.target.iOS18,
    compute_units=ct.ComputeUnit.CPU_AND_GPU,
)
mlmodel.save("gpt.mlpackage")
```

**Path B — `torch.export.export` (beta, coremltools 8.0+)**

~70% op coverage. Usable but treat as experimental for custom ops.

### Mandatory Pre-Conversion Steps

1. `model.eval()` — disables dropout
2. Break weight tying before tracing: `self.head.weight = nn.Parameter(self.tok_emb.weight.clone())`
3. `use_cache=False` for non-stateful version; flip when adding stateful KV cache

`F.scaled_dot_product_attention` is a fused CoreML op on iOS 18. Setting `minimum_deployment_target=ct.target.iOS18` is **mandatory** for LLM performance.

---

## Compute Backends

### ComputeUnit Options

| Constant | What runs where |
|----------|----------------|
| `CPU_ONLY` | All on CPU; always works, slowest |
| `CPU_AND_GPU` | CoreML routes between CPU + Metal GPU |
| `CPU_AND_NE` | CPU + Neural Engine; no GPU |
| `ALL` | CoreML picks per-op; default |

Set at **load time**, not conversion time:
```python
config = ct.models.MLModelConfiguration()
config.computeUnits = ct.ComputeUnit.CPU_AND_GPU
model = ct.models.MLModel("gpt.mlpackage", configuration=config)
```

### How CoreML Decides

Static analysis at model load time (not per inference). Partitions op graph based on:
- Whether op is in ANE's supported op set
- Tensor shapes/dtypes (ANE requires FP16 or INT8; FP32 → CPU)
- iOS version
- Your `ComputeUnit` restriction

**`MLComputePlan` API (iOS 18+):** Query supported/preferred compute device and estimated cost per-op before running.

---

## ANE Constraints

### The Tensor Format Problem

ANE requires **4D channels-first: `(B, C, 1, S)`**. Standard PyTorch transformers use `(B, S, C)`. Every reshape/transpose crossing the S axis costs a 32–64× overhead compared to a normal copy.

**Apple's `ml-ane-transformers` three principles:**

**Principle 1:** Replace `nn.Linear` with `nn.Conv2d(..., kernel_size=1)` (mathematically identical, keeps `(B,C,1,S)` format)

**Principle 2:** Chunk attention per head to improve ANE cache residency

**Principle 3:** Use einsum patterns that avoid transposes:
```python
scores = torch.einsum('bchq,bkhc->bkhq', q_i, k_i) * scale
out    = torch.einsum('bkhq,bchk->bchq', scores, v_i)
# b=batch, c=channel/head_dim, h=1 (spatial singleton), q/k=sequence
```

### Supported vs Unsupported on ANE

**Supported:** Convolution (1×1 = fast path), LayerNorm/RMSNorm, elementwise ops, softmax, SDPA fused (iOS 18)

**Forces CPU fallback:**
- Arbitrary reshapes moving the sequence axis
- Multiple transposes in one attention path
- FP32 tensors (ANE is FP16/INT8 only)
- Dynamic shapes with `RangeDim` (pre-iOS 17.4)
- Large embedding tables (50k+ vocab) — often CPU-bound regardless
- Sequence length > ~1024 (exceeds ANE SRAM)

### When to Use ANE vs GPU

| Workload | Best Backend | Reason |
|----------|-------------|--------|
| Prefill (long prompt) | ANE | 19 TFLOPS FP16 at 6.6 TFLOPS/W |
| Single-token decode | GPU (CoreML) or CPU+SME | ANE dispatch ~95 ms overhead kills latency |
| Dynamic sequence length | GPU | ANE requires static or enumerated shapes |
| Battery-constrained | ANE | ~80× more efficient per FLOP than A100 |

**Best practice:** Hybrid — prefill on ANE, decode on GPU.

---

## Stateful KV Cache (iOS 18+)

Without state: each step copies the full growing KV cache across CPU↔GPU/ANE. At 512 tokens, 12 layers: ~144 MB per step.

With state: CoreML manages the KV buffer internally, in-place on-device. 18× speedup in toy model benchmark.

### PyTorch Model

```python
class GPTWithKVCache(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ... your existing layers ...
        kv_shape = (1, cfg.n_layer, cfg.n_head, cfg.block_size, cfg.head_dim)
        self.register_buffer("k_cache", torch.zeros(kv_shape))
        self.register_buffer("v_cache", torch.zeros(kv_shape))

    def forward(self, input_ids, position, causal_mask):
        # input_ids: (1, 1) — single token
        x = self.tok_emb(input_ids)
        for i, block in enumerate(self.blocks):
            q, new_k, new_v = block.attn.qkv_for_new_token(x)
            self.k_cache[:, i, :, position, :] = new_k
            self.v_cache[:, i, :, position, :] = new_v
            k = self.k_cache[:, i, :, :position+1, :]
            v = self.v_cache[:, i, :, :position+1, :]
            x = block.attn.attend(q, k, v, causal_mask) + x
            x = block.mlp(block.norm2(x)) + x
        return self.head(self.norm(x))
```

### Conversion

```python
kv_shape = (1, cfg.n_layer, cfg.n_head, cfg.block_size, cfg.head_dim)

states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
        name="k_cache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=kv_shape, dtype=np.float16),
        name="v_cache",
    ),
]

mlmodel = ct.convert(
    traced,
    inputs=[
        ct.TensorType(name="input_ids",   shape=(1, 1),   dtype=np.int32),
        ct.TensorType(name="position",    shape=(1,),     dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=(1, 1, 1, cfg.block_size), dtype=np.float16),
    ],
    outputs=[ct.TensorType(name="logits", dtype=np.float16)],
    states=states,
    minimum_deployment_target=ct.target.iOS18,   # mandatory
)
```

### Runtime (Python test)

```python
state = mlmodel.make_state()
for step in range(max_tokens):
    out = mlmodel.predict(
        {"input_ids": token_ids, "position": np.array([step]), "causal_mask": mask},
        state=state,
    )
    next_token = np.argmax(out["logits"])
```

### Swift (actual iPhone)

```swift
let state = try model.makeState()
let output = try await model.prediction(from: input, using: state)
```

---

## CoreML int4 Quantization

### Two Techniques

| Technique | Target | How |
|-----------|--------|-----|
| **Palettization (LUT)** | ANE (iPhone) | K-means cluster weights → lookup table, N-bit |
| **Linear quantization** | GPU (Mac) | Affine `w = scale*(q - zero)`, int4/int8 |

Apple docs: *"INT4 per-block weight quantization works well for GPU on Mac; palettization typically works best on the Neural Engine."* For iPhone, prefer palettization.

### 4-bit Palettization (ANE target, data-free)

```python
import coremltools.optimize as cto

config = cto.coreml.OptimizationConfig(
    global_config=cto.coreml.OpPalettizerConfig(
        mode="kmeans",
        nbits=4,
        granularity="per_grouped_channel",  # iOS 18+
        group_size=16,
    )
)
quantized = cto.coreml.palettize_weights(mlmodel, config)
quantized.save("gpt_4bit_palette.mlpackage")
```

### Linear int4 (GPU target, Mac)

```python
config = cto.coreml.OptimizationConfig(
    global_config=cto.coreml.OpLinearQuantizerConfig(
        dtype="int4",
        granularity="per_block",
        block_size=32,
    )
)
quantized = cto.coreml.linear_quantize_weights(mlmodel, config)
```

### CoreML vs MLX int4 TPS Reality Check

Current MLX int4 on M4: **625 TPS** at 188 MB.

Equivalent CoreML on iPhone 16 (A18): estimated **20–50 TPS** for 117M model. CoreML's per-op dispatch overhead + lower memory bandwidth (A18 ~60 GB/s vs M4's 120 GB/s) explain the gap. The memory budget is fine (int4 palettization lands ~same 60 MB).

---

## iPhone Memory Limits

All iPhone 16 class devices: **8 GB RAM**. iOS `JETSAM` will kill your process around 4–5 GB total app footprint.

Practical limits:
- iOS kernel + system: ~1.5–2 GB baseline
- Safe model footprint: **≤ 4 GB**
- KV cache (512-token, 12-layer, 768-dim FP16): ~18 MB — negligible
- Activations during prefill: hold layer-by-layer, don't materialize full sequence

At int4: 117M model = ~60 MB. Comfortable. 7B model = ~3.5–4 GB. Tight.

---

## A18 vs M4 Comparison

| Spec | A18 | A18 Pro | M4 (base) |
|------|-----|---------|-----------|
| Process | N3E | N3E | N3E |
| GPU cores | 5 | 6 | 10 |
| NE TOPS | 35 | 35 | 38 |
| RAM | 8 GB | 8 GB LPDDR5X | 16–32 GB |
| Mem BW | ~60 GB/s | ~60 GB/s | 120 GB/s |

**Key:** Same Metal GPU family (Apple9) — Metal shaders port directly. Main constraint: 2× lower memory bandwidth than M4. Decode TPS scales linearly with bandwidth.

---

## On-Device Benchmarks

**GPT-2 class via `more-ane-transformers` (M1/M2 ANE):**

| Model | Params | M2 latency | ~TPS |
|-------|--------|-----------|------|
| gpt2 | 124M | ~21 ms/tok | ~48 |
| gpt2-medium | 350M | ~23 ms/tok | ~43 |
| gpt2-large | 774M | ~46 ms/tok | ~22 |
| gpt2-xl | 1.5B | ~95 ms/tok | ~10 |

These run "~99% on ANE" with `CPU_AND_NE` and the Conv2d-substituted architecture.

iPhone-specific numbers aren't widely published. M1 ANE ≈ A18 ANE in per-token throughput (same 16-core NE, slightly less bandwidth). Extrapolated **117M on iPhone 16: 20–50 TPS** at int4 palettization.

---

## Why iPhone Has No MLX

MLX depends on:
1. **`MPSGraph`** (Metal Performance Shaders Graph) — not available on iOS
2. **Runtime Metal shader JIT** (`MTLDevice.makeLibrary(source:)`) — blocked by iOS App Store entitlements

CoreML precompiles shaders at model load time into `.mlmodelc` format → allowed on iOS. MLX-style dynamic kernel compilation → not allowed.

---

## Conversion Gotchas (This Project Specifically)

### 1. Weight Tying

`self.tok_emb.weight = self.head.weight` creates a shared node in traced graph. Before converting:
```python
self.head.weight = nn.Parameter(self.tok_emb.weight.clone())
```

### 2. Causal Mask for KV-Cache Decode

`is_causal=True` in SDPA works for prefill but not for decode (single query attending to all past keys). For the decode-step model, pass an explicit `(1, 1, 1, ctx_len)` mask instead.

### 3. Flexible Shapes → CPU Fallback

`ct.RangeDim` on sequence length forces CPU before iOS 17.4. Production pattern: **`EnumeratedShapes`** with discrete buckets (1, 16, 64, 128, 256, 512). Pad inputs to next bucket.

### 4. `rotate_half` with `torch.cat`

The current `rotate_half` does `torch.cat([-x[..., d:], x[..., :d]])`. Slice indices become dynamic under flexible shapes → CPU fallback. Fix: restructure RoPE as elementwise multiply on `(B, C, 1, S)` tensors using the precomputed cos/sin buffers.

### 5. int/float Type Mismatches

`mask.fill_(0)` vs `mask.fill_(0.0)` can cause trace failures. Audit mask construction for integer literals inside float ops.

### 6. Embedding Table

`nn.Embedding` (50k × 768 = 150 MB at FP16) typically runs on CPU even with `CPU_AND_NE`. Not a blocker — embedding lookup is fast on CPU and not the bottleneck.

### 7. SwiGLU w1/w2 for ANE

For ANE compatibility, replace `nn.Linear` with `nn.Conv2d(..., kernel_size=1)` for both w1 and w2 projections, keeping data in `(B, C, 1, S)` format throughout the MLP.

---

## Recommended Conversion Roadmap

1. **Non-stateful baseline:** `torch.jit.trace` → `ct.convert()` with `iOS18`, `CPU_AND_GPU`. Validate at `atol=1e-3`.

2. **Stateful KV cache:** Restructure Attention to use registered buffers per-layer. Add `StateType` in conversion. Verify 18× latency improvement.

3. **4-bit palettization:** `mode="kmeans"`, `nbits=4`, `per_grouped_channel`. Target ~60 MB footprint.

4. **Benchmark `CPU_AND_NE` vs `CPU_AND_GPU`:** For 117M decode the ANE may be faster; verify with Xcode Instruments → Core ML instrument.

5. **ANE-optimize architecture** (Conv2d substitution, RoPE restructure): only if Step 4 shows a CPU/GPU bottleneck. Significant refactoring; validate parity at each step.

---

## Sources

- [Apple ml-ane-transformers — GitHub + research article](https://machinelearning.apple.com/research/neural-engine-transformers)
- [coremltools stateful models docs](https://apple.github.io/coremltools/docs-guides/source/stateful-models.html)
- [coremltools optimization overview](https://apple.github.io/coremltools/docs-guides/source/opt-overview.html)
- [more-ane-transformers — smpanaro (GPT-2 benchmarks)](https://github.com/smpanaro/more-ane-transformers)
- [HuggingFace swift-coreml-llm blog](https://huggingface.co/blog/swift-coreml-llm)
- [WWDC24 session 10159: Bring your models to Apple Silicon](https://developer.apple.com/videos/play/wwdc2024/10159/)
- [WWDC24 session 10161: Deploy ML/AI on-device](https://developer.apple.com/videos/play/wwdc2024/10161/)
- [Orion: Characterizing Apple's ANE — arxiv 2603.06728](https://arxiv.org/html/2603.06728v1)
- [Inside the M4 Neural Engine — Maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine)
- [Wikipedia A18/M4 chip specs](https://en.wikipedia.org)
