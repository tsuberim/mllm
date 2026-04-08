# CoreML Export from MLX/PyTorch: Workflow, Limitations, Quantization

## Sources
- Apple CoreML Llama 3.1: https://machinelearning.apple.com/research/core-ml-on-device-llama
- HuggingFace Mistral CoreML blog: https://huggingface.co/blog/mistral-coreml
- CoreML quantization docs: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-algos.html
- CoreML quantization overview: https://apple.github.io/coremltools/docs-guides/source/opt-quantization-overview.html
- ExecuTorch CoreML backend: https://docs.pytorch.org/executorch/0.7/backends-coreml.html
- smpanaro CoreML LLM CLI: https://github.com/smpanaro/coreml-llm-cli

---

## CoreML Export Workflow (as of 2024)

### Step 1: PyTorch model → traced TorchScript
```python
traced_model = torch.jit.trace(model, example_inputs)
# NOTE: torch.export is preferred but still beta as of 2024
```

### Step 2: Convert to CoreML via coremltools
```python
import coremltools as ct
mlmodel = ct.convert(
    traced_model,
    inputs=[...],
    states=[...],  # for KV cache (iOS 18+ / macOS 15+ only)
    outputs=[...],
    minimum_deployment_target=ct.target.iOS18,
    skip_model_load=True,
)
```

### Step 3: Apply quantization
```python
op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
    mode="linear_symmetric",
    dtype="int4",         # or "int8"
    granularity="per_block",
    block_size=32,        # smaller = higher quality, larger = more compressed
)
config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
mlmodel_int4 = ct.optimize.coreml.linear_quantize_weights(mlmodel_fp16, config=config)
mlmodel_int4.save("model.mlpackage")
```

---

## KV Cache: The Critical Innovation (iOS 18 / macOS 15)

**Without state**: KV cache passed as model input/output tensors → expensive data copies between inference calls → ~1.25 t/s

**With state** (new in iOS 18 / macOS 15 Sequoia):
- KV cache stored as internal model state on GPU
- Updated in-place without CPU-GPU data copies
- **13x faster** than pass-as-I/O approach
- Llama 3.1 8B: 0.19 t/s → 33.67 t/s with KV cache state + int4 quant

**Implementation**:
```python
states = [
    ct.StateType(
        wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
        name="keyCache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(shape=kv_cache_shape, dtype=np.float16),
        name="valueCache",
    ),
]
```

KV cache pre-allocated to maximum context length. Uses `RangeDim` for variable-length input.

---

## Performance Numbers (Llama 3.1 8B)

| Configuration | TTFT | Throughput |
|---------------|------|------------|
| Baseline fp16 (no KV state) | 5,374ms | 0.19 t/s |
| KV cache as I/O | ~5s | 1.25 t/s |
| KV cache as State (macOS 15) | — | 16.26 t/s |
| KV state + int4 quantization | 51.91ms | **33.67 t/s** |

At 2048 context, performance degrades when KV cache exceeds GPU cache bounds → increased cache miss frequency. Smaller context windows = higher throughput.

---

## Attention Mechanism Handling

Standard HuggingFace attention implementations don't convert cleanly to CoreML. Required modifications:

1. **In-place cache update**: Replace concatenation with in-place write to state buffer
2. **4D tensor layout**: Attention requires `(B, C, 1, S)` shape — reshape before/after QKV projections
3. **Fused SDPA**: macOS 15+ maps `scaled_dot_product_attention` to single fused Metal kernel
4. **Causal mask externalized**: Pass causal mask as model input (not computed internally) to allow flexible masking

**Code change required**:
```python
# Instead of: k_cache = torch.cat([past_k, k], dim=2)
# Use: k_state[:, :, past_len:past_len+1, :] = k  # in-place update
```

Each architecture needs custom attention implementation for CoreML — not automatic.

---

## Quantization Hardware Support

| Hardware | Weights | Activations | Notes |
|----------|---------|-------------|-------|
| M1/M2/M3, A15/A16 | int8 on ANE | int8 on ANE | ANE GPU only |
| M4 / A17 Pro | int8 on ANE | int8 on ANE | Hardware W8A8 support |
| GPU (Metal) | int4, int8 | fp16 | W4A16 primary LLM path |

**Key insight**: The Neural Engine (ANE) is fastest for small models but requires specific operator support. For 117M decoder model, GPU via Metal is likely faster than ANE due to autoregressive nature and dynamic KV cache.

**int4 block_size=32 vs 64**: CoreML uses 32 (vs MLX default 64). Smaller block = higher quality per parameter, larger blocks = more compression. Apple's choice of 32 suggests quality sensitivity.

---

## Known Limitations and Gotchas

### Operations that don't export cleanly
- `einsum`: only partially supported — not all possible equations translate to MIL (Model Intermediate Language)
- `scaled_dot_product_attention` (PyTorch 2): complex op, requires custom handling
- `torch.export`: still beta in 2024, `torch.jit.trace` is stable path
- In-place operations: need careful handling of traced computation graph

### Architecture-specific issues
- Tokenization must happen outside the model (Python/Swift preprocessing)
- Variable sequence length requires `RangeDim` specification
- Batched inference: all sequences in batch must be same length (padding + mask workaround)

### iOS vs macOS differences
- Stateful KV cache: **requires iOS 18 / macOS 15** — no backward compatibility
- Older devices: must fall back to KV cache as I/O (13x slower)
- iPhone Neural Engine: requires additional optimization beyond Mac conversion

### iPhone-specific challenges (from HuggingFace blog)
- Neural Engine requires tensor layout (B, C, 1, S) — not standard transformer layout
- Fused SDPA kernel optimized for GPU, not ANE
- Separate ANE-optimized implementation needed (based on Apple 2022 research on Neural Engine transformers)
- Even 7B models won't fit on iPhone (3.8GB after int4); sub-3B models are candidates

---

## Export Path for mllm

### Mac target (priority)
1. Convert PyTorch model → TorchScript via `torch.jit.trace`
2. Modify attention layer for in-place KV state updates
3. Convert with `ct.convert()`, target macOS 15+
4. Apply int4 block-wise quantization (block_size=32)
5. Use fused SDPA (automatic on macOS 15)
6. Expected: >100 t/s for 117M model (much smaller than Llama 8B's 33 t/s)

### iPhone target (planned)
1. Start with smaller block_size (32) or int8 (safer for ANE)
2. Target iOS 18 minimum (stateful KV cache)
3. Investigate ANE-specific attention layout (Apple 2022 Neural Engine paper)
4. 117M model at int4: ~70MB — well within iPhone memory budget
5. Expected: 50-150 t/s on iPhone 16 Pro (extrapolating from Apple benchmarks)

### Key open question
Does MLX's weight format (group_size=64) convert cleanly to CoreML's weight format (block_size=32)? Likely need to re-quantize at CoreML export time, meaning the MLX int4 weights are intermediate format, not directly reusable.

---

## Alternative: ExecuTorch + CoreML Backend

ExecuTorch provides a structured PyTorch → CoreML pipeline:
- `torch.export()` → ExecuTorch → CoreML delegate
- More principled op coverage than `torch.jit.trace`
- Still experimental but maturing rapidly
- May handle attention more gracefully through explicit op decomposition

Worth monitoring for mllm's CoreML export path.
