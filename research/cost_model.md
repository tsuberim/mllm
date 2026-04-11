# Full 3B Run — Cost Model

Researched April 2026. CPU (Modal): $0.0000131/core-sec. Storage: free on Modal volume.

## Pre-training Platform Comparison

Training is resumable (checkpoint every 1K steps) so preemption/spot interruption costs at most ~1K steps of wasted compute — negligible. Spot pricing is therefore viable.

MFU assumption: **35% effective** for providers without guaranteed NVLink topology (Modal, Vast.ai); **38% effective** for providers with confirmed 8-GPU NVLink/NVSwitch HGX nodes (RunPod SXM pod, CoreWeave, Spheron SXM5). See §4 for how these translate to GPU-hours.

| Provider | $/GPU-hr | GPU-hours | Pre-train cost | Notes |
|---|---|---|---|---|
| Vast.ai (spot/market) | ~$1.55 | 1,533 | **$2,377** | Marketplace; variable quality; topology not guaranteed |
| RunPod 8×H100 SXM | $2.69 | 1,520 | **$4,089** | $21.52/hr per 8-GPU node; proper NVLink ✓ |
| Lambda Labs | $2.89 | 1,533 | **$4,430** | Verify 8-GPU NVLink availability |
| Hyperbolic (SXM) | $3.20 | 1,520 | **$4,864** | SXM confirmed |
| Modal | $3.95 | 1,533 | **$6,055** | 8-GPU NVLink topology unconfirmed |
| CoreWeave 8×SXM5 | $6.155 | 1,520 | **$9,356** | Dedicated HGX + InfiniBand; most expensive |
| AWS (p4de.24xl) | $6.88 | 1,533 | **$10,547** | |
| Azure (ND H100 v5) | $12.29 | 1,533 | **$18,840** | |

Non-pre-training cost (Modal, fixed): **~$1,820**
(data pipeline ~$20, repo scanning ~$250, trace gen ~$950, post-training ~$150, experiments ~$450)

| Provider | Pre-train | Total project |
|---|---|---|
| Vast.ai | $2,377 | **~$4,200** |
| RunPod 8×SXM | $4,089 | **~$5,900** |
| Lambda Labs | $4,430 | **~$6,250** |
| Hyperbolic | $4,864 | **~$6,700** |
| Modal | $6,055 | **~$7,900** |
| CoreWeave | $9,356 | **~$11,200** |
| AWS | $10,547 | **~$12,400** |
| Azure | $18,840 | **~$20,700** |

**Recommendation:** RunPod's 8×H100 SXM node ($21.52/hr) is the sweet spot — confirmed NVLink topology at $2.69/GPU-hr, 32% cheaper than Modal and 56% cheaper than CoreWeave.

---

## Assumptions & Methodology

**Pre-training throughput baseline:**
Actual model config: N = 3.17B params, block_size = 4096, n_embd = 3072, n_layer = 20.
FLOPs/token = 6N = 6 × 3.17B = 19.02×10⁹.
H100 SXM5 BF16 dense: 989 TFLOPS.
At 40% MFU (realistic for 8-GPU BF16 DDP): 989 × 0.40 = 395.6 TFLOPS/GPU → **~21k tok/s/GPU**.
SmolLM 3B achieved 41k tok/s at 45% MFU but with TorchTitan + FP8 — not our stack.

Total compute: C = 6 × 3.17B × 100B = 1.902×10²¹ FLOPs.
At 40% MFU, 1× H100: 1.902×10²¹ / (395.6×10¹² × 3600) ≈ **1,334 GPU-hours**.
With 8× H100 + 10% DDP overhead: **~1,480 GPU-hours**, ~185h wall clock.

**Trace gen throughput:**
Qwen2.5-Coder-32B at BF16 requires 2× H100 (tensor parallel).
vLLM decode throughput: ~3,000 tok/s (output tokens only). Prefill: ~30,000 tok/s.
Assume 40% of attempts produce a usable trace → 500K attempts to get 200K successful.
Avg 1,500 input + 1,000 output tokens per attempt.

**CPU pricing:** $0.0000131/core/sec (non-preemptible). 2 cores per worker typical.

---

## Cost Breakdown by Phase

### 1. Data pipeline — full corpus (Milestone 8) ~$15–25

Scripts: `00_download.py` → `05_tokenize.py` (no filter or dedup step)

| Step | What | Wall clock | Cores | Cost |
|---|---|---|---|---|
| Download | Stream 7 HF sources (Stack Python/Bash/Markdown, SO, commits, issues, tldr) | ~2h | 8 | ~$3 |
| Tokenize + pack | Phase 1: parallel BPE tokenize per shard → `.tok.bin`/`.idx.bin`. Phase 2: serial shuffle + 90/10 split + pack → `corpus_train.bin` + `corpus_val.bin` | ~3h | 32 | ~$9 |
| Upload to HF | Push ~200GB corpus bins | ~2h | 4 | ~$2 |

**Subtotal: ~$15–25**

Note: Current 1.19B token corpus is already done. Full run is ~85× the current scale.

---

### 2. Repo scanning (Milestone 3a) ~$250

Target: 50K candidates → 20K passing repos.
Each scan: clone + service-dep filter + pytest → ~3 min avg, 2 cores.

| Step | Repos | CPU-hours | Cost |
|---|---|---|---|
| GitHub search API → candidates | 50K | negligible | ~$0 |
| Modal scan (clone + pytest) | 50K | ~5,000 core-hrs | ~$236 |

Wall clock with 500 concurrent containers: ~5 hours.

**Subtotal: ~$200–300**

Risk: observed passing rate is ~10% (227/2185), not the targeted 40%. At 10%, hitting 20K passing repos requires scanning ~200K candidates → cost ×4 (~$950). Worth validating pass rate before committing.

---

### 3. Trace generation (Milestone 3b)

Volume: 500K attempts × (1,500 input + 1,000 output tokens) = 750M input + 500M output tokens.

**Two cost centres: inference + Docker sandbox validation (independent of each other).**

#### Inference options

| Approach | $/GPU-hr or /M tok | Inference cost | Notes |
|---|---|---|---|
| Self-hosted BF16 2×H100 (Modal) | $3.95/GPU-hr | $419 | 106 GPU-hrs; decode-only timing |
| Self-hosted BF16 2×H100 (RunPod) | $2.69/GPU-hr | $285 | Same throughput, cheaper platform |
| Self-hosted INT4/AWQ 1×H100 (RunPod) | $2.69/GPU-hr | $143 | 32B INT4 fits in 80GB; ~50% fewer GPU-hrs |
| API: DeepInfra Qwen2.5-Coder-32B | $0.20/M tokens | $250 | 1.25B tokens × $0.20/M |
| **API: DeepInfra Qwen3-32B** | **$0.08/M tokens** | **$100** | 1.25B tokens × $0.08/M; newer model |
| API: Hyperbolic Qwen2.5-Coder-32B | $0.12/$0.30 in/out | $240 | 750M×$0.12 + 500M×$0.30 |

#### Docker sandbox validation

| Setup | Time/attempt | CPU cost |
|---|---|---|
| Cold containers (current) | 30s | $393 |
| **Pre-warmed containers** | **~5s** | **$66** |

Pre-warming is already supported by Modal sandbox. Implement before the full run.

#### Total cost combinations

| Inference | Sandbox | **Total** |
|---|---|---|
| Modal BF16 2×H100 (baseline) | Cold | $812 |
| RunPod INT4 1×H100 | Cold | $536 |
| DeepInfra Qwen3-32B API | Cold | $493 |
| **DeepInfra Qwen3-32B API** | **Pre-warmed** | **$166** |

**Recommendation: DeepInfra Qwen3-32B API + pre-warmed sandbox → ~$166 (5× cheaper than baseline).**

Caveats:
- Qwen3-32B is general-purpose; Qwen2.5-Coder-32B is code-specialized. If pass rate drops from 40% → 25%, attempts grow to 800K and API cost rises to ~$160 — still fine.
- Check DeepInfra rate limits before assuming 1.25B tokens can be generated without multi-day queuing.

---

### 4. Pre-training 3B (Milestone 9) ~$5,500–9,500

3B model (N=3.17B, block_size=4096), 100B tokens (80B general + 20B protocol-heavy warmup).
Total compute: C = 6 × 3.17B × 100B = 1.902×10²¹ FLOPs.

At 38% effective MFU (8-GPU DDP with NVLink, InfiniBand):
- 8× GPUs: ~146k tok/s, ~190h wall clock, ~1,520 GPU-hours

At 35% effective MFU (8-GPU DDP, degraded interconnect):
- 8× GPUs: ~145k effective tok/s... actually MFU degrades throughput per GPU, not scaling
- Per GPU: 989 × 0.35 = 346 TFLOPS → ~18.2k tok/s; 8× with 10% DDP overhead: ~131k tok/s
- ~211h wall clock, ~1,688 GPU-hours

See platform comparison table above. At 35% eff. MFU (no guaranteed NVLink) → 1,533 GPU-hours. At 38% (confirmed NVLink HGX) → 1,520 GPU-hours.

**Decision:** run a 100-step throughput benchmark before committing. If tok/s is within 10% of the NVLink baseline, use the cheaper platform.

### 4b. Pre-training 7B (Milestone 9b) ~$9,500–21,800

7B model (N=7.19B, block_size=4096), 100B tokens.
Total compute: C = 6 × 7.19B × 100B = 4.314×10²¹ FLOPs.

At 38% effective MFU (8-GPU DDP with NVLink):
- 8× GPUs: ~142k tok/s, ~443h wall clock, ~3,543 GPU-hours

| Provider | $/GPU-hr | GPU-hours | Cost |
|---|---|---|---|
| RunPod 8×H100 SXM | $2.69 | 3,543 | **$9,530** |
| CoreWeave 8×SXM5 | $6.155 | 3,543 | **$21,806** |

---

### 5. Post-training (Milestone 10) ~$100–500

From `research/post_training/README.md`:

| Phase | H100-hours | Cost |
|---|---|---|
| SFT (traces + instruction mix) | 2–6 | $8–24 |
| RL (GRPO, G=4) | 8–48 | $32–190 |
| Thinking fusion SFT | 4–8 | $16–32 |
| **Total** | **14–62** | **$56–246** |

RL dominates and has highest uncertainty. Use spot H100s; checkpoint every 50 steps.

---

### 6. Experiments & ablations ~$300

| Run | Purpose | H100-hours | Cost |
|---|---|---|---|
| Loss convergence (330M, 100B tokens) | Validate stability before 3B | ~100 | ~$395 |
| Config sweeps (arch, LR, batch) | 3–5 short runs on 330M | ~50 | ~$200 |

**Subtotal: ~$300–600**

Run the convergence run first. If loss doesn't converge cleanly, fix before spending on 3B.

---

### 7. Storage ~$0

Modal volume storage: **no per-GB charge** (as of April 2026; 10TB cap).
- Repo zips: 50K repos × ~10MB = ~500GB ✓
- Tokenized corpus (100B tokens × 2 bytes): ~200GB ✓
- Checkpoints: ~50GB ✓

HuggingFace: free for public datasets/models.

---

## Grand Total

| Phase | Provider | Cost |
|---|---|---|
| Data pipeline | Modal CPU | $20 |
| Repo scanning | Modal CPU | $250 |
| Trace generation | DeepInfra API + pre-warmed Modal sandbox | $166 |
| Pre-training 3B (8× H100 SXM) | RunPod on-demand | $4,089 |
| Pre-training 7B (8× H100 SXM) | RunPod on-demand | $9,530 |
| Post-training (SFT + RL + thinking) | Modal GPU | ~$150 |
| Experiments & ablations | Modal GPU | ~$450 |
| Storage | Modal volume | $0 |
| **Total (3B only)** | | **~$5,100** |
| **Total (3B + 7B)** | | **~$14,700** |

**Unoptimized baseline (all Modal, cold sandbox, self-hosted Qwen):** ~$7,700.

Optimizations that drove the reduction:
- RunPod over Modal for pre-training: **−$2,000**
- DeepInfra API over self-hosted Qwen: **−$320**
- Pre-warmed sandbox containers: **−$327**

---

## Key Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Low trace success rate (<20%) | Doubles trace gen cost | Pilot with 1K repos; tune sandbox before full run |
| Poor MFU on pre-training (<30%) | Adds $1,000+ | Profile early with 1-step timing; optimize data loader |
| RL instability requiring restarts | +$200–500 | Checkpoint every 50 steps; curriculum to reduce sparse reward |
| Data pipeline I/O bottlenecks (HF download) | Delays, minor cost | Use streaming + sharded download |

---

## Grant Coverage

If AI Grant ($50K) or Amazon Research Award ($120K combined) comes through, the full project is covered with headroom for 7B experiments.
At ~$11.5K total cost, AI Grant ($50K) or Amazon Research Award ($120K combined cash + AWS credits) covers the full project with significant headroom for 7B experiments. CoreWeave startup credits alone would eliminate the dominant cost center entirely.

---

## Sources

- [Modal Pricing](https://modal.com/pricing)
- [H100 Cloud Pricing Comparison](https://getdeploying.com/gpus/nvidia-h100)
- [SmolLM 3B throughput benchmark — Tzafon](https://www.tzafon.ai/blog/breaking-40k-tokens)
- [H100 rental price comparison](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)
- [Lambda H100 pricing](https://www.thundercompute.com/blog/nvidia-h100-pricing)
