# CoreWeave Startup Accelerator

Apply at: https://www.coreweave.com/startup-accelerator (or contact sales@coreweave.com)
Value: GPU credits + discounts (amount not published)
Rolling, no hard deadline.

Note: CoreWeave is the most expensive provider at $6.155/GPU-hr. Credits here would be most valuable since they'd otherwise be the last choice. Even $5K in credits covers 812 GPU-hours — meaningful for experiments.

---

## Application / inquiry text

**Subject:** Startup accelerator inquiry — open-source 3B LLM training

Hi CoreWeave team,

I'm building Merlin, an open-source 3B language model for agentic coding — pre-trained from scratch with RL post-training, targeting local inference on Apple Silicon. The project is fully open-source under MIT.

Total training compute: ~1,520 GPU-hours on 8×H100 SXM for the full 3B pre-training run (100B tokens). We're also running ongoing experiments on 330M configurations to validate convergence before scaling.

I'd like to understand what's available through the CoreWeave Startup Accelerator for independent OSS ML researchers. Our pre-training is the dominant cost center (~$4K at commercial rates), and CoreWeave's HGX topology + InfiniBand would be the highest-quality environment for it.

**Project links:**
- HuggingFace: https://huggingface.co/tsuberim/merlin-tokenizer-v0
- Corpus (1.19B token v0): https://huggingface.co/datasets/tsuberim/merlin-corpus-v0
- GitHub: (public before submission)

Happy to share a technical writeup or jump on a call.

Thanks,
Matan
