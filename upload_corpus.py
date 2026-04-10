"""
Upload tokenized corpus shards to HuggingFace as tsuberim/merlin-corpus-v0.
Version: 0.0.1
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not set")

REPO_ID = "tsuberim/merlin-corpus-v0"
TOKENIZED = Path("data/tokenized")

DATASET_CARD = """\
---
license: other
tags:
  - code
  - pretraining
  - merlin
---

# merlin-corpus-v0 — v0.0.1

Tokenized pretraining corpus for [Merlin](https://github.com/tsuberim/mllm), a 3B LLM for agentic coding on Apple Silicon.

> **Note:** v0 does not yet include agentic traces. Tokenizer and corpus will be retrained after trace generation.

## Contents

- `corpus_train.bin` + `corpus_val.bin`, uint16, shape `[N, 6144]`
- 90/10 train/val split at document boundaries (seed=42 shuffle)
- ~1.19B tokens total, 100% packing efficiency
- Fits entirely in H100 HBM (~2.4GB)
- Tokenizer: [tsuberim/merlin-tokenizer-v0](https://huggingface.co/tsuberim/merlin-tokenizer-v0)

## Sources

| Source | Records | Notes |
|---|---|---|
| The Stack — Python | ~730K | AST-parsed, has def/class, no generated files |
| The Stack — Bash | ~214K | ≥2 shell commands |
| The Stack — Markdown | ~63K | English, has backtick, no HTML blobs |
| Stack Exchange Q&A | ~21K | Python/bash/linux/git/docker topic filter |
| GitHub commits | ~84K | Non-trivial commit messages |
| GitHub issues | ~71K | Topic-filtered, no bot posts |
| tldr-pages | ~7K | Clean command references |

## Format

```python
import numpy as np
# Load into memory (~2.4GB total, fits in H100 HBM)
train = np.fromfile("corpus_train.bin", dtype=np.uint16).reshape(-1, 6144)  # (N, 6144)
val   = np.fromfile("corpus_val.bin",   dtype=np.uint16).reshape(-1, 6144)
```
"""

api = HfApi(token=token)

print(f"Creating/verifying repo {REPO_ID} ...")
create_repo(REPO_ID, repo_type="dataset", exist_ok=True, token=token)

print("Uploading README ...")
api.upload_file(
    path_or_fileobj=DATASET_CARD.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message="v0.0.1: updated corpus with quality filters",
)

for fname, repo_path, desc in [
    ("corpus_train.bin", "data/corpus_train.bin", "train split (173,911 chunks × 6144)"),
    ("corpus_val.bin",   "data/corpus_val.bin",   "val split (19,340 chunks × 6144)"),
]:
    f = TOKENIZED / fname
    size_gb = f.stat().st_size / 1e9
    print(f"Uploading {fname} ({size_gb:.2f} GB) ...")
    api.upload_file(
        path_or_fileobj=str(f),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message=f"v0.0.1: {fname} — {desc}",
    )
    print(f"  done.")

print(f"\nDone. https://huggingface.co/datasets/{REPO_ID}")
