"""
Upload tokenizer to HuggingFace as tsuberim/merlin-tokenizer-v0.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not set")

REPO_ID = "tsuberim/merlin-tokenizer-v0"
TOK_DIR = Path("data/tokenizer")

README = """\
---
license: apache-2.0
tags:
  - merlin
  - tokenizer
---

# merlin-tokenizer-v0

BPE tokenizer for [Merlin](https://github.com/tsuberim/mllm), trained on Python, Bash, Markdown,
Stack Exchange Q&A, GitHub commits/issues, and tldr-pages.

## Vocab

- 32,016 tokens total: 32,000 BPE + 16 special tokens
- Special tokens:
  - IDs 0–13: legacy slots (unused)
  - ID 32000: `<|bos|>`
  - ID 32001: `<|eos|>`
  - ID 32002–32013: agent protocol tokens (`<|tool_call|>`, `<|/tool_call|>`, etc.)
  - ID 32014: `<|done|>`
  - ID 32015: `<|pad|>`

## Usage

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer.json")
ids = tok.encode("def hello(): pass").ids
```

> **Note:** Will be retrained after agentic trace generation. This is v0 — trained on pretraining corpus only.
"""

api = HfApi(token=token)

print(f"Creating/verifying repo {REPO_ID} ...")
create_repo(REPO_ID, repo_type="model", exist_ok=True, token=token)

print("Uploading README ...")
api.upload_file(
    path_or_fileobj=README.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="v0: tokenizer trained on pretraining corpus",
)

tok_file = TOK_DIR / "tokenizer.json"
print(f"Uploading {tok_file} ({tok_file.stat().st_size/1e6:.1f}MB) ...", end=" ", flush=True)
api.upload_file(
    path_or_fileobj=str(tok_file),
    path_in_repo="tokenizer.json",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="v0: tokenizer.json",
)
print("done")

print(f"\nDone. https://huggingface.co/tsuberim/{REPO_ID.split('/')[1]}")
