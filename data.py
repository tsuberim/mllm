"""Tokenize TinyStories and write train/val binary files."""
import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path

def main():
    enc = tiktoken.get_encoding("gpt2")
    ds  = load_dataset("roneneldan/TinyStories")

    for split in ("train", "validation"):
        out = Path(f"data_{split}.bin")
        if out.exists():
            print(f"{out} already exists, skipping")
            continue

        tokens = []
        for ex in ds[split]:
            tokens.extend(enc.encode_ordinary(ex["text"]))
            tokens.append(enc.eot_token)

        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(out)
        print(f"{split}: {len(arr):,} tokens → {out}")

if __name__ == "__main__":
    main()
