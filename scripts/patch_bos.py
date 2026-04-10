"""
Patch existing corpus .bin files to insert <|bos|> at the start of each document.

Documents in the flat stream are separated by <|eos|> tokens. This script inserts
a <|bos|> token at position 0 and immediately after every <|eos|>.

Run once after corpus was built without BOS tokens:
    python scripts/patch_bos.py
"""

import sys
from pathlib import Path

import numpy as np
import tok as tok_mod

DATA_DIR = Path("data/tokenized")
FILES = ["corpus_train.bin", "corpus_val.bin"]


def patch(path: Path, bos_id: int, eos_id: int) -> None:
    print(f"{path.name}: loading ...", flush=True)
    data = np.fromfile(path, dtype=np.uint16)
    n_before = len(data)

    eos_pos = np.where(data == eos_id)[0]
    # Insert BOS at position 0 and after each EOS (skip trailing EOS at end of file)
    insert_at = np.concatenate([[0], eos_pos[eos_pos < len(data) - 1] + 1])
    patched = np.insert(data, insert_at, np.uint16(bos_id))

    n_docs = len(insert_at)
    print(f"  {n_before:,} → {len(patched):,} tokens (+{n_docs:,} BOS, {n_docs:,} docs)")
    patched.tofile(path)
    print(f"  written.")


def main():
    enc = tok_mod.load()
    bos_id = enc.token_to_id("<|bos|>")
    eos_id = enc.token_to_id("<|eos|>")
    assert bos_id is not None and eos_id is not None

    for name in FILES:
        path = DATA_DIR / name
        if not path.exists():
            print(f"skipping {name} (not found)")
            continue
        patch(path, bos_id, eos_id)


if __name__ == "__main__":
    main()
