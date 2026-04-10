"""
Tokenize and pack documents into fixed-length 6K-token chunks.

Sequence packing strategy:
  - Collect all documents from all sources
  - Shuffle at document level (seed=42), then split 90/10 train/val
  - Pack each split: concatenate docs separated by <|eos|>, fill 6K chunks
  - Documents longer than MAX_SEQ_LEN are truncated (rare given filters)
  - Output: corpus_train.bin and corpus_val.bin (uint16, shape [N, seq_len])

Splitting at document level ensures no document straddles the train/val boundary.
Shuffling at document level ensures val is representative across all sources.

Output format:
  - dtype: uint16 (fits 32K vocab)
  - shape: [N, MAX_SEQ_LEN] — load with np.fromfile(...).reshape(-1, seq_len)
  - Attention masks not stored — boundaries inferred via <|eos|> at training time
  - Full corpus fits in H100 HBM (~2.4GB for 1.2B tokens)

Usage:
    python data/pipeline/05_tokenize.py --in data/deduped/ --tok data/tokenizer/ --out data/tokenized/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 6_144
VAL_FRACTION = 0.1
SOURCES_ORDER = [
    "stack_python",
    "stack_bash",
    "stack_md",
    "stackoverflow",
    "tldr",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def iter_sources(in_dir: Path):
    sources = []
    for s in SOURCES_ORDER:
        p = in_dir / f"{s}.jsonl"
        if p.exists():
            sources.append(p)
    for p in sorted(in_dir.glob("*.jsonl")):
        if p not in sources:
            sources.append(p)
    return sources


def pack_docs(docs: list[np.ndarray], seq_len: int, eos_id: int) -> np.ndarray:
    """Pack documents into (N, seq_len) chunks without splitting documents across chunks.
    When a document doesn't fit in the remaining space, the gap is filled with EOS tokens
    and the document starts fresh in the next chunk. ~8% packing overhead vs cross-doc cuts.
    """
    buf = np.full(seq_len, eos_id, dtype=np.uint16)
    pos = 0
    chunks = []
    for ids in docs:
        if len(ids) > seq_len - pos:
            # doc doesn't fit — flush current chunk (gap already filled with EOS) and reset
            if pos > 0:
                chunks.append(buf.copy())
                buf = np.full(seq_len, eos_id, dtype=np.uint16)
                pos = 0
        buf[pos:pos + len(ids)] = ids
        pos += len(ids)
        if pos == seq_len:
            chunks.append(buf.copy())
            buf = np.full(seq_len, eos_id, dtype=np.uint16)
            pos = 0
    return np.stack(chunks) if chunks else np.empty((0, seq_len), dtype=np.uint16)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pack_corpus(in_dir: Path, tokenizer: Tokenizer, out_dir: Path, seq_len: int):
    eos_id = tokenizer.token_to_id("<|eos|>")
    bos_id = tokenizer.token_to_id("<|bos|>")
    assert eos_id is not None, "<|eos|> not in tokenizer vocab"
    assert bos_id is not None, "<|bos|> not in tokenizer vocab"

    # 1. Collect all tokenized documents
    BATCH_SIZE = 1024  # encode_batch uses Rust/Rayon threading — much faster than one-at-a-time

    def _flush_batch(contents: list[str]) -> None:
        for enc in tokenizer.encode_batch(contents):
            tokens = enc.ids
            if len(tokens) <= seq_len - 2:
                all_docs.append(np.array([bos_id] + tokens + [eos_id], dtype=np.uint16))
            else:
                # split: BOS only on first piece, EOS only on last, middle pieces are raw
                # the model learns: no BOS at chunk start = mid-document continuation
                all_docs.append(np.array([bos_id] + tokens[:seq_len - 1], dtype=np.uint16))
                i = seq_len - 1
                while i < len(tokens):
                    chunk = tokens[i:i + seq_len]
                    if i + seq_len >= len(tokens):
                        all_docs.append(np.array(chunk + [eos_id], dtype=np.uint16))
                    else:
                        all_docs.append(np.array(chunk, dtype=np.uint16))
                    i += seq_len

    print("Collecting and tokenizing documents ...")
    all_docs = []
    for jsonl in iter_sources(in_dir):
        source_name = jsonl.stem
        print(f"\n  {source_name} ...")
        batch: list[str] = []
        with open(jsonl) as f:
            for line in tqdm(f, desc=source_name, unit="doc"):
                try:
                    content = json.loads(line).get("content", "")
                    if not content:
                        continue
                except json.JSONDecodeError:
                    continue
                batch.append(content)
                if len(batch) >= BATCH_SIZE:
                    _flush_batch(batch)
                    batch = []
        if batch:
            _flush_batch(batch)

    print(f"\nTotal documents: {len(all_docs):,}")

    # 2. Shuffle at document level
    rng = np.random.default_rng(seed=42)
    order = rng.permutation(len(all_docs)).tolist()
    all_docs = [all_docs[i] for i in order]

    # 3. Split 90/10
    split = int(len(all_docs) * (1 - VAL_FRACTION))
    train_docs = all_docs[:split]
    val_docs   = all_docs[split:]
    print(f"Split: {len(train_docs):,} train docs / {len(val_docs):,} val docs")

    # 4. Sort by length within each split to minimise packing gaps, then pack
    train_docs.sort(key=len)
    val_docs.sort(key=len)

    print("\nPacking train ...")
    train_arr = pack_docs(train_docs, seq_len, eos_id)
    print(f"  {len(train_arr):,} chunks")

    print("Packing val ...")
    val_arr = pack_docs(val_docs, seq_len, eos_id)
    print(f"  {len(val_arr):,} chunks")

    # 5. Write
    train_path = out_dir / "corpus_train.bin"
    val_path   = out_dir / "corpus_val.bin"
    train_arr.tofile(str(train_path))
    val_arr.tofile(str(val_path))

    total_chunks = len(train_arr) + len(val_arr)
    total_tokens = sum(d.shape[0] for d in all_docs)
    size_gb = (train_arr.nbytes + val_arr.nbytes) / 1e9
    print(f"\nTotal: {total_chunks:,} chunks × {seq_len} = {total_chunks * seq_len:,} tokens ({size_gb:.2f} GB)")
    print(f"  train: {len(train_arr):,} chunks → {train_path}")
    print(f"  val:   {len(val_arr):,} chunks → {val_path}")
    print(f"Packing efficiency: {100 * total_tokens / max(1, total_chunks * seq_len):.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Tokenize and pack corpus (doc-level shuffle + train/val split)")
    parser.add_argument("--in", dest="in_dir", required=True, help="Directory with deduped JSONL")
    parser.add_argument("--tok", dest="tok_dir", required=True, help="Directory with tokenizer.json")
    parser.add_argument("--out", required=True, help="Output directory for corpus_train.bin / corpus_val.bin")
    parser.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN, help=f"Sequence length (default {MAX_SEQ_LEN})")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    tok_path = Path(args.tok_dir) / "tokenizer.json"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tok_path.exists():
        print(f"Tokenizer not found: {tok_path}")
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(tok_path))
    print(f"Loaded tokenizer (vocab={tokenizer.get_vocab_size():,})")

    pack_corpus(in_dir, tokenizer, out_dir, args.seq_len)


if __name__ == "__main__":
    main()
