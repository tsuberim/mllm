"""
Tokenize and pack documents into fixed-length 6K-token chunks.

Two-phase design that scales to 100B+ tokens without loading all docs into RAM:

  Phase 1 (parallel, per-shard):
    Each shard file → tokenize with multiprocessing → write:
      <out>/.tok/<source>/<shard>.tok.bin  — flat uint16 token stream
      <out>/.tok/<source>/<shard>.idx.bin  — uint32 doc lengths
    Resumable: completed shards tracked via .done sentinel files.

  Phase 2 (serial, global):
    Load all .idx files (lengths only, ~2.4 GB for 200M docs at 100B tokens).
    Shuffle at doc level (seed=42), split 90/10 train/val.
    Stream tokens from mmap'd .tok files, pack into corpus_train.bin / corpus_val.bin.
    Working RAM: index + one packing buffer. Never loads full corpus into RAM.

Output format:
  - dtype: uint16, shape [N, MAX_SEQ_LEN]
  - load: np.fromfile(...).reshape(-1, seq_len)

Usage:
    # pipeline.py output (data/processed/<source>/*.jsonl.gz):
    python data/pipeline/05_tokenize.py --in data/processed/ --tok data/tokenizer/ --out data/tokenized/

    # Legacy flat JSONL (old pipeline output):
    python data/pipeline/05_tokenize.py --in data/deduped/ --tok data/tokenizer/ --out data/tokenized/ --flat
"""

import argparse
import gzip
import json
import multiprocessing as mp
import os
import struct
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_SEQ_LEN  = 6_144
VAL_FRACTION = 0.1
BATCH_SIZE   = 1024   # encode_batch uses Rust/Rayon threading

SOURCES_ORDER = [
    "stack_python", "stack_bash", "stack_md",
    "stackoverflow", "github_commits", "github_issues", "tldr",
]


# ---------------------------------------------------------------------------
# Phase 1: tokenize one shard → .tok.bin + .idx.bin
# ---------------------------------------------------------------------------

def _tokenize_shard(args) -> tuple[int, int]:
    """
    Tokenize one shard file.
    Writes:
      tok_path: flat uint16 token stream (all docs concatenated)
      idx_path: uint32 doc lengths (one per doc)
    Returns (n_docs, n_tokens).
    Skips if both output files already exist (resumability).
    """
    shard_path, tok_path, idx_path, tokenizer_path, seq_len, bos_id, eos_id = args

    if Path(tok_path).exists() and Path(idx_path).exists():
        # Resume: read existing counts from idx
        lengths = np.fromfile(idx_path, dtype=np.uint32)
        return len(lengths), int(lengths.sum())

    tokenizer = Tokenizer.from_file(tokenizer_path)
    open_fn = gzip.open if str(shard_path).endswith(".gz") else open

    tok_out = open(tok_path, "wb")
    idx_out = open(idx_path, "wb")
    n_docs = n_tokens = 0

    batch: list[str] = []

    def flush(batch: list[str]):
        nonlocal n_docs, n_tokens
        for enc in tokenizer.encode_batch(batch):
            tokens = enc.ids
            if len(tokens) <= seq_len - 2:
                doc = [bos_id] + tokens + [eos_id]
            else:
                # split long doc: BOS on first piece, EOS on last
                doc = [bos_id] + tokens[:seq_len - 1]
                # write first piece
                arr = np.array(doc, dtype=np.uint16)
                tok_out.write(arr.tobytes())
                idx_out.write(struct.pack("<I", len(arr)))
                n_docs += 1
                n_tokens += len(arr)
                i = seq_len - 1
                while i < len(tokens):
                    chunk = tokens[i:i + seq_len]
                    if i + seq_len >= len(tokens):
                        chunk = chunk + [eos_id]
                    arr = np.array(chunk, dtype=np.uint16)
                    tok_out.write(arr.tobytes())
                    idx_out.write(struct.pack("<I", len(arr)))
                    n_docs += 1
                    n_tokens += len(arr)
                    i += seq_len
                continue  # already written above
            arr = np.array(doc, dtype=np.uint16)
            tok_out.write(arr.tobytes())
            idx_out.write(struct.pack("<I", len(arr)))
            n_docs += 1
            n_tokens += len(arr)

    with open_fn(shard_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                record = json.loads(line)
                content = record.get("text") or record.get("content") or ""
                if content:
                    batch.append(content)
            except json.JSONDecodeError:
                continue
            if len(batch) >= BATCH_SIZE:
                flush(batch)
                batch = []
    if batch:
        flush(batch)

    tok_out.close()
    idx_out.close()
    return n_docs, n_tokens


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------

def _iter_shards(in_dir: Path, flat: bool) -> list[tuple[str, list[Path]]]:
    if flat:
        shards_by_source: dict[str, list[Path]] = {}
        for p in sorted(in_dir.glob("*.jsonl")):
            shards_by_source.setdefault(p.stem, []).append(p)
        ordered = []
        for s in SOURCES_ORDER:
            if s in shards_by_source:
                ordered.append((s, shards_by_source.pop(s)))
        for s, paths in sorted(shards_by_source.items()):
            ordered.append((s, paths))
        return ordered
    else:
        ordered = []
        seen = set()
        for s in SOURCES_ORDER:
            d = in_dir / s
            if d.is_dir():
                shards = sorted(d.glob("*.jsonl.gz")) + sorted(d.glob("*.jsonl"))
                if shards:
                    ordered.append((s, shards))
                    seen.add(s)
        for d in sorted(in_dir.iterdir()):
            if d.is_dir() and d.name not in seen:
                shards = sorted(d.glob("*.jsonl.gz")) + sorted(d.glob("*.jsonl"))
                if shards:
                    ordered.append((d.name, shards))
        return ordered


# ---------------------------------------------------------------------------
# Phase 2: global shuffle + pack (streaming, O(index) RAM)
# ---------------------------------------------------------------------------

def _pack_corpus(
    tok_dir: Path,
    all_shards: list[tuple[Path, Path]],  # [(tok_path, idx_path), ...]
    lengths_all: np.ndarray,              # uint32, one per doc
    offsets_all: np.ndarray,              # uint64, byte offset in tok file per doc
    shard_ids: np.ndarray,                # int, which shard each doc belongs to
    out_dir: Path,
    seq_len: int,
    eos_id: int,
    rng_seed: int = 42,
):
    n_docs = len(lengths_all)
    rng = np.random.default_rng(rng_seed)
    order = rng.permutation(n_docs)

    split = int(n_docs * (1 - VAL_FRACTION))
    splits = {"train": order[:split], "val": order[split:]}

    # Open all .tok.bin files as mmaps
    mmaps = []
    for tok_path, _ in all_shards:
        f = open(tok_path, "rb")
        size = os.path.getsize(tok_path)
        if size == 0:
            mmaps.append(None)
        else:
            mmaps.append(np.memmap(tok_path, dtype=np.uint16, mode="r"))
        f.close()

    for split_name, indices in splits.items():
        out_path = out_dir / f"corpus_{split_name}.bin"
        print(f"\nPacking {split_name} ({len(indices):,} docs) → {out_path}")

        # Sort by length to minimise packing gaps
        lens = lengths_all[indices]
        sort_order = np.argsort(lens)
        indices = indices[sort_order]

        buf = np.full(seq_len, eos_id, dtype=np.uint16)
        pos = 0
        n_chunks = 0

        with open(out_path, "wb") as out_f:
            for doc_idx in tqdm(indices, desc=split_name, unit="doc"):
                sid    = shard_ids[doc_idx]
                offset = offsets_all[doc_idx]   # byte offset
                length = int(lengths_all[doc_idx])

                mmap = mmaps[sid]
                if mmap is None:
                    continue
                tok_offset = int(offset) // 2  # bytes → uint16 elements
                doc_tokens = mmap[tok_offset:tok_offset + length]

                if length > seq_len - pos:
                    if pos > 0:
                        out_f.write(buf.tobytes())
                        n_chunks += 1
                        buf = np.full(seq_len, eos_id, dtype=np.uint16)
                        pos = 0
                buf[pos:pos + length] = doc_tokens
                pos += length
                if pos == seq_len:
                    out_f.write(buf.tobytes())
                    n_chunks += 1
                    buf = np.full(seq_len, eos_id, dtype=np.uint16)
                    pos = 0

            if pos > 0:
                out_f.write(buf.tobytes())
                n_chunks += 1

        size_gb = out_path.stat().st_size / 1e9
        print(f"  {n_chunks:,} chunks ({size_gb:.2f} GB)")

    for mmap in mmaps:
        if mmap is not None:
            del mmap


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_phase1(in_dir: Path, tokenizer_path: str, out_dir: Path, seq_len: int,
               workers: int, flat: bool, source: str | None = None) -> int:
    """
    Phase 1 only: tokenize shards → .tok.bin + .idx.bin files under out_dir/.tok/.
    Optionally restricted to a single source (for parallel per-source containers).
    Returns total token count.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|eos|>")
    bos_id = tokenizer.token_to_id("<|bos|>")
    assert eos_id is not None, "<|eos|> not in tokenizer vocab"
    assert bos_id is not None, "<|bos|> not in tokenizer vocab"

    all_sources = _iter_shards(in_dir, flat)
    if source:
        all_sources = [(s, shards) for s, shards in all_sources if s == source]
    if not all_sources:
        print(f"No shards found (source={source})")
        return 0

    tok_dir = out_dir / ".tok"
    tok_dir.mkdir(parents=True, exist_ok=True)

    shard_worker_args = []
    for src, shards in all_sources:
        src_tok_dir = tok_dir / src
        src_tok_dir.mkdir(exist_ok=True)
        for shard in shards:
            stem = Path(shard).stem.replace(".jsonl", "")
            tok_path = src_tok_dir / f"{stem}.tok.bin"
            idx_path = src_tok_dir / f"{stem}.idx.bin"
            shard_worker_args.append((
                str(shard), str(tok_path), str(idx_path),
                tokenizer_path, seq_len, bos_id, eos_id,
            ))

    total_docs = total_tokens = 0
    with mp.Pool(workers) as pool:
        for n_docs, n_tokens in tqdm(
            pool.imap(_tokenize_shard, shard_worker_args),
            total=len(shard_worker_args),
            desc=f"shards({source or 'all'})",
            unit="shard",
        ):
            total_docs += n_docs
            total_tokens += n_tokens

    print(f"Phase 1 done ({source or 'all'}): {total_docs:,} docs, {total_tokens:,} tokens ({total_tokens/1e9:.2f}B)")
    return total_tokens


def run_phase2(out_dir: Path, tokenizer_path: str, seq_len: int) -> None:
    """
    Phase 2 only: discover all .tok.bin/.idx.bin already written, shuffle, stream-pack.
    Run after all Phase 1 containers have finished.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    eos_id = tokenizer.token_to_id("<|eos|>")
    assert eos_id is not None, "<|eos|> not in tokenizer vocab"

    tok_dir = out_dir / ".tok"
    # Discover all shard pairs from disk (stable sort for reproducibility)
    all_shard_pairs: list[tuple[Path, Path]] = []
    for idx_path in sorted(tok_dir.rglob("*.idx.bin")):
        tok_path = idx_path.with_suffix("").with_suffix(".tok.bin")
        all_shard_pairs.append((tok_path, idx_path))

    if not all_shard_pairs:
        print(f"No .idx.bin files found under {tok_dir}")
        sys.exit(1)

    print(f"\nPhase 2: loading indices ({len(all_shard_pairs):,} shards) ...")
    lengths_list  = []
    offsets_list  = []
    shard_id_list = []

    for shard_id, (tok_path, idx_path) in enumerate(tqdm(all_shard_pairs, desc="loading indices")):
        if not idx_path.exists() or idx_path.stat().st_size == 0:
            continue
        lengths = np.fromfile(str(idx_path), dtype=np.uint32)
        if len(lengths) == 0:
            continue
        byte_offsets = np.concatenate([[0], np.cumsum(lengths[:-1].astype(np.uint64) * 2)])
        lengths_list.append(lengths)
        offsets_list.append(byte_offsets)
        shard_id_list.append(np.full(len(lengths), shard_id, dtype=np.int32))

    lengths_all = np.concatenate(lengths_list)
    offsets_all = np.concatenate(offsets_list)
    shard_ids   = np.concatenate(shard_id_list)
    print(f"Index: {len(lengths_all):,} docs, {lengths_all.sum():,} tokens")

    _pack_corpus(tok_dir, all_shard_pairs, lengths_all, offsets_all, shard_ids,
                 out_dir, seq_len, eos_id)

    total_chunks = sum(
        (out_dir / f"corpus_{s}.bin").stat().st_size // (seq_len * 2)
        for s in ("train", "val")
        if (out_dir / f"corpus_{s}.bin").exists()
    )
    eff = 100 * lengths_all.sum() / max(1, total_chunks * seq_len)
    print(f"\nPacking efficiency: {eff:.1f}%")


def pack_corpus(in_dir: Path, tokenizer_path: str, out_dir: Path, seq_len: int, workers: int, flat: bool):
    """Run both phases sequentially (original behaviour, used when --phase not specified)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_phase1(in_dir, tokenizer_path, out_dir, seq_len, workers, flat)
    run_phase2(out_dir, tokenizer_path, seq_len)


def main():
    parser = argparse.ArgumentParser(description="Tokenize and pack corpus (scalable two-phase)")
    parser.add_argument("--in",      dest="in_dir",   required=True)
    parser.add_argument("--tok",     dest="tok_dir",  required=True, help="Directory with tokenizer.json")
    parser.add_argument("--out",     required=True,   help="Output directory for .bin files")
    parser.add_argument("--seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--flat",    action="store_true", help="Legacy flat JSONL input")
    parser.add_argument("--phase",   type=int, choices=[1, 2], default=None,
                        help="Run only phase 1 (tokenize shards) or phase 2 (pack). Default: both.")
    parser.add_argument("--source",  default=None,
                        help="Phase 1 only: restrict to this source directory name.")
    args = parser.parse_args()

    in_dir   = Path(args.in_dir)
    out_dir  = Path(args.out)
    tok_path = str(Path(args.tok_dir) / "tokenizer.json")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(tok_path).exists():
        print(f"Tokenizer not found: {tok_path}")
        sys.exit(1)

    print(f"workers: {args.workers}  seq_len: {args.seq_len}  phase: {args.phase or 'both'}  source: {args.source or 'all'}")

    if args.phase == 1:
        run_phase1(in_dir, tok_path, out_dir, args.seq_len, args.workers, args.flat, args.source)
    elif args.phase == 2:
        run_phase2(out_dir, tok_path, args.seq_len)
    else:
        pack_corpus(in_dir, tok_path, out_dir, args.seq_len, args.workers, args.flat)


if __name__ == "__main__":
    main()
