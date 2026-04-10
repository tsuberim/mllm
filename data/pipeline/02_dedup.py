"""
Duplicate removal via two-pass strategy:
  1. Exact dedup — SHA1 of normalized content (fast, handles copy-paste)
  2. Near-dedup — MinHash LSH with 5-gram shingling (slower, handles minor edits)

Near-dedup is opt-in via --near flag. The Stack v1 was already deduped by BigCode,
so exact dedup alone is usually sufficient for our filtered subsets.

Performance: exact ~50K rec/s; near-dedup ~200-400 rec/s (64 perms, 3K chars)

Usage:
    python data/pipeline/02_dedup.py --source stack_python --in data/filtered/ --out data/deduped/
    python data/pipeline/02_dedup.py --all --in data/filtered/ --out data/deduped/
    python data/pipeline/02_dedup.py --source stack_python --near --in data/filtered/ --out data/deduped/
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NGRAM_SIZE = 5           # character 5-grams
NUM_PERM = 64            # MinHash permutations (64 is fast; 128 more accurate)
JACCARD_THRESHOLD = 0.7  # docs with Jaccard >= 0.7 treated as duplicates
NEAR_MAX_CHARS = 3000    # chars to shingle (enough for structural similarity)

SOURCES = ["stack_python", "stack_bash", "stack_md", "tldr", "stackoverflow", "github_commits", "github_issues"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """SHA1 of whitespace-normalized content."""
    normalized = " ".join(text.split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def make_minhash(text: str):
    from datasketch import MinHash
    m = MinHash(num_perm=NUM_PERM)
    text = text[:NEAR_MAX_CHARS]
    for i in range(len(text) - NGRAM_SIZE + 1):
        m.update(text[i:i + NGRAM_SIZE].encode("utf-8"))
    return m


def dedup_exact(source: str, in_path: Path, out_path: Path):
    """Fast exact dedup via content hash."""
    seen = set()
    total = kept = dup = 0

    with open(in_path) as in_f, open(out_path, "w") as out_f:
        for line in tqdm(in_f, desc=f"{source} (exact)", unit="rec"):
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = record.get("content", "")
            if not content:
                continue
            h = content_hash(content)
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    pct = 100 * kept / max(total, 1)
    print(f"  {source}: {kept:,}/{total:,} kept ({pct:.1f}%), {dup:,} exact dupes → {out_path}")
    return kept


def dedup_near(source: str, in_path: Path, out_path: Path, threshold: float):
    """Near-dedup via MinHash LSH. Slower — ~200-400 rec/s. Use after exact dedup."""
    from datasketch import MinHashLSH
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    total = kept = dup = 0

    with open(in_path) as in_f, open(out_path, "w") as out_f:
        for i, line in enumerate(tqdm(in_f, desc=f"{source} (near)", unit="rec")):
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = record.get("content", "")
            if not content:
                continue

            mh = make_minhash(content)
            if lsh.query(mh):
                dup += 1
                continue
            lsh.insert(f"d{i}", mh)
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    pct = 100 * kept / max(total, 1)
    print(f"  {source}: {kept:,}/{total:,} kept ({pct:.1f}%), {dup:,} near-dupes → {out_path}")


def dedup_source(source: str, in_dir: Path, out_dir: Path, near: bool, threshold: float):
    in_path = in_dir / f"{source}.jsonl"
    if not in_path.exists():
        print(f"  {source}: not found at {in_path}, skipping")
        return

    out_path = out_dir / f"{source}.jsonl"

    if not near:
        dedup_exact(source, in_path, out_path)
    else:
        # Two-pass: exact first, then near-dedup on survivors
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as tmp:
            tmp_path = Path(tmp.name)
        dedup_exact(source, in_path, tmp_path)
        dedup_near(source, tmp_path, out_path, threshold)
        tmp_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Dedup filtered corpus shards")
    parser.add_argument("--in", dest="in_dir", required=True, help="Directory with filtered JSONL")
    parser.add_argument("--out", required=True, help="Output directory for deduped JSONL")
    parser.add_argument("--source", choices=SOURCES, help="Single source to dedup")
    parser.add_argument("--all", action="store_true", help="Dedup all sources")
    parser.add_argument("--near", action="store_true", help="Also run near-dedup (slower)")
    parser.add_argument(
        "--threshold", type=float, default=JACCARD_THRESHOLD,
        help=f"Jaccard similarity threshold for near-dedup (default {JACCARD_THRESHOLD})",
    )
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        sources = SOURCES
    elif args.source:
        sources = [args.source]
    else:
        parser.print_help()
        sys.exit(1)

    for source in sources:
        dedup_source(source, in_dir, out_dir, near=args.near, threshold=args.threshold)


if __name__ == "__main__":
    main()
