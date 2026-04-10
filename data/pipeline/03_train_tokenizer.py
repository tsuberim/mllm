"""
Train a BPE tokenizer on the deduplicated corpus.

Tokenizer spec:
  - Algorithm: BPE (Byte-level)
  - Vocab size: 32,000
  - Backend: HuggingFace tokenizers (Rust)
  - Special tokens: 22 (essential + full agent protocol)
  - Pre-tokenizer: ByteLevel (handles all Unicode without UNK)

Usage:
    python data/pipeline/03_train_tokenizer.py --in data/deduped/ --out data/tokenizer/
    python data/pipeline/03_train_tokenizer.py --in data/deduped/ --out data/tokenizer/ --vocab-size 32000
    # Patch existing tokenizer with updated special tokens (no BPE retraining):
    python data/pipeline/03_train_tokenizer.py --patch data/tokenizer/tokenizer.json
"""

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# ---------------------------------------------------------------------------
# Special tokens — must stay in sync with harness/protocol.py
# ---------------------------------------------------------------------------

# Essential sequence tokens (always first — IDs 0-1)
ESSENTIAL_TOKENS = [
    "<|bos|>",   # beginning of sequence
    "<|eos|>",   # end of sequence / document separator
]

# Agent protocol tokens — matches harness/protocol.py ALL_SPECIAL_TOKENS
PROTOCOL_TOKENS = [
    "<|task|>",        "<|/task|>",
    "<|think|>",       "<|/think|>",
    "<|tool_call|>",   "<|/tool_call|>",
    "<|tool_result|>", "<|/tool_result|>",
    "<|spawn|>",       "<|/spawn|>",
    "<|agent_id|>",    "<|/agent_id|>",
    "<|wait|>",        "<|/wait|>",
    "<|wait_result|>", "<|/wait_result|>",
    "<|done|>",        "<|/done|>",
]

SPECIAL_TOKENS = ESSENTIAL_TOKENS + PROTOCOL_TOKENS  # 22 total

VOCAB_SIZE = 32_000


# ---------------------------------------------------------------------------
# Training data iterator
# ---------------------------------------------------------------------------

def corpus_iterator(in_dir: Path, max_chars_per_doc: int = 8_000):
    """Yield text strings from all deduped JSONL files."""
    for jsonl in sorted(in_dir.glob("*.jsonl")):
        with open(jsonl) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    content = record.get("content", "")
                    if content:
                        yield content[:max_chars_per_doc]
                except json.JSONDecodeError:
                    continue


def patch_tokenizer(tokenizer_path: str):
    """
    Add missing special tokens to an existing tokenizer without retraining BPE.
    Safe to run multiple times — skips already-present tokens.
    """
    path = Path(tokenizer_path)
    tokenizer = Tokenizer.from_file(str(path))
    vocab = tokenizer.get_vocab()

    missing = [tok for tok in SPECIAL_TOKENS if tok not in vocab]
    if not missing:
        print("All special tokens already present — nothing to do.")
        return

    from tokenizers import AddedToken
    tokenizer.add_special_tokens([AddedToken(tok, special=True) for tok in missing])

    tokenizer.save(str(path))
    print(f"Added {len(missing)} special tokens: {missing}")
    print(f"New vocab size: {tokenizer.get_vocab_size():,}")
    _sanity_check(tokenizer)


def _sanity_check(tokenizer: Tokenizer):
    test_inputs = [
        "def foo(x):\n    return x + 1\n",
        "<|task|>find all Python files<|/task|>\n<|tool_call|>find . -name '*.py'<|/tool_call|>",
        "<|think|>I'll use grep.<|/think|>\n<|done|>done<|/done|>",
    ]
    print("\nSanity checks:")
    for text in test_inputs:
        enc = tokenizer.encode(text)
        dec = tokenizer.decode(enc.ids, skip_special_tokens=False)
        match = "OK" if dec == text else f"MISMATCH: {dec!r}"
        print(f"  {len(enc.ids):3d} tokens | {match} | {text[:60]!r}")

    print("\nSpecial token IDs:")
    vocab = tokenizer.get_vocab()
    for tok in SPECIAL_TOKENS:
        print(f"  {vocab.get(tok, 'MISSING'):6}  {tok}")


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--in", dest="in_dir", help="Directory with deduped JSONL")
    parser.add_argument("--out", help="Output directory for tokenizer files")
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE, help=f"Vocab size (default {VOCAB_SIZE})")
    parser.add_argument(
        "--max-chars", type=int, default=8_000,
        help="Max chars per document fed to trainer (default 8000)",
    )
    parser.add_argument(
        "--patch", metavar="TOKENIZER_JSON",
        help="Patch an existing tokenizer.json with updated special tokens (no BPE retraining)",
    )
    args = parser.parse_args()

    if args.patch:
        patch_tokenizer(args.patch)
        return

    if not args.in_dir or not args.out:
        parser.error("--in and --out are required unless --patch is used")

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not any(in_dir.glob("*.jsonl")):
        print(f"No JSONL files found in {in_dir}")
        sys.exit(1)

    # Build tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print(f"Training BPE tokenizer (vocab={args.vocab_size}) on {in_dir} ...")
    tokenizer.train_from_iterator(
        corpus_iterator(in_dir, args.max_chars),
        trainer=trainer,
    )

    out_path = out_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Saved tokenizer → {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    _sanity_check(tokenizer)


if __name__ == "__main__":
    main()
