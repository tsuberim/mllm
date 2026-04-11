#!/usr/bin/env python3
"""
Prepare SFT data: download instruction data, format into Merlin protocol,
tokenize, and pack into binary files.

Default source: HuggingFaceH4/CodeAlpaca_20K (MIT) — code instruction pairs.
Swap --dataset for real agentic traces once available (see milestone 3).

Each example is formatted as a single-turn Merlin trace:
    <|task|>{instruction}<|/task|>
    <|done|>{response}<|/done|>

Loss mask: 0 for task tokens (harness-injected), 1 for done tokens (model-generated).

Outputs (in --out dir):
    sft_train.bin       uint16  [N, SEQ_LEN]  input ids
    sft_train_mask.bin  uint8   [N, SEQ_LEN]  1 = response token (has loss)
    sft_val.bin
    sft_val_mask.bin

Usage:
    python data/pipeline/06_prepare_sft.py --tok data/tokenizer --out data/sft
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

T_TASK_O = "<|task|>"
T_TASK_C = "<|/task|>"
T_DONE_O = "<|done|>"
T_DONE_C = "<|/done|>"


def to_merlin_trace(example: dict) -> tuple[str, str]:
    """Convert a dataset example to (prompt, response) in Merlin protocol."""
    instruction = example.get("instruction", example.get("prompt", ""))
    context = example.get("input", "")
    if context:
        instruction = f"{instruction}\n\nInput:\n{context}"
    response = example.get("output", example.get("response", example.get("completion", "")))
    prompt = f"{T_TASK_O}{instruction}{T_TASK_C}\n"
    response = f"{T_DONE_O}{response}{T_DONE_C}"
    return prompt, response


def tokenize_example(tokenizer, prompt: str, response: str, eos_id: int) -> tuple[list, list]:
    """Tokenize a (prompt, response) pair. Returns (ids, loss_mask)."""
    prompt_ids   = tokenizer.encode(prompt).ids
    response_ids = tokenizer.encode(response).ids
    ids  = prompt_ids + response_ids + [eos_id]
    mask = [0] * len(prompt_ids) + [1] * len(response_ids) + [0]  # eos: no loss
    return ids, mask


def pack(examples: list, seq_len: int, eos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sequentially pack tokenized examples into fixed-length chunks.
    Chunks are padded with EOS (mask=0) when an example would overflow.
    """
    all_ids, all_masks = [], []
    buf_ids: list = []
    buf_mask: list = []

    for ids, mask in examples:
        if len(ids) > seq_len:
            continue  # silently skip; already warned at filter time
        if len(buf_ids) + len(ids) > seq_len:
            pad = seq_len - len(buf_ids)
            buf_ids  += [eos_id] * pad
            buf_mask += [0]      * pad
            all_ids.append(buf_ids[:seq_len])
            all_masks.append(buf_mask[:seq_len])
            buf_ids, buf_mask = [], []
        buf_ids  += ids
        buf_mask += mask

    if buf_ids:
        pad = seq_len - len(buf_ids)
        buf_ids  += [eos_id] * pad
        buf_mask += [0]      * pad
        all_ids.append(buf_ids[:seq_len])
        all_masks.append(buf_mask[:seq_len])

    return np.array(all_ids, dtype=np.uint16), np.array(all_masks, dtype=np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tok",          default="data/tokenizer", help="tokenizer dir")
    parser.add_argument("--out",          default="data/sft",       help="output dir")
    parser.add_argument("--dataset",      default="HuggingFaceH4/CodeAlpaca_20K")
    parser.add_argument("--split",        default="train",          help="HF dataset split to load")
    parser.add_argument("--seq-len",      type=int, default=2048)
    parser.add_argument("--val-frac",     type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=None,   help="cap for quick tests")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── tokenizer ─────────────────────────────────────────────────────────────
    from tokenizers import Tokenizer
    tok_file = Path(args.tok) / "tokenizer.json"
    if not tok_file.exists():
        from huggingface_hub import hf_hub_download
        import os
        tok_file = Path(hf_hub_download(
            "tsuberim/merlin-tokenizer-v0", "tokenizer.json",
            token=os.environ.get("HF_TOKEN"),
        ))
    tokenizer = Tokenizer.from_file(str(tok_file))
    eos_id = tokenizer.token_to_id("<|eos|>")
    assert eos_id is not None, "tokenizer missing <|eos|>"
    for tok in (T_TASK_O, T_TASK_C, T_DONE_O, T_DONE_C):
        assert tokenizer.token_to_id(tok) is not None, f"tokenizer missing {tok}"
    print(f"tokenizer: vocab_size={tokenizer.get_vocab_size()}, eos_id={eos_id}")

    # ── dataset ───────────────────────────────────────────────────────────────
    from datasets import load_dataset
    print(f"loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))
    print(f"  {len(ds)} examples")

    # ── tokenize ──────────────────────────────────────────────────────────────
    examples = []
    skipped = 0
    for ex in ds:
        prompt, response = to_merlin_trace(ex)
        ids, mask = tokenize_example(tokenizer, prompt, response, eos_id)
        if len(ids) > args.seq_len:
            skipped += 1
            continue
        examples.append((ids, mask))

    print(f"tokenized: {len(examples)} kept, {skipped} skipped (>{args.seq_len} tokens)")

    # ── shuffle + split ───────────────────────────────────────────────────────
    random.shuffle(examples)
    val_n = max(1, int(len(examples) * args.val_frac))
    splits = {"train": examples[val_n:], "val": examples[:val_n]}

    # ── pack + write ──────────────────────────────────────────────────────────
    for split, exs in splits.items():
        ids_arr, mask_arr = pack(exs, args.seq_len, eos_id)
        n_resp  = int(mask_arr.sum())
        n_total = int(ids_arr.size)
        print(f"{split}: {len(ids_arr)} chunks  |  {n_resp:,} response tokens "
              f"({100*n_resp/n_total:.1f}% of all tokens)")
        ids_arr.tofile(out / f"sft_{split}.bin")
        mask_arr.tofile(out / f"sft_{split}_mask.bin")

    print(f"\ndone → {out}/")


if __name__ == "__main__":
    main()
