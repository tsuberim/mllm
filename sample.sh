#!/bin/sh
# Convert latest checkpoint and sample from the iphone model.
# Usage: ./sample.sh ["your prompt here"] [max_new_tokens]
MODEL=iphone
CKPT="checkpoints/ckpt_${MODEL}.pt"
WEIGHTS="checkpoints/weights_${MODEL}.npz"
PROMPT="${1:-Once upon a time}"
MAX_NEW="${2:-200}"

if [ ! -f "$CKPT" ]; then
  echo "no checkpoint found at $CKPT — still training?"
  exit 1
fi

python convert.py "$CKPT" "$WEIGHTS"
python infer.py --model "$MODEL" --weights "$WEIGHTS" --prompt "$PROMPT" --max_new "$MAX_NEW"
