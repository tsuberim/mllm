#!/bin/sh
python train.py \
  --model      iphone \
  --batch_size 2 \
  --max_steps  20000 \
  --val_every  500 \
  --val_steps  50 \
  --save_every 1000 \
  --bf16
