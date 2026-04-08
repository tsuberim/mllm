#!/bin/sh
python train.py \
  --model      medium \
  --batch_size 8 \
  --max_steps  20000 \
  --val_every  500 \
  --val_steps  50 \
  --save_every 1000
