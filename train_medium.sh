#!/bin/sh
python train.py \
  --model      medium \
  --batch_size 8 \
  --max_steps  2000 \
  --val_every  200 \
  --val_steps  20 \
  --save_every 200
