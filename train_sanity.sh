#!/bin/sh
python train.py \
  --model      sanity \
  --batch_size 2 \
  --max_steps  2 \
  --val_every  1 \
  --val_steps  2 \
  --save_every 1 \
  --wandb      disabled
