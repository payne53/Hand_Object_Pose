#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1
python main.py \
  --input_file ./datasets/ob/ \
  --output_file ./checkpoints \
  --exp $2 \
  --train \
  --val \
  --batch_size 64 \
  --model_def ContextNet \
  --gpu \
  --gpu_number 0 \
  --learning_rate 0.0001 \
  --lr_step 100 \
  --lr_step_gamma 0.9 \
  --log_batch 100 \
  --val_epoch 10 \
  --snapshot_epoch 10 \
  --num_iterations 1000 \
  --pretrained_graph /S4/MI/zhuangn/hand_pose/codes/HOPE/checkpoints/ob_graphunet/ckpt-7000.pkl \
