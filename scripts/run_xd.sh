#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  BiMamba-NAPM: Training on XD-Violence from Scratch"
echo "  Target: AP > 87.0%"
echo "========================================================"

python src/xd_train.py \
    --dataset xd \
    --ssm_type s4d \
    --num_layers 2 \
    --num_prototypes 16 \
    --batch_size 64 \
    --lr 1e-4 \
    --max_epoch 20 \
    --loss_napm_weight 0.5 \
    --loss_align_weight 0.5 \
    --save_dir models/xd
