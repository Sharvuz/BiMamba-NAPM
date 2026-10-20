#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  BiMamba-NAPM: Training on UCF-Crime from Scratch"
echo "  Target: AUC >= 90.0%"
echo "========================================================"

python src/ucf_train.py \
    --dataset ucf \
    --ssm_type s4d \
    --num_layers 2 \
    --num_prototypes 16 \
    --batch_size 64 \
    --lr 5e-5 \
    --max_epoch 15 \
    --loss_napm_weight 0.5 \
    --loss_align_weight 0.5 \
    --save_dir models/ucf
