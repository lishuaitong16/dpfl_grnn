#!/bin/bash
# 步骤一：传统联邦学习基线（无差分隐私）
# 结果保存至 results/baseline_acc.png

python -m experiments.exp0_fl_baseline \
    --clients      5    \
    --rounds       30   \
    --local_epochs 1    \
    --local_lr     0.01 \
    --batch_size   64   \
    --gpu          0    \
    --out          ./results/baseline_acc.png
