#!/bin/bash
# 步骤四：差分隐私防御 GRNN 攻击对比
#
# 对齐官方实现（GRNN.py）：直接用 autograd 算单样本梯度，噪声加到梯度上。
# sensitivity = 2 * lr * C / 1（batchsize=1），与官方 cal_client_sensitivity 一致。
# DP 参数与 FL 训练实验（exp1）完全对齐：C=1.0，ε_round ∈ {10,25,50,75,100,∞}。
#
# 结果保存至：
#   results/defense/defense_laplace.png  — Laplace 防御对比大图
#   results/defense/defense_gaussian.png — Gaussian 防御对比大图
#   results/defense/psnr_bar.png         — PSNR 柱状图
#   results/defense/psnr_data.csv        — PSNR 原始数据

python -m experiments.exp3_defense \
    --epsilon_round 10,25,50,75,100,inf \
    --clip_C        1.0                 \
    --local_lr      0.01                \
    --iterations    2000                \
    --tv_alpha      1e-3                \
    --train_iters   0                   \
    --seed          0                   \
    --gpu           2                   \
    --outdir        ./results/defense
