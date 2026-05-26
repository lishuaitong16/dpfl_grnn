#!/bin/bash
# 步骤四：差分隐私防御 GRNN 攻击对比
#
# 对同一张图片的单样本梯度，施加 per-round delta DP 噪声（裁剪 C=1.0，δ=1e-5），
# 再跑 GRNN 攻击，对比各 ε_total 下的还原 PSNR。
# ε_round = ε_total / 30（与 FL 训练一致）。
#
# 结果保存至：
#   results/defense/defense_laplace.png  — Laplace 防御对比大图
#   results/defense/defense_gaussian.png — Gaussian 防御对比大图
#   results/defense/psnr_bar.png         — PSNR 柱状图
#   results/defense/psnr_data.csv        — PSNR 原始数据

python -m experiments.exp3_defense \
    --epsilons     1,5,10,50,100,inf  \
    --iterations   2000               \
    --tv_alpha     1e-3               \
    --train_iters  0                  \
    --seed         0                  \
    --gpu          2                  \
    --outdir       ./results/defense
