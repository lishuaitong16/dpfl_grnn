#!/bin/bash
# 步骤四：联邦学习 + 差分隐私 + GRNN 攻击防御对比
#
# 对同一张图片，在不同 ε_total 下施加 DP 噪声后跑 GRNN 攻击，
# 对比无 DP 和有 DP 时的还原效果（PSNR）。
# 噪声基于简单组合定理：ε_round = ε_total / T
#
# 结果保存至：
#   results/defense/defense_laplace.png  — 拉普拉斯防御对比大图
#   results/defense/defense_gaussian.png — 高斯防御对比大图
#   results/defense/psnr_bar.png         — PSNR 柱状图
#   results/defense/psnr_data.csv        — PSNR 原始数据

python -m experiments.exp3_defense \
    --T           30                          \
    --clip_C      0.3                         \
    --epsilons    inf,2000,10000,50000,200000 \
    --iterations  2000                        \
    --tv_alpha    1e-3                        \
    --train_iters 0                           \
    --seed        0                           \
    --gpu         2                           \
    --outdir      ./results/defense
