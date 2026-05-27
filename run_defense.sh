#!/bin/bash
# 步骤四：差分隐私防御 GRNN 攻击对比
#
# 对同一张图片的单样本梯度，施加 per-round delta DP 噪声（裁剪 C=0.05，δ=1e-5），
# 再跑 GRNN 攻击，对比各 ε_round 下的还原 PSNR。
# C=0.05 略大于实际 delta norm（≈0.044），既不裁剪梯度方向，又使噪声量合理。
# ε_round 扫描范围涵盖 SNR≈0 到 SNR>>1 的过渡区，攻击者截获单轮梯度，ε_total = ε_round × 1。
#
# 结果保存至：
#   results/defense/defense_laplace.png  — Laplace 防御对比大图
#   results/defense/defense_gaussian.png — Gaussian 防御对比大图
#   results/defense/psnr_bar.png         — PSNR 柱状图
#   results/defense/psnr_data.csv        — PSNR 原始数据

python -m experiments.exp3_defense \
    --epsilon_round 100,500,1000,2000,5000,inf \
    --clip_C       0.05                \
    --iterations   2000               \
    --tv_alpha     1e-3               \
    --train_iters  0                  \
    --seed         0                  \
    --gpu          2                  \
    --outdir       ./results/defense
