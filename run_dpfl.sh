#!/bin/bash
# 步骤二：差分隐私联邦学习（ε 扫描）
#
# per-round delta DP：每轮裁剪客户端 delta（C=1.0），加校准噪声（δ=1e-5）。
# 扫描 ε_round ∈ {10,25,50,75,100,∞}，对比 Laplace / Gaussian 两种机制。
# ε_total = ε_round × 30（T=30 轮，顺序合成定理）。
#
# 结果保存至：
#   results/dp_laplace_acc.png   — Laplace DP 精度曲线
#   results/dp_gaussian_acc.png  — Gaussian DP 精度曲线
#   results/dp_acc_data.csv      — 所有精度原始数据

python -m experiments.exp1_dp_params \
    --clients      5                   \
    --rounds       30                  \
    --local_epochs 1                   \
    --local_lr     0.01                \
    --batch_size   64                  \
    --epsilon_round 10,25,50,75,100,inf \
    --clip_C       1.0                 \
    --gpu          1                   \
    --outdir       ./results
