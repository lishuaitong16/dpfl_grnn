#!/bin/bash
# 步骤二：差分隐私联邦学习参数扫描
#
# 实际 delta 范数 ≈ 0.21，C 设为 0.3（紧裁剪），
# 收支平衡点 ε_total ≈ 15000（T=30, d=61706）。
# ε 范围覆盖"完全崩溃 → 学得不稳定 → 基本正常 → 无损"四个区间。
#
# 主实验：固定 clip_C=0.3，对拉普拉斯和高斯各扫一遍 ε_total
# 副实验：固定 ε_total=50000，扫裁剪范数 C
#
# 结果保存至：
#   results/dp_laplace_acc.png   — 拉普拉斯精度曲线
#   results/dp_gaussian_acc.png  — 高斯精度曲线
#   results/dp_clip_acc.png      — 裁剪范数 C 对比曲线
#   results/dp_acc_data.csv      — 所有精度原始数据

python -m experiments.exp1_dp_params \
    --clients      5                           \
    --rounds       30                          \
    --local_epochs 1                           \
    --local_lr     0.01                        \
    --batch_size   64                          \
    --clip_C       0.3                         \
    --epsilons     2000,10000,50000,200000,inf \
    --clip_values  0.1,0.3,1.0,3.0            \
    --clip_eps     50000                       \
    --gpu          1                           \
    --outdir       ./results
