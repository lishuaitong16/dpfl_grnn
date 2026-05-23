#!/bin/bash
# 步骤二：差分隐私联邦学习参数扫描
#
# 主实验：固定 clip_C=1.0，对拉普拉斯和高斯各扫一遍 ε_total
# 副实验：固定 ε_total=1.0，扫裁剪范数 C
#
# 结果保存至：
#   results/dp_laplace_acc.png   — 拉普拉斯精度曲线
#   results/dp_gaussian_acc.png  — 高斯精度曲线
#   results/dp_clip_acc.png      — 裁剪范数 C 对比曲线
#   results/dp_acc_data.csv      — 所有精度原始数据

python -m experiments.exp1_dp_params \
    --clients      5                        \
    --rounds       30                       \
    --local_epochs 1                        \
    --local_lr     0.01                     \
    --batch_size   64                       \
    --clip_C       1.0                      \
    --epsilons     0.1,0.5,1.0,5.0,10.0,inf \
    --clip_values  0.5,1.0,4.0             \
    --clip_eps     1.0                      \
    --gpu          0                        \
    --outdir       ./results
