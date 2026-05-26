#!/bin/bash
# 步骤二：物理噪声联邦学习参数扫描
#
# 所有实验都走 FedAvg：客户端本地训练每步梯度上加噪（DP-SGD 风格），再上传 delta。
# 不做梯度裁剪，不计算敏感度，也不使用 epsilon/delta 换算噪声。
#
# 结果保存至：
#   results/dp_laplace_acc.png   — 拉普拉斯物理噪声精度曲线
#   results/dp_gaussian_acc.png  — 高斯物理噪声精度曲线
#   results/dp_acc_data.csv      — 所有精度原始数据

python -m experiments.exp1_dp_params \
    --clients      5                       \
    --rounds       30                      \
    --local_epochs 1                       \
    --local_lr     0.01                    \
    --batch_size   64                      \
    --noise_stds   1e-1,1e-2,1e-3,1e-4,0  \
    --gpu          1                       \
    --outdir       ./results
