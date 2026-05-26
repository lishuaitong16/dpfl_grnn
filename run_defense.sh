#!/bin/bash
# 步骤四：物理噪声 + GRNN 攻击防御对比
#
# 对同一张图片的单样本梯度，直接注入均值为 0、指定标准差的物理噪声，
# 再跑 GRNN 攻击，对比无噪声 / 有噪声时的还原 PSNR。
# 不做梯度裁剪，不计算敏感度，也不使用 epsilon/delta 换算噪声。
#
# 结果保存至：
#   results/defense/defense_laplace.png  — 拉普拉斯防御对比大图
#   results/defense/defense_gaussian.png — 高斯防御对比大图
#   results/defense/psnr_bar.png         — PSNR 柱状图
#   results/defense/psnr_data.csv        — PSNR 原始数据

python -m experiments.exp3_defense \
    --noise_stds  1e-1,1e-2,1e-3,1e-4,0 \
    --iterations  2000                    \
    --tv_alpha    1e-3                    \
    --train_iters 0                       \
    --seed        0                       \
    --gpu         2                       \
    --outdir      ./results/defense
