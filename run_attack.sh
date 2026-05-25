#!/bin/bash
# 步骤三：GRNN 梯度逆向攻击（无差分隐私）
#
# 结果保存至：
#   results/attack/recover_process.png  — 逐步还原过程
#   results/attack/compare.png          — 还原图 vs 真实图对比
#   results/attack/loss_curve.png       — 损失下降曲线

python -m experiments.exp2_attack \
    --batch_size  1    \
    --iterations  2000 \
    --tv_alpha    1e-3 \
    --train_iters 0    \
    --seed        0    \
    --gpu         3    \
    --outdir      ./results/attack
