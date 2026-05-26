#!/bin/bash
# 步骤五：训练阶段 vs 攻击效果实验
#
# 固定同一张图，对不同预训练步数的全局模型跑 GRNN 攻击，
# 分析梯度范数与还原质量（PSNR）随训练进度的变化规律。
#
# 结果保存至：
#   results/train_stage/stage_curve.png   — 梯度范数 & PSNR 双轴曲线
#   results/train_stage/recovered.png     — 各阶段还原图对比
#   results/train_stage/stage_data.csv    — 原始数据

python -m experiments.exp4_train_stage \
    --train_epochs_list  0,1,3,5,10,20,30,50 \
    --iterations         2000                \
    --tv_alpha           1e-3                \
    --seed               0                  \
    --gpu                3                  \
    --outdir             ./results/train_stage
