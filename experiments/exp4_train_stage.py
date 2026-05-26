"""阶段六：训练阶段 vs 攻击效果实验。

对同一张图、同一个初始模型，分别预训练 train_iters 步后：
  1. 计算真实梯度范数 ||g_true||_2（反映模型训练阶段）
  2. 跑 GRNN 攻击，记录 PSNR

结果保存至：
  results/train_stage/stage_curve.png  — 梯度范数 & PSNR vs 训练步数
  results/train_stage/recovered.png    — 各阶段还原图对比
  results/train_stage/stage_data.csv   — 原始数据
"""

import os
import sys
import csv
import copy
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import build_lenet
from fl.data import get_datasets, make_test_loader, denormalize
from fl.train import evaluate
from attack.run_attack import compute_true_gradient, grnn_attack, psnr


def pretrain(model, train_set, epochs, device):
    """用 SGD 在训练集上预训练 model epochs 轮，返回新模型（不修改原模型）。"""
    model = copy.deepcopy(model)
    if epochs == 0:
        return model
    loader = DataLoader(train_set, batch_size=64, shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_epochs_list", type=str,
                   default="0,1,3,5,10,20,30,50",
                   help="逗号分隔的预训练 epoch 数列表")
    p.add_argument("--iterations",  type=int,   default=2000,  help="GRNN 攻击迭代次数")
    p.add_argument("--tv_alpha",    type=float, default=1e-3)
    p.add_argument("--data_root",   type=str,   default="./data")
    p.add_argument("--outdir",      type=str,   default="./results/train_stage")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--gpu",         type=int,   default=1)
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    steps_list = [int(s) for s in args.train_epochs_list.split(",")]

    train_set, test_set = get_datasets(args.data_root)
    test_loader = make_test_loader(test_set)

    # 固定同一张图（所有训练阶段共用）
    loader_single = DataLoader(train_set, batch_size=1, shuffle=True)
    x_real, y_real = next(iter(loader_single))
    x_real, y_real = x_real.to(device), y_real.to(device)
    real_vis = denormalize(x_real).cpu()
    print(f"Fixed image label: {y_real.item()}")

    # 随机初始化的基础模型（所有阶段从同一初始化出发）
    base_model = build_lenet(num_classes=10, in_channels=1, act="sigmoid").to(device)

    records = []  # [(steps, grad_norm, psnr_val)]
    fake_imgs = []

    for steps in steps_list:
        print(f"\n{'='*50}")
        print(f"  train_epochs = {steps}")
        print(f"{'='*50}")

        model = pretrain(base_model, train_set, steps, device)

        acc = evaluate(model, test_loader, device)
        print(f"  test acc  = {acc*100:.2f}%")

        true_grad = compute_true_gradient(model, x_real, y_real, device)
        grad_norm = true_grad.norm().item()
        print(f"  grad_norm = {grad_norm:.4f}")

        result = grnn_attack(
            model, true_grad, batch_size=1,
            num_classes=10, img_size=32, in_channels=1,
            iterations=args.iterations, tv_alpha=args.tv_alpha,
            device=device, seed=args.seed, verbose=False,
        )

        fake_vis = result["fake_img"].clamp(0, 1)
        ps = psnr(real_vis[0], fake_vis[0])
        print(f"  PSNR = {ps:.2f} dB")

        records.append((steps, acc, grad_norm, ps))
        fake_imgs.append(fake_vis[0])

    # ---- 保存 CSV ----
    csv_path = os.path.join(args.outdir, "stage_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["train_epochs", "test_acc", "grad_norm", "psnr_db"])
        for row in records:
            w.writerow([row[0], f"{row[1]*100:.2f}", f"{row[2]:.6f}", f"{row[3]:.4f}"])
    print(f"\nCSV saved: {csv_path}")

    # ---- 画双 Y 轴曲线 ----
    steps_arr   = [r[0] for r in records]
    norms_arr   = [r[2] for r in records]
    psnrs_arr   = [r[3] for r in records]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    l1, = ax1.plot(steps_arr, norms_arr, "o-", color="steelblue", label="Grad Norm ||g||₂")
    l2, = ax2.plot(steps_arr, psnrs_arr, "s--", color="coral",    label="PSNR (dB)")

    ax1.set_xlabel("Pre-training Epochs")
    ax1.set_ylabel("Gradient Norm  ||g_true||₂", color="steelblue")
    ax2.set_ylabel("PSNR (dB)  [higher = better recovery]", color="coral")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="coral")

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    plt.title("Training Stage vs Attack Effectiveness (GRNN)")
    plt.tight_layout()
    curve_path = os.path.join(args.outdir, "stage_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"Curve saved: {curve_path}")

    # ---- 画各阶段还原图 ----
    n = len(steps_list)
    fig, axes = plt.subplots(2, n + 1, figsize=(2.5 * (n + 1), 6))
    for ax in axes.flatten():
        ax.axis("off")

    # 第一列：原图
    axes[0][0].imshow(real_vis[0].mean(0).numpy(), cmap="gray")
    axes[0][0].set_title(f"Original\nlabel={y_real.item()}", fontsize=8)
    axes[1][0].axis("off")

    for i, (steps, acc, grad_norm, ps) in enumerate(records):
        axes[0][i + 1].imshow(fake_imgs[i].mean(0).numpy(), cmap="gray")
        axes[0][i + 1].set_title(f"epoch={steps}\nPSNR={ps:.1f}dB", fontsize=7)
        axes[1][i + 1].text(0.5, 0.5, f"acc={acc*100:.1f}%\n‖g‖={grad_norm:.2f}",
                             ha="center", va="center", fontsize=8,
                             transform=axes[1][i + 1].transAxes)

    plt.suptitle("Recovered Images at Different Training Stages", fontsize=10)
    plt.tight_layout(pad=0.5)
    rec_path = os.path.join(args.outdir, "recovered.png")
    plt.savefig(rec_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Recovered images saved: {rec_path}")

    # ---- 打印汇总 ----
    print("\n" + "=" * 70)
    print(f"  {'train_epochs':>12}  {'test_acc':>10}  {'grad_norm':>12}  {'PSNR (dB)':>12}")
    print("-" * 70)
    for steps, acc, grad_norm, ps in records:
        print(f"  {steps:>12}  {acc*100:>9.2f}%  {grad_norm:>12.4f}  {ps:>12.2f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
