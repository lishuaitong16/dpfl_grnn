"""阶段五：用差分隐私防御 GRNN 梯度攻击。

流程：
  1. 取同一张真实 MNIST 图，用 Sigmoid 版 LeNet 计算真实梯度 g。
  2. 对 g 分别施加 ε_total ∈ {∞, 10, 1, 0.5, 0.1} 的"裁剪+加噪"（拉普拉斯和高斯各一组）。
     噪声强度基于简单组合定理：假设 T=30 轮，每轮隐私预算 ε_round = ε_total / T。
  3. 对每个带噪梯度跑 GRNN 攻击还原图像。
  4. 输出：
     - 防御对比大图（行=ε 配置，列=真实图/还原图）
     - PSNR 柱状图
     - 简单组合定理预算分配表

运行：
    python -m experiments.exp3_defense
（在 dpfl_grnn 目录下运行）
"""

import os
import sys
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import build_lenet
from fl.data import get_datasets, make_test_loader, denormalize
from fl.dp import privatize, per_round_epsilon
from attack.run_attack import compute_true_gradient, grnn_attack, psnr
from torch.utils.data import DataLoader
import torch.nn as nn


def noisy_gradient(true_grad, mechanism, total_eps, T, clip_C, delta=1e-5):
    """对真实梯度施加裁剪+加噪（simulating one round of DP-FL）。"""
    if total_eps is None:
        return true_grad.clone()
    eps_round = per_round_epsilon(total_eps, T)
    return privatize(true_grad.clone(), mechanism=mechanism, C=clip_C,
                     epsilon=eps_round, delta=delta)


def fake_to_vis(fake_img_tensor):
    """将 GRNN 生成器输出（tanh 范围 [-1,1]）转为 [0,1] 用于显示。"""
    return ((fake_img_tensor.clamp(-1, 1) + 1) / 2)


def save_defense_grid(real_img_vis, results_by_eps, eps_list, mech_name, save_path):
    """保存防御对比大图。

    布局：每行 = 一个 ε 配置，两列 = [真实图, 还原图]。
    第一行额外展示"无噪声下的还原图"作为攻击成功基准。
    """
    n_rows = len(eps_list)
    fig, axes = plt.subplots(n_rows, 2, figsize=(5, 2.2 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, eps in enumerate(eps_list):
        fake_vis = results_by_eps[eps]["fake_vis"][0]  # 取第 0 个样本
        ps = results_by_eps[eps]["psnr"]
        real_arr = real_img_vis[0].numpy()             # (C,H,W)
        fake_arr = fake_vis.numpy()

        axes[row][0].imshow(real_arr[0], cmap="gray")
        axes[row][0].set_title("Original", fontsize=8)
        axes[row][0].axis("off")

        eps_str = "No DP (inf)" if eps is None else f"eps_total={eps}"
        axes[row][1].imshow(fake_arr[0], cmap="gray")
        axes[row][1].set_title(f"{eps_str}\nPSNR={ps:.1f}dB", fontsize=8)
        axes[row][1].axis("off")

    plt.suptitle(f"GRNN Attack vs. DP Defense ({mech_name})", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Defense comparison figure saved: {save_path}")


def save_psnr_bar(psnr_lap, psnr_gau, eps_list, save_path):
    """保存 PSNR 柱状图（拉普拉斯 vs 高斯）。"""
    labels = ["inf\n(No DP)" if e is None else str(e) for e in eps_list]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 5))
    bars_lap = plt.bar(x - width / 2, psnr_lap, width, label="Laplace", color="steelblue")
    bars_gau = plt.bar(x + width / 2, psnr_gau, width, label="Gaussian", color="coral")

    for bar in bars_lap:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}", ha="center", fontsize=8)
    for bar in bars_gau:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"{h:.1f}", ha="center", fontsize=8)

    plt.xticks(x, [f"eps={l}" for l in labels], fontsize=9)
    plt.xlabel("Total Privacy Budget (eps_total)")
    plt.ylabel("PSNR (dB)  [higher = more similar]")
    plt.title("PSNR of GRNN Recovered Images (lower = better DP protection)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PSNR bar chart saved: {save_path}")


def print_budget_table(eps_list, T):
    """打印简单组合定理预算分配表。"""
    print("\n" + "=" * 65)
    print(f"  Composition Theorem Budget Allocation (T={T} rounds, delta=1e-5)")
    print(f"  {'eps_total':>12}  {'eps_round=eps_total/T':>20}  {'delta_total=T*delta':>20}")
    print("-" * 65)
    for eps in eps_list:
        if eps is None:
            print(f"  {'inf (No DP)':>12}  {'inf':>20}  {'-':>20}")
        else:
            eps_r = per_round_epsilon(eps, T)
            delta_total = T * 1e-5
            print(f"  {eps:>12.1f}  {eps_r:>20.4f}  {delta_total:>18.1e}")
    print("=" * 65 + "\n")


def parse_epsilons(s):
    """解析逗号分隔的 ε 列表，'inf'/'none' 表示无 DP。"""
    result = []
    for part in s.split(","):
        part = part.strip()
        if part.lower() in ("none", "inf", "nodp"):
            result.append(None)
        else:
            result.append(float(part))
    return result


def save_psnr_csv(results_all, path):
    """保存各机制、各 ε 下的 PSNR 到 CSV，供后续画图。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mechanism", "epsilon", "psnr_db"])
        for mechanism, results_by_eps in results_all.items():
            for eps, data in results_by_eps.items():
                eps_str = "inf" if eps is None else str(eps)
                writer.writerow([mechanism, eps_str, f"{data['psnr']:.4f}"])
    print(f"PSNR data saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T",          type=int,   default=30,
                   help="联邦学习总轮数（用于简单组合定理）")
    p.add_argument("--clip_C",     type=float, default=1.0,  help="梯度裁剪范数")
    p.add_argument("--epsilons",   type=str,   default="inf,10.0,1.0,0.5,0.1",
                   help="ε_total 列表，逗号分隔，'inf' 表示无 DP")
    p.add_argument("--iterations", type=int,   default=5000, help="直接像素攻击迭代次数")
    p.add_argument("--train_iters", type=int,  default=300,
                   help="攻击前在训练集上预训练模型的步数")
    p.add_argument("--data_root",  type=str,   default="./data")
    p.add_argument("--outdir",     type=str,   default="./results/defense")
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--gpu",        type=int,   default=0,
                   help="使用的 GPU 编号，-1 表示 CPU")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    eps_list = parse_epsilons(args.epsilons)

    print_budget_table(eps_list, args.T)

    # 构造 Sigmoid 版 LeNet，先在训练集上预训练使梯度携带有效信息
    model = build_lenet(num_classes=10, in_channels=1, act="sigmoid").to(device)

    train_set, test_set = get_datasets(args.data_root)

    if args.train_iters > 0:
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        model.train()
        t_iter = iter(train_loader)
        for step in range(args.train_iters):
            try:
                xb, yb = next(t_iter)
            except StopIteration:
                t_iter = iter(train_loader)
                xb, yb = next(t_iter)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
            if (step + 1) % 100 == 0:
                print(f"  Pre-training step {step+1}/{args.train_iters}")
        model.eval()
        print(f"Pre-training done ({args.train_iters} steps)")

    test_loader = make_test_loader(test_set, batch_size=1)
    x_real, y_real = next(iter(test_loader))
    x_real, y_real = x_real.to(device), y_real.to(device)

    true_grad = compute_true_gradient(model, x_real, y_real, device)
    print(f"True label: {y_real.item()}  Gradient dim: {true_grad.numel()}  Norm: {true_grad.norm().item():.4f}")
    real_vis = denormalize(x_real).cpu()

    # ------------------------------------------------------------------
    # 对两种机制分别跑所有 ε 配置
    # ------------------------------------------------------------------
    results_all = {}  # {mechanism: results_by_eps}  用于 CSV 保存
    for mech_name, mechanism in [("Laplace", "laplace"), ("Gaussian", "gaussian")]:
        print(f"\n{'='*55}")
        print(f"  Mechanism: {mech_name}")
        print(f"{'='*55}")

        results_by_eps = {}
        for eps in eps_list:
            eps_str = "inf (No DP)" if eps is None else f"eps_total={eps}"
            print(f"\n--- {eps_str} ---")

            noisy_grad = noisy_gradient(
                true_grad.detach(), mechanism, eps, args.T, args.clip_C
            ).to(device)

            result = grnn_attack(
                model, noisy_grad, batch_size=1,
                num_classes=10, img_size=32, in_channels=1,
                iterations=args.iterations, device=device, seed=args.seed,
                verbose=True,
            )

            fake_vis = fake_to_vis(result["fake_img"])
            ps = psnr(real_vis[0], fake_vis[0])
            print(f"PSNR = {ps:.2f} dB")

            results_by_eps[eps] = {
                "fake_vis": fake_vis,
                "psnr": ps,
            }

        # 保存防御对比大图
        mech_tag = "laplace" if mechanism == "laplace" else "gaussian"
        save_defense_grid(
            real_vis, results_by_eps, eps_list, mech_name,
            os.path.join(args.outdir, f"defense_{mech_tag}.png"),
        )

        results_all[mechanism] = results_by_eps
        # 收集 PSNR 用于柱状图
        if mechanism == "laplace":
            psnr_lap = [results_by_eps[e]["psnr"] for e in eps_list]
        else:
            psnr_gau = [results_by_eps[e]["psnr"] for e in eps_list]

    # 保存 PSNR 柱状图
    save_psnr_bar(psnr_lap, psnr_gau, eps_list,
                  os.path.join(args.outdir, "psnr_bar.png"))

    # 打印 trade-off 总结
    print("\n" + "=" * 65)
    print("  Privacy-Utility Trade-off Summary")
    print(f"  {'eps_total':>12}  {'Laplace PSNR':>14}  {'Gaussian PSNR':>14}")
    print("-" * 65)
    for eps, pl, pg in zip(eps_list, psnr_lap, psnr_gau):
        eps_str = "inf (No DP)" if eps is None else str(eps)
        print(f"  {eps_str:>12}  {pl:>14.2f}dB  {pg:>14.2f}dB")
    print("=" * 65)

    # 保存原始 PSNR 数据 CSV
    save_psnr_csv(results_all, os.path.join(args.outdir, "psnr_data.csv"))
    print("\nAll defense experiments done.")


if __name__ == "__main__":
    main()
