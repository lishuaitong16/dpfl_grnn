"""阶段五：用差分隐私噪声防御 GRNN 梯度攻击。

流程：
  1. 取同一张真实 MNIST 图，用 Sigmoid 版 LeNet 计算单样本真实梯度 g。
  2. 对 g 应用 per-round delta DP（裁剪 + 加噪），得到 g_hat。
  3. 对每个带噪梯度跑 GRNN 攻击还原图像。
  4. 输出防御对比图、PSNR 柱状图和 CSV。

DP 参数：C=1.0，δ=1e-5，ε_total ∈ sweep（∞ 表示无噪声基线）。
每轮 ε_round = ε_total / rounds（rounds=30 与 exp1 一致）。
"""

import os
import sys
import csv
import math
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import build_lenet
from fl.data import get_datasets, make_test_loader, denormalize
from fl.dp import privatize, noise_std_from_epsilon
from attack.run_attack import compute_true_gradient, grnn_attack, psnr
from torch.utils.data import DataLoader
import torch.nn as nn


CLIP_C   = 1.0
DELTA_DP = 1e-5
FL_ROUNDS = 30   # 与 exp1 保持一致，用于换算 ε_round


def noisy_gradient(true_grad, mechanism, total_epsilon):
    """对真实梯度应用 per-round delta DP 噪声（裁剪+加噪）。

    使用单轮预算 ε_round = ε_total / FL_ROUNDS，与 FL 训练中每轮噪声量等价。
    ε_total=∞ 时返回裁剪后的原始梯度（无噪声）。
    """
    g = true_grad.clone()
    if math.isinf(total_epsilon):
        return g
    epsilon_round = total_epsilon / FL_ROUNDS
    return privatize(g, mechanism, epsilon_round, CLIP_C, DELTA_DP)


def fake_to_vis(fake_img_tensor):
    return fake_img_tensor.clamp(0, 1)


def eps_label(eps):
    return "No DP" if math.isinf(eps) else f"ε={eps:g}"


def save_defense_grid(real_img_vis, results_by_eps, epsilons, mech_name, save_path):
    """保存防御对比大图。布局：每行 = 一个 ε 配置，两列 = [真实图, 还原图]。"""
    n_rows = len(epsilons)
    fig, axes = plt.subplots(n_rows, 2, figsize=(5, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, eps in enumerate(epsilons):
        fake_vis = results_by_eps[eps]["fake_vis"][0]
        ps = results_by_eps[eps]["psnr"]
        real_arr = real_img_vis[0].mean(0).numpy()
        fake_arr = fake_vis.mean(0).numpy()

        for ax in axes[row]:
            ax.axis("off")

        axes[row][0].imshow(real_arr, cmap="gray")
        axes[row][0].set_title("Original", fontsize=8, pad=4)

        axes[row][1].imshow(fake_arr, cmap="gray")
        axes[row][1].set_title(f"{eps_label(eps)}  PSNR={ps:.1f}dB", fontsize=8, pad=4)

    plt.suptitle(f"GRNN Attack vs. DP Defense ({mech_name})", fontsize=10, y=1.01)
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Defense comparison figure saved: {save_path}")


def save_psnr_bar(psnr_lap, psnr_gau, epsilons, save_path):
    """保存 PSNR 柱状图（Laplace vs Gaussian）。"""
    labels = [eps_label(e) for e in epsilons]
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

    plt.xticks(x, labels, fontsize=9)
    plt.xlabel("Privacy Budget  ε_total")
    plt.ylabel("PSNR (dB)  [higher = more similar to original]")
    plt.title("PSNR of GRNN Recovered Images under DP Defense\n"
              f"(C={CLIP_C}, δ={DELTA_DP}, lower PSNR = better protection)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PSNR bar chart saved: {save_path}")


def parse_epsilons(s):
    """解析逗号分隔的 ε 列表，inf/∞/nodp 表示无 DP。"""
    result = []
    for part in s.split(","):
        part = part.strip()
        if part.lower() in ("inf", "∞", "nodp", "none"):
            result.append(float("inf"))
        else:
            result.append(float(part))
    return result


def save_psnr_csv(results_all, epsilons, path):
    """保存各机制、各 ε 下的 PSNR 和噪声标准差到 CSV。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mechanism", "epsilon_total", "epsilon_round", "noise_std", "psnr_db"])
        for mechanism, results_by_eps in results_all.items():
            for eps, data in results_by_eps.items():
                eps_str = "inf" if math.isinf(eps) else f"{eps:g}"
                if math.isinf(eps):
                    eps_r_str = "inf"
                    sigma_str = "0"
                else:
                    eps_r = eps / FL_ROUNDS
                    sigma = noise_std_from_epsilon(mechanism, eps_r, CLIP_C, DELTA_DP)
                    eps_r_str = f"{eps_r:.4f}"
                    sigma_str = f"{sigma:.4f}"
                writer.writerow([mechanism, eps_str, eps_r_str, sigma_str,
                                 f"{data['psnr']:.4f}"])
    print(f"PSNR data saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epsilons",    type=str,   default="20,50,100,200,500,inf",
                   help="逗号分隔的 ε_total 列表，inf 表示无 DP")
    p.add_argument("--iterations",  type=int,   default=5000, help="GRNN 攻击迭代次数")
    p.add_argument("--train_iters", type=int,   default=300,
                   help="攻击前在训练集上预训练模型的步数")
    p.add_argument("--tv_alpha",    type=float, default=1e-3)
    p.add_argument("--data_root",   type=str,   default="./data")
    p.add_argument("--outdir",      type=str,   default="./results/defense")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument("--gpu",         type=int,   default=0,
                   help="使用的 GPU 编号，-1 表示 CPU")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    epsilons = parse_epsilons(args.epsilons)
    print("\nε_total sweep:", ", ".join(eps_label(e) for e in epsilons))

    # Sigmoid LeNet：GRNN 需要 sigmoid 做二阶梯度反传。
    model = build_lenet(num_classes=10, in_channels=1, act="sigmoid").to(device)

    train_set, test_set = get_datasets(args.data_root)

    train_loader_single = DataLoader(train_set, batch_size=1, shuffle=True)
    x_real, y_real = next(iter(train_loader_single))
    x_real, y_real = x_real.to(device), y_real.to(device)

    if args.train_iters > 0:
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
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

    true_grad = compute_true_gradient(model, x_real, y_real, device)
    true_norm = true_grad.norm().item()
    print(f"True label: {y_real.item()}  Gradient dim: {true_grad.numel()}  "
          f"Gradient norm: {true_norm:.4f}")
    real_vis = denormalize(x_real).cpu()

    # ------------------------------------------------------------------
    # 打印噪声参数对照表
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  DP 参数表  C={CLIP_C}  δ={DELTA_DP}  FL_ROUNDS={FL_ROUNDS}")
    print(f"  {'ε_total':>10}  {'ε_round':>10}  {'σ_Lap':>10}  {'σ_Gau':>10}  {'SNR_Lap':>10}")
    print(f"  {'-'*60}")
    for eps in epsilons:
        if math.isinf(eps):
            print(f"  {'∞':>10}  {'∞':>10}  {'0':>10}  {'0':>10}  {'∞':>10}")
        else:
            eps_r = eps / FL_ROUNDS
            sigma_lap = noise_std_from_epsilon("laplace",  eps_r, CLIP_C, DELTA_DP)
            sigma_gau = noise_std_from_epsilon("gaussian", eps_r, CLIP_C, DELTA_DP)
            snr = true_norm / sigma_lap if sigma_lap > 0 else float("inf")
            print(f"  {eps:>10g}  {eps_r:>10.4f}  {sigma_lap:>10.4f}  {sigma_gau:>10.4f}  {snr:>10.4f}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # 对两种机制分别跑所有 ε 配置
    # ------------------------------------------------------------------
    results_all = {}
    for mech_name, mechanism in [("Laplace", "laplace"), ("Gaussian", "gaussian")]:
        print(f"\n{'='*55}")
        print(f"  Mechanism: {mech_name}")
        print(f"{'='*55}")

        results_by_eps = {}
        for eps in epsilons:
            print(f"\n--- {eps_label(eps)} ---")

            noisy_grad = noisy_gradient(true_grad.detach(), mechanism, eps).to(device)

            result = grnn_attack(
                model, noisy_grad, batch_size=1,
                num_classes=10, img_size=32, in_channels=1,
                iterations=args.iterations, tv_alpha=args.tv_alpha,
                device=device, seed=args.seed,
                verbose=True,
            )

            fake_vis = fake_to_vis(result["fake_img"])
            ps = psnr(real_vis[0], fake_vis[0])
            print(f"PSNR = {ps:.2f} dB")

            results_by_eps[eps] = {
                "fake_vis": fake_vis,
                "psnr": ps,
            }

        mech_tag = "laplace" if mechanism == "laplace" else "gaussian"
        save_defense_grid(
            real_vis, results_by_eps, epsilons, mech_name,
            os.path.join(args.outdir, f"defense_{mech_tag}.png"),
        )

        results_all[mechanism] = results_by_eps
        if mechanism == "laplace":
            psnr_lap = [results_by_eps[e]["psnr"] for e in epsilons]
        else:
            psnr_gau = [results_by_eps[e]["psnr"] for e in epsilons]

    save_psnr_bar(psnr_lap, psnr_gau, epsilons,
                  os.path.join(args.outdir, "psnr_bar.png"))

    print("\n" + "=" * 70)
    print("  Privacy-Utility Summary")
    print(f"  {'ε_total':>12}  {'Laplace PSNR':>14}  {'Gaussian PSNR':>14}")
    print("-" * 70)
    for eps, pl, pg in zip(epsilons, psnr_lap, psnr_gau):
        print(f"  {eps_label(eps):>12}  {pl:>14.2f}dB  {pg:>14.2f}dB")
    print("=" * 70)

    save_psnr_csv(results_all, epsilons, os.path.join(args.outdir, "psnr_data.csv"))
    print("\nAll defense experiments done.")


if __name__ == "__main__":
    main()
