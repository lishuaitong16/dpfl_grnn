"""阶段五：用物理噪声防御 GRNN 梯度攻击。

流程：
  1. 取同一张真实 MNIST 图，用 Sigmoid 版 LeNet 计算单样本真实梯度 g。
  2. 直接在梯度空间对 g 注入不同标准差的零均值物理噪声，得到 g_hat。
  3. 对每个带噪梯度跑 GRNN 攻击还原图像。
  4. 输出防御对比图、PSNR 柱状图和 CSV。

本实验不做梯度裁剪，不计算敏感度，也不使用 epsilon/delta 换算噪声。
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
from fl.dp import privatize
from attack.run_attack import compute_true_gradient, grnn_attack, psnr
from torch.utils.data import DataLoader
import torch.nn as nn


def noisy_gradient(true_grad, mechanism, noise_std):
    """对真实梯度直接注入指定标准差的物理噪声。"""
    if noise_std == 0:
        return true_grad.clone()
    return privatize(true_grad.clone(), mechanism=mechanism, noise_std=noise_std)


def fake_to_vis(fake_img_tensor):
    """将 GRNN 生成器输出（sigmoid 范围 [0,1]）截断后返回。"""
    return fake_img_tensor.clamp(0, 1)


def save_defense_grid(real_img_vis, results_by_noise, noise_stds, mech_name, save_path):
    """保存防御对比大图。

    布局：每行 = 一个噪声标准差配置，两列 = [真实图, 还原图]。
    """
    n_rows = len(noise_stds)
    fig, axes = plt.subplots(n_rows, 2, figsize=(5, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for row, std in enumerate(noise_stds):
        fake_vis = results_by_noise[std]["fake_vis"][0]
        ps = results_by_noise[std]["psnr"]
        real_arr = real_img_vis[0].mean(0).numpy()
        fake_arr = fake_vis.mean(0).numpy()

        for ax in axes[row]:
            ax.axis("off")

        axes[row][0].imshow(real_arr, cmap="gray")
        axes[row][0].set_title("Original", fontsize=8, pad=4)

        noise_str = "No noise" if std == 0 else f"std={std:g}"
        axes[row][1].imshow(fake_arr, cmap="gray")
        axes[row][1].set_title(f"{noise_str}  PSNR={ps:.1f}dB", fontsize=8, pad=4)

    plt.suptitle(f"GRNN Attack vs. Physical Noise Defense ({mech_name})", fontsize=10, y=1.01)
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Defense comparison figure saved: {save_path}")


def save_psnr_bar(psnr_lap, psnr_gau, noise_stds, save_path):
    """保存 PSNR 柱状图（拉普拉斯 vs 高斯）。"""
    labels = ["0\n(No noise)" if s == 0 else f"{s:g}" for s in noise_stds]
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

    plt.xticks(x, [f"std={l}" for l in labels], fontsize=9)
    plt.xlabel("Physical Noise Std")
    plt.ylabel("PSNR (dB)  [higher = more similar]")
    plt.title("PSNR of GRNN Recovered Images (lower = better DP protection)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PSNR bar chart saved: {save_path}")


def parse_noise_stds(s):
    """解析逗号分隔的噪声标准差列表，0/inf/none 表示无噪声。"""
    result = []
    for part in s.split(","):
        part = part.strip()
        if part.lower() in ("none", "inf", "nodp", "nonoise"):
            result.append(0.0)
        else:
            result.append(float(part))
    return result


def noise_label(std):
    return "No noise" if std == 0 else f"std={std:g}"


def save_psnr_csv(results_all, path):
    """保存各机制、各噪声标准差下的 PSNR 到 CSV，供后续画图。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mechanism", "noise_std", "psnr_db"])
        for mechanism, results_by_noise in results_all.items():
            for std, data in results_by_noise.items():
                writer.writerow([mechanism, f"{std:g}", f"{data['psnr']:.4f}"])
    print(f"PSNR data saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noise_stds", type=str,   default="1e-1,1e-2,1e-3,1e-4,0",
                   help="直接注入到真实梯度上的噪声标准差列表，0 表示无噪声")
    p.add_argument("--iterations", type=int,   default=5000, help="直接像素攻击迭代次数")
    p.add_argument("--train_iters", type=int,  default=300,
                   help="攻击前在训练集上预训练模型的步数")
    p.add_argument("--tv_alpha",   type=float, default=1e-3, help="TV 损失权重")
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

    noise_stds = parse_noise_stds(args.noise_stds)
    print("\nNoise std sweep:", ", ".join(noise_label(s) for s in noise_stds))

    # Sigmoid LeNet：GRNN 需要 sigmoid 做二阶梯度反传（见 attack/run_attack.py）。
    # exp1 FL 训练使用 ReLU，原因见 exp1_dp_params.py 注释。
    model = build_lenet(num_classes=10, in_channels=1, act="sigmoid").to(device)

    train_set, test_set = get_datasets(args.data_root)

    # 先从训练集取一张图（FL 中梯度来自客户端训练数据）
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
    print(f"True label: {y_real.item()}  Gradient dim: {true_grad.numel()}  "
          f"Gradient norm: {true_grad.norm().item():.4f}")
    real_vis = denormalize(x_real).cpu()

    # ------------------------------------------------------------------
    # 对两种机制分别跑所有噪声配置
    # ------------------------------------------------------------------
    results_all = {}  # {mechanism: results_by_noise}  用于 CSV 保存
    for mech_name, mechanism in [("Laplace", "laplace"), ("Gaussian", "gaussian")]:
        print(f"\n{'='*55}")
        print(f"  Mechanism: {mech_name}")
        print(f"{'='*55}")

        results_by_noise = {}
        for std in noise_stds:
            print(f"\n--- {noise_label(std)} ---")

            noisy_grad = noisy_gradient(
                true_grad.detach(), mechanism, std
            ).to(device)

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

            results_by_noise[std] = {
                "fake_vis": fake_vis,
                "psnr": ps,
            }

        # 保存防御对比大图
        mech_tag = "laplace" if mechanism == "laplace" else "gaussian"
        save_defense_grid(
            real_vis, results_by_noise, noise_stds, mech_name,
            os.path.join(args.outdir, f"defense_{mech_tag}.png"),
        )

        results_all[mechanism] = results_by_noise
        # 收集 PSNR 用于柱状图
        if mechanism == "laplace":
            psnr_lap = [results_by_noise[s]["psnr"] for s in noise_stds]
        else:
            psnr_gau = [results_by_noise[s]["psnr"] for s in noise_stds]

    # 保存 PSNR 柱状图
    save_psnr_bar(psnr_lap, psnr_gau, noise_stds,
                  os.path.join(args.outdir, "psnr_bar.png"))

    # 打印 trade-off 总结
    print("\n" + "=" * 65)
    print("  Noise-Robustness Summary")
    print(f"  {'noise_std':>12}  {'Laplace PSNR':>14}  {'Gaussian PSNR':>14}")
    print("-" * 65)
    for std, pl, pg in zip(noise_stds, psnr_lap, psnr_gau):
        print(f"  {noise_label(std):>12}  {pl:>14.2f}dB  {pg:>14.2f}dB")
    print("=" * 65)

    # 保存原始 PSNR 数据 CSV
    save_psnr_csv(results_all, os.path.join(args.outdir, "psnr_data.csv"))
    print("\nAll defense experiments done.")


if __name__ == "__main__":
    main()
