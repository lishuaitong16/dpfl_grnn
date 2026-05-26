"""阶段二：物理噪声规模对联邦学习性能的影响。

实验目标：不做梯度裁剪，不计算严格 DP 敏感度，直接在标准 FedAvg 的客户端
模型更新 delta 上注入不同标准差的零均值噪声，观察训练精度随噪声规模变化。
"""

import os
import sys
import csv
import math
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import build_lenet
from fl.data import get_datasets, split_iid, make_client_loaders, make_test_loader
from fl.train import federated_train


def run_one(mechanism, noise_std, rounds, clients, local_epochs,
            local_lr, batch_size, data_root, device):
    """Run one FL experiment and return acc_history."""
    train_set, test_set = get_datasets(data_root)
    client_subsets = split_iid(train_set, clients)
    client_loaders = make_client_loaders(client_subsets, batch_size)
    test_loader = make_test_loader(test_set)

    # FL accuracy experiments use ReLU for stable convergence.
    model = build_lenet(num_classes=10, in_channels=1, act="relu")
    _, acc_hist = federated_train(
        model, client_loaders, test_loader,
        rounds=rounds, local_epochs=local_epochs,
        local_lr=local_lr, device=device,
        mechanism=mechanism, noise_std=noise_std,
        verbose=True,
    )
    return acc_hist


def parse_noise_stds(s):
    """Parse comma-separated noise stds. 0/none/inf/nonoise mean no noise."""
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part.lower() in ("none", "inf", "nodp", "nonoise"):
            vals.append(0.0)
        else:
            vals.append(float(part))
    return vals


def noise_label(std):
    return "No noise" if std == 0 else f"std={std:g}"


def plot_noise_sweep(acc_dict, noise_stds, rounds, title, save_path):
    plt.figure(figsize=(7, 5))
    xs = list(range(1, rounds + 1))
    for std in noise_stds:
        acc = [a * 100 for a in acc_dict[std]]
        plt.plot(xs, acc, marker="o", ms=3, label=noise_label(std))
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curve saved: {save_path}")


def print_composition_table(noise_stds, rounds, local_epochs, batch_size, clients,
                            train_size=60000, delta=1e-5):
    """用简单组合定理把 noise_std 换算成对应的 ε。

    C 取梯度经验范数 ≈ 4.3（无裁剪，近似敏感度）。
    T = rounds × steps_per_round，steps_per_round = local_epochs × (每客户端样本数/batch_size)。
    """
    C = 1.0
    steps_per_round = local_epochs * (train_size // clients // batch_size)
    T = rounds * steps_per_round

    print("\n" + "=" * 78)
    print(f"  简单组合定理换算  C={C}  δ={delta}")
    print(f"  T = {rounds} 轮 × {steps_per_round} 步/轮 = {T} 总步数")
    print(f"  {'noise_std':>10}  {'ε_Lap/步':>10}  {'ε_Lap总':>10}  {'ε_Gau/步':>10}  {'ε_Gau总':>10}")
    print("-" * 78)
    for std in noise_stds:
        if std == 0:
            print(f"  {'0 (无噪声)':>10}  {'∞':>10}  {'∞':>10}  {'∞':>10}  {'∞':>10}")
        else:
            eps_lap_step = C * math.sqrt(2) / std
            eps_gau_step = C * math.sqrt(2 * math.log(1.25 / delta)) / std
            print(f"  {std:>10g}"
                  f"  {eps_lap_step:>10.2f}  {eps_lap_step * T:>10.1f}"
                  f"  {eps_gau_step:>10.2f}  {eps_gau_step * T:>10.1f}")
    print("=" * 78)


def print_summary_table(acc_dict_lap, acc_dict_gau, noise_stds):
    print("\n" + "=" * 70)
    print(f"  {'noise_std':>12}  {'Laplace Final Acc':>18}  {'Gaussian Final Acc':>18}")
    print("-" * 70)
    for std in noise_stds:
        lap_acc = acc_dict_lap[std][-1] * 100 if std in acc_dict_lap else float("nan")
        gau_acc = acc_dict_gau[std][-1] * 100 if std in acc_dict_gau else float("nan")
        print(f"  {noise_label(std):>12}  {lap_acc:>18.2f}%  {gau_acc:>18.2f}%")
    print("=" * 70 + "\n")


def save_csv(lap_acc, gau_acc, noise_stds, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "mechanism", "noise_std", "round", "accuracy"])
        for mechanism, acc_dict in (("laplace", lap_acc), ("gaussian", gau_acc)):
            for std in noise_stds:
                for r, acc in enumerate(acc_dict[std], 1):
                    writer.writerow(["noise_sweep", mechanism, f"{std:g}", r, f"{acc:.6f}"])
    print(f"Raw data saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clients",      type=int,   default=5)
    p.add_argument("--rounds",       type=int,   default=30)
    p.add_argument("--local_epochs", type=int,   default=1)
    p.add_argument("--local_lr",     type=float, default=0.01)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--noise_stds",   type=str,   default="1e-1,1e-2,1e-3,1e-4,0",
                   help="直接注入到梯度上的噪声标准差列表，0 表示无噪声")
    p.add_argument("--data_root",    type=str,   default="./data")
    p.add_argument("--outdir",       type=str,   default="./results")
    p.add_argument("--gpu",          type=int,   default=0,
                   help="使用的 GPU 编号，-1 表示 CPU")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    noise_stds = parse_noise_stds(args.noise_stds)
    print("\nNoise std sweep:", ", ".join(noise_label(s) for s in noise_stds))

    print("\n===== Main experiment: sweeping physical noise std, Laplace =====")
    lap_acc = {}
    for std in noise_stds:
        mech = "none" if std == 0 else "laplace"
        print(f"\n-- noise_std={std:g} ({mech}) --")
        lap_acc[std] = run_one(
            mechanism=mech, noise_std=std,
            rounds=args.rounds, clients=args.clients,
            local_epochs=args.local_epochs, local_lr=args.local_lr,
            batch_size=args.batch_size, data_root=args.data_root, device=device,
        )

    print("\n===== Main experiment: sweeping physical noise std, Gaussian =====")
    gau_acc = {}
    for std in noise_stds:
        mech = "none" if std == 0 else "gaussian"
        print(f"\n-- noise_std={std:g} ({mech}) --")
        gau_acc[std] = run_one(
            mechanism=mech, noise_std=std,
            rounds=args.rounds, clients=args.clients,
            local_epochs=args.local_epochs, local_lr=args.local_lr,
            batch_size=args.batch_size, data_root=args.data_root, device=device,
        )

    plot_noise_sweep(
        lap_acc, noise_stds, args.rounds,
        f"Laplace Physical Noise - Accuracy vs Round (N={args.clients})",
        os.path.join(args.outdir, "dp_laplace_acc.png"),
    )
    plot_noise_sweep(
        gau_acc, noise_stds, args.rounds,
        f"Gaussian Physical Noise - Accuracy vs Round (N={args.clients})",
        os.path.join(args.outdir, "dp_gaussian_acc.png"),
    )

    print_summary_table(lap_acc, gau_acc, noise_stds)
    print_composition_table(noise_stds, args.rounds, args.local_epochs,
                            args.batch_size, args.clients)
    save_csv(lap_acc, gau_acc, noise_stds, os.path.join(args.outdir, "dp_acc_data.csv"))
    print("\nAll experiments done.")


if __name__ == "__main__":
    main()
