"""阶段二：DP 隐私预算 ε 对联邦学习精度的影响。

实验目标：固定 C=1.0、δ=1e-5，扫描 ε_total ∈ {1,5,10,50,100,∞}，
对比 Laplace 和 Gaussian 两种机制下精度随 ε_total 的变化，
量化差分隐私的"隐私-效用权衡"。

DP 方案：per-round delta DP（顺序合成定理）
    ε_round = ε_total / T（T = 通信轮数）
    Laplace  噪声标准差 σ = C√2 / ε_round
    Gaussian 噪声标准差 σ = C√(2 ln(1.25/δ)) / ε_round
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
from fl.dp import noise_std_from_epsilon


CLIP_C   = 1.0
DELTA_DP = 1e-5


def run_one(mechanism, total_epsilon, rounds, clients, local_epochs,
            local_lr, batch_size, data_root, device):
    """Run one FL experiment and return acc_history."""
    train_set, test_set = get_datasets(data_root)
    client_subsets = split_iid(train_set, clients)
    client_loaders = make_client_loaders(client_subsets, batch_size)
    test_loader = make_test_loader(test_set)

    model = build_lenet(num_classes=10, in_channels=1, act="relu")
    _, acc_hist = federated_train(
        model, client_loaders, test_loader,
        rounds=rounds, local_epochs=local_epochs,
        local_lr=local_lr, device=device,
        mechanism=mechanism, total_epsilon=total_epsilon,
        clip_C=CLIP_C, delta_dp=DELTA_DP,
        verbose=True,
    )
    return acc_hist


def parse_epsilons(s):
    """解析逗号分隔的 ε 列表，inf/∞/nodp 表示无 DP。"""
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part.lower() in ("inf", "∞", "nodp", "none"):
            vals.append(float("inf"))
        else:
            vals.append(float(part))
    return vals


def eps_label(eps):
    return "∞ (No DP)" if math.isinf(eps) else f"ε={eps:g}"


def plot_eps_sweep(acc_dict, epsilons, rounds, title, save_path):
    plt.figure(figsize=(7, 5))
    xs = list(range(1, rounds + 1))
    for eps in epsilons:
        acc = [a * 100 for a in acc_dict[eps]]
        plt.plot(xs, acc, marker="o", ms=3, label=eps_label(eps))
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curve saved: {save_path}")


def print_composition_table(epsilons, rounds, delta=DELTA_DP):
    """打印 DP 参数组合表：ε_total → ε_round → σ_Lap / σ_Gau。"""
    print("\n" + "=" * 72)
    print(f"  per-round delta DP 参数表  C={CLIP_C}  δ={delta}  T={rounds} 轮")
    print(f"  {'ε_total':>10}  {'ε_round':>10}  {'σ_Lap':>10}  {'σ_Gau':>10}")
    print("-" * 72)
    for eps in epsilons:
        if math.isinf(eps):
            print(f"  {'∞ (No DP)':>10}  {'∞':>10}  {'0':>10}  {'0':>10}")
        else:
            eps_r = eps / rounds
            sigma_lap = noise_std_from_epsilon("laplace",  eps_r, CLIP_C, delta)
            sigma_gau = noise_std_from_epsilon("gaussian", eps_r, CLIP_C, delta)
            print(f"  {eps:>10g}  {eps_r:>10.4f}  {sigma_lap:>10.4f}  {sigma_gau:>10.4f}")
    print("=" * 72)


def print_summary_table(acc_dict_lap, acc_dict_gau, epsilons):
    print("\n" + "=" * 72)
    print(f"  {'ε_total':>12}  {'Laplace Final Acc':>20}  {'Gaussian Final Acc':>20}")
    print("-" * 72)
    for eps in epsilons:
        lap_acc = acc_dict_lap[eps][-1] * 100 if eps in acc_dict_lap else float("nan")
        gau_acc = acc_dict_gau[eps][-1] * 100 if eps in acc_dict_gau else float("nan")
        print(f"  {eps_label(eps):>12}  {lap_acc:>19.2f}%  {gau_acc:>19.2f}%")
    print("=" * 72 + "\n")


def save_csv(lap_acc, gau_acc, epsilons, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "mechanism", "epsilon_total", "round", "accuracy"])
        for mechanism, acc_dict in (("laplace", lap_acc), ("gaussian", gau_acc)):
            for eps in epsilons:
                eps_str = "inf" if math.isinf(eps) else f"{eps:g}"
                for r, acc in enumerate(acc_dict[eps], 1):
                    writer.writerow(["dp_sweep", mechanism, eps_str, r, f"{acc:.6f}"])
    print(f"Raw data saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clients",      type=int,   default=5)
    p.add_argument("--rounds",       type=int,   default=30)
    p.add_argument("--local_epochs", type=int,   default=1)
    p.add_argument("--local_lr",     type=float, default=0.01)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--epsilons",     type=str,   default="1,5,10,50,100,inf",
                   help="逗号分隔的全局 ε_total 列表，inf 表示无 DP")
    p.add_argument("--data_root",    type=str,   default="./data")
    p.add_argument("--outdir",       type=str,   default="./results")
    p.add_argument("--gpu",          type=int,   default=0,
                   help="使用的 GPU 编号，-1 表示 CPU")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    epsilons = parse_epsilons(args.epsilons)
    print("\nε_total sweep:", ", ".join(eps_label(e) for e in epsilons))
    print_composition_table(epsilons, args.rounds)

    print("\n===== Main experiment: ε sweep, Laplace =====")
    lap_acc = {}
    for eps in epsilons:
        mech = "none" if math.isinf(eps) else "laplace"
        print(f"\n-- {eps_label(eps)} ({mech}) --")
        lap_acc[eps] = run_one(
            mechanism=mech, total_epsilon=eps,
            rounds=args.rounds, clients=args.clients,
            local_epochs=args.local_epochs, local_lr=args.local_lr,
            batch_size=args.batch_size, data_root=args.data_root, device=device,
        )

    print("\n===== Main experiment: ε sweep, Gaussian =====")
    gau_acc = {}
    for eps in epsilons:
        mech = "none" if math.isinf(eps) else "gaussian"
        print(f"\n-- {eps_label(eps)} ({mech}) --")
        gau_acc[eps] = run_one(
            mechanism=mech, total_epsilon=eps,
            rounds=args.rounds, clients=args.clients,
            local_epochs=args.local_epochs, local_lr=args.local_lr,
            batch_size=args.batch_size, data_root=args.data_root, device=device,
        )

    plot_eps_sweep(
        lap_acc, epsilons, args.rounds,
        f"Laplace DP-FL — Accuracy vs Round (N={args.clients}, C={CLIP_C})",
        os.path.join(args.outdir, "dp_laplace_acc.png"),
    )
    plot_eps_sweep(
        gau_acc, epsilons, args.rounds,
        f"Gaussian DP-FL — Accuracy vs Round (N={args.clients}, C={CLIP_C})",
        os.path.join(args.outdir, "dp_gaussian_acc.png"),
    )

    print_summary_table(lap_acc, gau_acc, epsilons)
    save_csv(lap_acc, gau_acc, epsilons, os.path.join(args.outdir, "dp_acc_data.csv"))
    print("\nAll experiments done.")


if __name__ == "__main__":
    main()
