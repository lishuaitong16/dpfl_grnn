"""阶段二+三：差分隐私参数对联邦学习性能的影响。

扫描内容：
  1. 主实验：固定 T=30, N=5, C=1.0，扫 ε_total ∈ {0.1, 0.5, 1, 5, 10, ∞}
             对拉普拉斯和高斯各跑一遍，画"精度 vs 轮数"曲线。
  2. 副实验：固定 ε_total=1, T=30, N=5，扫 C ∈ {0.5, 1, 4}。
  3. 简单组合定理汇总表：打印 T=30 时各 ε_total 对应的 ε_round。

运行：
    python -m experiments.exp1_dp_params
（在 dpfl_grnn 目录下运行）
"""

import os
import sys
import csv
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import build_lenet
from fl.data import get_datasets, split_iid, make_client_loaders, make_test_loader
from fl.train import federated_train
from fl.dp import per_round_epsilon


def run_one(mechanism, total_eps, clip_C, rounds, clients, local_epochs,
            local_lr, batch_size, data_root, device):
    """跑一次联邦学习，返回 acc_history。"""
    train_set, test_set = get_datasets(data_root)
    client_subsets = split_iid(train_set, clients)
    client_loaders = make_client_loaders(client_subsets, batch_size)
    test_loader = make_test_loader(test_set)

    # ReLU：sigmoid 在 DP 噪声扰动权重后会饱和，导致 delta 趋近零、FL 无法收敛。
    # 注：GRNN 攻击（exp2/exp3）需要 sigmoid 激活以支持二阶梯度反传，两者使用不同激活函数
    # 是技术约束，报告中需说明此差异。
    model = build_lenet(num_classes=10, in_channels=1, act="relu")
    _, acc_hist = federated_train(
        model, client_loaders, test_loader,
        rounds=rounds, local_epochs=local_epochs,
        local_lr=local_lr, device=device,
        mechanism=mechanism,
        total_epsilon=total_eps,
        clip_C=clip_C,
        verbose=True,
    )
    return acc_hist


def plot_eps_sweep(acc_dict, epsilon_vals, rounds, title, save_path):
    """画"精度 vs 轮数"曲线，每条线对应一个 ε_total。"""
    plt.figure(figsize=(7, 5))
    xs = list(range(1, rounds + 1))
    for eps in epsilon_vals:
        label = f"eps_total={eps}" if eps is not None else "No DP (inf)"
        acc = [a * 100 for a in acc_dict[eps]]
        plt.plot(xs, acc, marker="o", ms=3, label=label)
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curve saved: {save_path}")


def plot_clip_sweep(acc_dict, clip_vals, rounds, title, save_path):
    """画"精度 vs 轮数"曲线，每条线对应一个裁剪范数 C。"""
    plt.figure(figsize=(7, 5))
    xs = list(range(1, rounds + 1))
    for C in clip_vals:
        acc = [a * 100 for a in acc_dict[C]]
        plt.plot(xs, acc, marker="o", ms=3, label=f"C={C}")
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curve saved: {save_path}")


def print_composition_table(epsilon_vals, T):
    """打印简单组合定理预算分配表。"""
    print("\n" + "=" * 55)
    print(f"  Composition Theorem Budget Table (T={T} rounds)")
    print(f"  {'eps_total':>12}  {'eps_round = eps_total/T':>22}")
    print("-" * 55)
    for eps in epsilon_vals:
        if eps is None:
            print(f"  {'inf (No DP)':>12}  {'inf':>22}")
        else:
            eps_r = per_round_epsilon(eps, T)
            print(f"  {eps:>12.1f}  {eps_r:>22.4f}")
    print("=" * 55 + "\n")


def print_summary_table(acc_dict_lap, acc_dict_gau, epsilon_vals):
    """打印最终精度汇总表。"""
    print("\n" + "=" * 60)
    print(f"  {'eps_total':>12}  {'Laplace Final Acc':>18}  {'Gaussian Final Acc':>18}")
    print("-" * 60)
    for eps in epsilon_vals:
        key = eps
        lap_acc = acc_dict_lap[key][-1] * 100 if key in acc_dict_lap else float("nan")
        gau_acc = acc_dict_gau[key][-1] * 100 if key in acc_dict_gau else float("nan")
        eps_str = "inf (No DP)" if eps is None else str(eps)
        print(f"  {eps_str:>12}  {lap_acc:>18.2f}%  {gau_acc:>18.2f}%")
    print("=" * 60 + "\n")


def parse_epsilons(s):
    """解析逗号分隔的 ε 列表，'inf'/'none' 表示无 DP（ε=∞）。
    示例: "0.1,0.5,1.0,5.0,10.0,inf" -> [0.1, 0.5, 1.0, 5.0, 10.0, None]
    """
    result = []
    for part in s.split(","):
        part = part.strip()
        if part.lower() in ("none", "inf", "nodp"):
            result.append(None)
        else:
            result.append(float(part))
    return result


def save_csv(lap_eps_acc, gau_eps_acc, clip_acc, eps_list, clip_list, clip_eps, path):
    """保存所有精度数据到 CSV，供后续自定义画图。"""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "mechanism", "epsilon", "clip_C", "round", "accuracy"])
        for eps in eps_list:
            eps_str = "inf" if eps is None else str(eps)
            for r, acc in enumerate(lap_eps_acc[eps], 1):
                writer.writerow(["eps_sweep", "laplace", eps_str, "", r, f"{acc:.6f}"])
        for eps in eps_list:
            eps_str = "inf" if eps is None else str(eps)
            for r, acc in enumerate(gau_eps_acc[eps], 1):
                writer.writerow(["eps_sweep", "gaussian", eps_str, "", r, f"{acc:.6f}"])
        for C in clip_list:
            for r, acc in enumerate(clip_acc[C], 1):
                writer.writerow(["clip_sweep", "laplace", str(clip_eps), str(C), r, f"{acc:.6f}"])
    print(f"Raw data saved: {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clients",      type=int,   default=5)
    p.add_argument("--rounds",       type=int,   default=30)
    p.add_argument("--local_epochs", type=int,   default=1)
    p.add_argument("--local_lr",     type=float, default=0.01)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--clip_C",       type=float, default=1.0,
                   help="主实验（ε 扫描）固定裁剪范数")
    p.add_argument("--epsilons",     type=str,   default="0.1,0.5,1.0,5.0,10.0,inf",
                   help="ε_total 扫描列表，逗号分隔，'inf' 表示无 DP")
    p.add_argument("--clip_values",  type=str,   default="0.5,1.0,4.0",
                   help="副实验裁剪范数 C 列表，逗号分隔")
    p.add_argument("--clip_eps",     type=float, default=1.0,
                   help="副实验（C 扫描）固定 ε_total")
    p.add_argument("--data_root",    type=str,   default="./data")
    p.add_argument("--outdir",       type=str,   default="./results")
    p.add_argument("--gpu",          type=int,   default=0,
                   help="使用的 GPU 编号，-1 表示 CPU")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    eps_list  = parse_epsilons(args.epsilons)
    clip_list = [float(x) for x in args.clip_values.split(",")]

    # ------------------------------------------------------------------
    # 打印简单组合定理预算分配表
    # ------------------------------------------------------------------
    print_composition_table(eps_list, args.rounds)

    # ------------------------------------------------------------------
    # 主实验：扫 ε_total，对拉普拉斯和高斯各跑一遍
    # ------------------------------------------------------------------
    print("\n===== Main experiment: sweeping eps_total, Laplace =====")
    lap_eps_acc = {}
    for eps in eps_list:
        mech = "none" if eps is None else "laplace"
        print(f"\n-- ε_total={eps} ({mech}) --")
        lap_eps_acc[eps] = run_one(
            mechanism=mech, total_eps=eps, clip_C=args.clip_C,
            rounds=args.rounds, clients=args.clients,
            local_epochs=args.local_epochs, local_lr=args.local_lr,
            batch_size=args.batch_size, data_root=args.data_root, device=device,
        )

    print("\n===== Main experiment: sweeping eps_total, Gaussian =====")
    gau_eps_acc = {}
    for eps in eps_list:
        mech = "none" if eps is None else "gaussian"
        print(f"\n-- ε_total={eps} ({mech}) --")
        gau_eps_acc[eps] = run_one(
            mechanism=mech, total_eps=eps, clip_C=args.clip_C,
            rounds=args.rounds, clients=args.clients,
            local_epochs=args.local_epochs, local_lr=args.local_lr,
            batch_size=args.batch_size, data_root=args.data_root, device=device,
        )

    plot_eps_sweep(
        lap_eps_acc, eps_list, args.rounds,
        f"Laplace DP — Accuracy vs Round (N={args.clients}, C={args.clip_C})",
        os.path.join(args.outdir, "dp_laplace_acc.png"),
    )
    plot_eps_sweep(
        gau_eps_acc, eps_list, args.rounds,
        f"Gaussian DP — Accuracy vs Round (N={args.clients}, C={args.clip_C})",
        os.path.join(args.outdir, "dp_gaussian_acc.png"),
    )

    print_summary_table(lap_eps_acc, gau_eps_acc, eps_list)

    # ------------------------------------------------------------------
    # 副实验：固定 ε_total，扫裁剪范数 C（拉普拉斯）
    # ------------------------------------------------------------------
    print(f"\n===== Sub-experiment: sweeping clip norm C (eps_total={args.clip_eps}, Laplace) =====")
    clip_acc = {}
    for C in clip_list:
        print(f"\n-- C={C} --")
        clip_acc[C] = run_one(
            mechanism="laplace", total_eps=args.clip_eps, clip_C=C,
            rounds=args.rounds, clients=args.clients,
            local_epochs=args.local_epochs, local_lr=args.local_lr,
            batch_size=args.batch_size, data_root=args.data_root, device=device,
        )

    plot_clip_sweep(
        clip_acc, clip_list, args.rounds,
        f"Effect of Clip Norm C on Accuracy (eps_total={args.clip_eps}, Laplace, N={args.clients})",
        os.path.join(args.outdir, "dp_clip_acc.png"),
    )

    # ------------------------------------------------------------------
    # 保存原始数据 CSV
    # ------------------------------------------------------------------
    save_csv(lap_eps_acc, gau_eps_acc, clip_acc, eps_list, clip_list,
             args.clip_eps, os.path.join(args.outdir, "dp_acc_data.csv"))

    print("\nAll experiments done.")


if __name__ == "__main__":
    main()
