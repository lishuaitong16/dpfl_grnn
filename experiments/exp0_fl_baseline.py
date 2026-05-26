"""阶段一：联邦学习基线（无差分隐私）。

运行：
    python -m experiments.exp0_fl_baseline
（在 dpfl_grnn 目录下运行，确保能 import 到 models/ 和 fl/）

产出：
    - 命令行打印每轮测试精度；
    - results/baseline_acc.png：精度 vs 通信轮数曲线。
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# 允许从项目根目录直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import build_lenet
from fl.data import get_datasets, split_iid, make_client_loaders, make_test_loader
from fl.train import federated_train


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--out", type=str, default="./results/baseline_acc.png")
    p.add_argument("--gpu", type=int, default=0, help="使用的 GPU 编号，-1 表示 CPU")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_set, test_set = get_datasets(args.data_root)
    client_subsets = split_iid(train_set, args.clients)
    client_loaders = make_client_loaders(client_subsets, args.batch_size)
    test_loader = make_test_loader(test_set)

    # FL 基线使用 ReLU，保证标准 FedAvg 稳定收敛。
    model = build_lenet(num_classes=10, in_channels=1, act="relu")

    model, acc_hist = federated_train(
        model, client_loaders, test_loader,
        rounds=args.rounds, local_epochs=args.local_epochs,
        local_lr=args.local_lr, device=device, mechanism="none",
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(acc_hist) + 1), [a * 100 for a in acc_hist], marker="o", ms=3)
    plt.xlabel("Communication Round")
    plt.ylabel("Test Accuracy (%)")
    plt.title(f"FedAvg Baseline (N={args.clients}, No DP)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nFinal accuracy: {acc_hist[-1]*100:.2f}%")
    print(f"Curve saved: {args.out}")


if __name__ == "__main__":
    main()
