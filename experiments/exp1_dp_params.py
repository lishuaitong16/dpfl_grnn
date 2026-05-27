"""阶段二：DP 隐私预算对联邦学习精度的影响。

实验目标：固定 C=1.0、δ=1e-5，扫描隐私预算，
对比 Laplace 和 Gaussian 两种机制下精度随 ε 的变化。

── 两种预算模式（对齐官方 Fed.py）──────────────────────────────────
模式 A：直接指定每轮预算 --epsilon_round（默认，简单直观）
    每轮用 ε_round，总预算 ε_total = T × ε_round（简单组合）

模式 B：指定总预算 --total_epsilon + 组合定理 --dp_composition
    Simple  : ε_round = total_ε / (frac × T)
    Advanced: 高级组合（更紧）
    Renyi   : Rényi-DP 组合（Gaussian 机制专用）
    与官方 Fed.py 完全一致的参数体系

DP 参数（与官方 cal_client_sensitivity 一致）：
    sensitivity = 2 * lr * clip / N  （N 为客户端本地样本数）
    Laplace  noise_scale = sensitivity / ε_round
    Gaussian noise_std   = sqrt(2 ln(1.25/δ)) * sensitivity / ε_round
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
from fl.dp import (
    cal_client_sensitivity,
    simple_composition,
    advanced_composition,
    renyi_gaussian_composition,
)


DELTA_DP = 1e-5


# ---------------------------------------------------------------------------
# 单次实验
# ---------------------------------------------------------------------------

def run_one(mechanism, epsilon_round, rounds, clients, local_epochs,
            local_lr, batch_size, data_root, device, clip_C,
            dp_composition="none", dp_alpha=4.0):
    """运行一次 FL 实验，返回 acc_history。"""
    train_set, test_set = get_datasets(data_root)
    client_subsets      = split_iid(train_set, clients)
    client_loaders      = make_client_loaders(client_subsets, batch_size)
    test_loader         = make_test_loader(test_set)

    model = build_lenet(num_classes=10, in_channels=1, act="relu")
    _, acc_hist = federated_train(
        model, client_loaders, test_loader,
        rounds=rounds, local_epochs=local_epochs,
        local_lr=local_lr, device=device,
        mechanism=mechanism, epsilon_round=epsilon_round,
        clip_C=clip_C, delta_dp=DELTA_DP,
        dp_composition=dp_composition, dp_alpha=dp_alpha,
        verbose=True,
    )
    return acc_hist


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def parse_epsilons(s: str):
    """解析逗号分隔的 ε 列表，inf/∞/nodp 表示无 DP。"""
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part.lower() in ("inf", "∞", "nodp", "none"):
            vals.append(float("inf"))
        else:
            vals.append(float(part))
    return vals


def eps_label(eps: float) -> str:
    return "∞ (No DP)" if math.isinf(eps) else f"ε={eps:g}"


def compute_per_round_epsilon(total_epsilon, total_delta, k, composition, mechanism):
    """根据组合定理将总预算换算为每轮预算。

    k: 参与总轮次（= frac × rounds）
    返回 (epsilon_round, delta_round, dp_alpha)。
    """
    if composition == "simple":
        eps_r, delta_r = simple_composition(k, total_epsilon, total_delta)
        return float(eps_r), float(delta_r), None
    elif composition == "advanced":
        eps_r, delta_r = advanced_composition(k, total_epsilon, total_delta)
        return float(eps_r[0]), float(delta_r), None
    elif composition == "renyi":
        if mechanism != "gaussian":
            raise ValueError("Rényi 组合仅适用于 Gaussian 机制")
        alpha, eps_r = renyi_gaussian_composition(k, total_epsilon, total_delta)
        return float(eps_r), total_delta / k, float(alpha)
    else:
        raise ValueError(f"未知组合方式: {composition}")


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


def print_param_table(epsilons, rounds, clip_C, local_lr, n_per_client, delta=DELTA_DP):
    """打印 DP 参数对照表（对齐官方 sensitivity 公式）。"""
    sensitivity = cal_client_sensitivity(local_lr, clip_C, n_per_client)
    print("\n" + "=" * 80)
    print(f"  per-round DP 参数表  C={clip_C}  lr={local_lr}"
          f"  N≈{n_per_client}  sensitivity={sensitivity:.6f}  δ={delta}  T={rounds} 轮")
    print(f"  {'ε_round':>10}  {'σ_Lap(scale b)':>16}  "
          f"{'σ_Gau(std)':>14}  {'ε_total':>10}")
    print("-" * 80)
    for eps in epsilons:
        if math.isinf(eps):
            print(f"  {'∞ (No DP)':>10}  {'0':>16}  {'0':>14}  {'∞':>10}")
        else:
            b     = sensitivity / eps
            sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / eps
            print(f"  {eps:>10g}  {b:>16.6f}  {sigma:>14.6f}  {eps * rounds:>10.1f}")
    print("=" * 80)


def print_summary_table(acc_dict_lap, acc_dict_gau, epsilons):
    print("\n" + "=" * 72)
    print(f"  {'ε_round':>12}  {'Laplace Final Acc':>20}  {'Gaussian Final Acc':>20}")
    print("-" * 72)
    for eps in epsilons:
        lap_acc = acc_dict_lap[eps][-1] * 100 if eps in acc_dict_lap else float("nan")
        gau_acc = acc_dict_gau[eps][-1] * 100 if eps in acc_dict_gau else float("nan")
        print(f"  {eps_label(eps):>12}  {lap_acc:>19.2f}%  {gau_acc:>19.2f}%")
    print("=" * 72 + "\n")


def save_csv(lap_acc, gau_acc, epsilons, rounds, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment", "mechanism", "epsilon_round",
            "epsilon_total", "round", "accuracy",
        ])
        for mechanism, acc_dict in (("laplace", lap_acc), ("gaussian", gau_acc)):
            for eps in epsilons:
                eps_str       = "inf" if math.isinf(eps) else f"{eps:g}"
                eps_total_str = "inf" if math.isinf(eps) else f"{eps * rounds:g}"
                for r, acc in enumerate(acc_dict[eps], 1):
                    writer.writerow([
                        "dp_sweep", mechanism, eps_str, eps_total_str,
                        r, f"{acc:.6f}",
                    ])
    print(f"Raw data saved: {path}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="DP-FL 实验：隐私预算对 FL 精度的影响（对齐官方 dp-fl 实现）"
    )

    # ── 基础 FL 参数 ──
    p.add_argument("--clients",      type=int,   default=5)
    p.add_argument("--rounds",       type=int,   default=30)
    p.add_argument("--local_epochs", type=int,   default=1)
    p.add_argument("--local_lr",     type=float, default=0.01)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--clip_C",       type=float, default=1.0,
                   help="L2 裁剪范数（delta 敏感度上界）")

    # ── 预算模式 A：直接指定每轮 ε（默认）──
    p.add_argument("--epsilon_round", type=str, default="1,3,5,10,20,inf",
                   help="[模式A] 逗号分隔的每轮隐私预算 ε_round，inf 表示无 DP")

    # ── 预算模式 B：总预算 + 组合定理（对齐官方 Fed.py）──
    p.add_argument("--total_epsilon", type=str, default=None,
                   help="[模式B] 总隐私预算，逗号分隔，如 '10,50,100'；"
                        "需同时指定 --dp_composition")
    p.add_argument("--dp_composition", type=str, default="none",
                   choices=["none", "simple", "advanced", "renyi"],
                   help="[模式B] 组合定理：simple/advanced/renyi（none=直接用 epsilon_round）")
    p.add_argument("--client_frac",   type=float, default=1.0,
                   help="每轮参与客户端比例（影响组合轮次 k = frac × rounds）")

    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--outdir",    type=str, default="./results")
    p.add_argument("--gpu",       type=int, default=0,
                   help="使用的 GPU 编号，-1 表示 CPU")
    args = p.parse_args()

    device = (f"cuda:{args.gpu}"
              if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)

    # ── 解析 ε 列表 ──
    if args.total_epsilon is not None and args.dp_composition != "none":
        # 模式 B：总预算 → 每轮预算
        total_epsilons = parse_epsilons(args.total_epsilon)
        k = max(1, int(args.client_frac * args.rounds))
        eps_configs = []
        for t_eps in total_epsilons:
            if math.isinf(t_eps):
                eps_configs.append((float("inf"), float("inf"), None, "none"))
            else:
                eps_r, delta_r, alpha = compute_per_round_epsilon(
                    t_eps, DELTA_DP, k, args.dp_composition, "gaussian")
                eps_configs.append((t_eps, eps_r, alpha, args.dp_composition))
        epsilons = [cfg[1] for cfg in eps_configs]  # per-round ε

        print(f"\n[模式B] 总预算 → 每轮预算（组合={args.dp_composition}，k={k}）")
        print(f"  {'total_ε':>10}  {'ε_round':>12}  {'α (Rényi)':>12}")
        print("  " + "-" * 40)
        for (t_eps, eps_r, alpha, _) in eps_configs:
            t_str = "∞" if math.isinf(t_eps) else f"{t_eps:g}"
            r_str = "∞" if math.isinf(eps_r)  else f"{eps_r:.6f}"
            a_str = str(alpha) if alpha else "—"
            print(f"  {t_str:>10}  {r_str:>12}  {a_str:>12}")

        dp_alpha_map = {cfg[1]: cfg[2] for cfg in eps_configs}
    else:
        # 模式 A：直接使用每轮 ε
        epsilons     = parse_epsilons(args.epsilon_round)
        dp_alpha_map = {eps: None for eps in epsilons}
        print(f"\n[模式A] 每轮预算 ε_round 列表：{[eps_label(e) for e in epsilons]}")

    # 估算每客户端样本数（用于参数表）
    n_per_client = 60000 // args.clients  # MNIST 训练集 60000 张

    print_param_table(epsilons, args.rounds, args.clip_C,
                      args.local_lr, n_per_client)

    # ── 扫描实验 ──
    print("\n===== Laplace DP-FL: ε_round sweep =====")
    lap_acc = {}
    for eps in epsilons:
        mech = "none" if math.isinf(eps) else "laplace"
        print(f"\n-- {eps_label(eps)} ({mech}) --")
        lap_acc[eps] = run_one(
            mechanism=mech, epsilon_round=eps,
            rounds=args.rounds, clients=args.clients,
            local_epochs=args.local_epochs, local_lr=args.local_lr,
            batch_size=args.batch_size, data_root=args.data_root,
            device=device, clip_C=args.clip_C,
            dp_composition="none",  # Laplace 不用 Rényi 组合
        )

    print("\n===== Gaussian DP-FL: ε_round sweep =====")
    gau_acc = {}
    for eps in epsilons:
        mech  = "none" if math.isinf(eps) else "gaussian"
        alpha = dp_alpha_map.get(eps)
        comp  = args.dp_composition if alpha is not None else "none"
        alpha = alpha if alpha is not None else 4.0
        print(f"\n-- {eps_label(eps)} ({mech}) --")
        gau_acc[eps] = run_one(
            mechanism=mech, epsilon_round=eps,
            rounds=args.rounds, clients=args.clients,
            local_epochs=args.local_epochs, local_lr=args.local_lr,
            batch_size=args.batch_size, data_root=args.data_root,
            device=device, clip_C=args.clip_C,
            dp_composition=comp, dp_alpha=alpha,
        )

    # ── 绘图 ──
    plot_eps_sweep(
        lap_acc, epsilons, args.rounds,
        f"Laplace DP-FL — Accuracy vs Round"
        f" (N={args.clients}, C={args.clip_C}, per-round ε, δ={DELTA_DP})",
        os.path.join(args.outdir, "dp_laplace_acc.png"),
    )
    plot_eps_sweep(
        gau_acc, epsilons, args.rounds,
        f"Gaussian DP-FL — Accuracy vs Round"
        f" (N={args.clients}, C={args.clip_C}, per-round ε, δ={DELTA_DP})",
        os.path.join(args.outdir, "dp_gaussian_acc.png"),
    )

    print_summary_table(lap_acc, gau_acc, epsilons)
    save_csv(lap_acc, gau_acc, epsilons, args.rounds,
             os.path.join(args.outdir, "dp_acc_data.csv"))
    print("\nAll experiments done.")


if __name__ == "__main__":
    main()
