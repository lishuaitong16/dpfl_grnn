"""阶段五：用差分隐私噪声防御 GRNN 梯度攻击。

对齐官方实现（dp-fl/GRNN.py + client/dp_mechanism.py）：
  1. 直接用 torch.autograd.grad() 计算单样本梯度，不经 delta → ÷lr 换算。
  2. 噪声直接加到梯度向量上（与 GRNN.py 中处理方式一致）。
  3. 敏感度公式（与官方 cal_client_sensitivity 完全一致）：
         sensitivity = 2 * lr * clip / batchsize
     batchsize=1（GRNN 攻击单样本场景）。
  4. 不做梯度裁剪（官方 GRNN.py 中亦无显式裁剪）。

DP 参数与 exp1 对齐：clip_C=1.0，δ=1e-5，ε_round ∈ {10,25,50,75,100,∞}。
"""

import os
import sys
import csv
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import build_lenet
from fl.data import get_datasets, make_test_loader, denormalize
from fl.dp import (
    cal_client_sensitivity,
    laplace_mechanism,
    gaussian_mechanism,
)
from attack.run_attack import compute_true_gradient, grnn_attack, psnr
from torch.utils.data import DataLoader


DELTA_DP = 1e-5


# ---------------------------------------------------------------------------
# DP 噪声：直接加到梯度向量上（对齐官方 GRNN.py）
# ---------------------------------------------------------------------------

def add_dp_noise_to_gradient(
    true_grad:  torch.Tensor,
    mechanism:  str,
    epsilon:    float,
    local_lr:   float,
    clip_C:     float,
    delta:      float = DELTA_DP,
) -> tuple:
    """把 DP 噪声加到梯度向量上（官方方式）。

    敏感度公式（对齐 cal_client_sensitivity，batchsize=1）：
        sensitivity = 2 * local_lr * clip_C / 1

    Args:
        true_grad : 真实梯度向量（由 compute_true_gradient 得到的 1-D tensor）。
        mechanism : "laplace" 或 "gaussian"。
        epsilon   : 每轮隐私预算 ε_round。
        local_lr  : 本地学习率（进入 sensitivity 公式）。
        clip_C    : 裁剪范数（进入 sensitivity 公式；此处不实际裁剪梯度）。
        delta     : (ε,δ)-DP 的 δ（Gaussian only）。

    Returns:
        noisy_grad : 加噪后的梯度向量（与 true_grad 同 device）。
        stats      : 包含 grad_norm / noise_norm / snr 的字典。
    """
    # sensitivity = 2 * lr * clip / N，N=1（单样本攻击）
    sensitivity = cal_client_sensitivity(local_lr, clip_C, dataset_size=1)

    size = (true_grad.shape[0],)
    if mechanism == "laplace":
        noise_np = laplace_mechanism(epsilon, sensitivity, size)
    elif mechanism == "gaussian":
        noise_np = gaussian_mechanism(epsilon, delta, sensitivity, size)
    else:
        raise ValueError(f"unknown mechanism: {mechanism}")

    noise = torch.from_numpy(noise_np).float().to(true_grad.device)
    noisy_grad = true_grad + noise

    grad_norm  = true_grad.norm().item()
    noise_norm = noise.norm().item()
    stats = {
        "grad_norm":  grad_norm,
        "noise_norm": noise_norm,
        "snr":        grad_norm / noise_norm if noise_norm > 0 else float("inf"),
    }
    return noisy_grad, stats


# ---------------------------------------------------------------------------
# 绘图 / CSV 工具
# ---------------------------------------------------------------------------

def eps_label(eps: float) -> str:
    return "No DP" if math.isinf(eps) else f"ε_r={eps:g}"


def save_defense_grid(real_img_vis, results_by_eps, epsilons, mech_name, save_path):
    n_rows = len(epsilons)
    fig, axes = plt.subplots(n_rows, 2, figsize=(5, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    for row, eps in enumerate(epsilons):
        fake_vis = results_by_eps[eps]["fake_vis"][0]
        ps       = results_by_eps[eps]["psnr"]
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


def save_psnr_bar(psnr_lap, psnr_gau, epsilons, clip_C, save_path):
    labels = [eps_label(e) for e in epsilons]
    x      = np.arange(len(labels))
    width  = 0.35
    plt.figure(figsize=(8, 5))
    bars_lap = plt.bar(x - width / 2, psnr_lap, width,
                       label="Laplace",  color="steelblue")
    bars_gau = plt.bar(x + width / 2, psnr_gau, width,
                       label="Gaussian", color="coral")
    for bar in list(bars_lap) + list(bars_gau):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                 f"{h:.1f}", ha="center", fontsize=8)
    plt.xticks(x, labels, fontsize=9)
    plt.xlabel("Per-round Privacy Budget  ε_round")
    plt.ylabel("PSNR (dB)  [higher = more similar to original]")
    plt.title("PSNR of GRNN Recovered Images under DP Defense\n"
              f"(C={clip_C}, δ={DELTA_DP}, lower PSNR = better protection)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"PSNR bar chart saved: {save_path}")


def save_psnr_csv(results_all, epsilons, clip_C, local_lr, path):
    sensitivity = cal_client_sensitivity(local_lr, clip_C, dataset_size=1)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "mechanism", "epsilon_round", "sensitivity",
            "noise_scale_lap", "noise_std_gau",
            "grad_norm", "noise_norm", "snr", "psnr_db",
        ])
        for mechanism, results_by_eps in results_all.items():
            for eps, data in results_by_eps.items():
                eps_str = "inf" if math.isinf(eps) else f"{eps:g}"
                if math.isinf(eps):
                    b_str, s_str = "0", "0"
                else:
                    b_str = f"{sensitivity / eps:.6f}"               # Laplace scale
                    import math as _m
                    sigma = _m.sqrt(2 * _m.log(1.25 / DELTA_DP)) * sensitivity / eps
                    s_str = f"{sigma:.6f}"                            # Gaussian std
                writer.writerow([
                    mechanism, eps_str, f"{sensitivity:.6f}",
                    b_str, s_str,
                    f"{data.get('grad_norm',  float('nan')):.4f}",
                    f"{data.get('noise_norm', float('nan')):.4f}",
                    f"{data.get('snr',        float('nan')):.4f}",
                    f"{data['psnr']:.4f}",
                ])
    print(f"PSNR data saved: {path}")


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_epsilons(s: str):
    result = []
    for part in s.split(","):
        part = part.strip()
        if part.lower() in ("inf", "∞", "nodp", "none"):
            result.append(float("inf"))
        else:
            result.append(float(part))
    return result


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epsilon_round", type=str,   default="10,25,50,75,100,inf",
                   help="逗号分隔的每轮隐私预算 ε_round，inf 表示无 DP（与 exp1 保持一致）")
    p.add_argument("--clip_C",        type=float, default=1.0,
                   help="L2 裁剪范数（进入 sensitivity = 2·lr·C/1，与 exp1 保持一致）")
    p.add_argument("--local_lr",      type=float, default=0.01,
                   help="本地学习率（进入 sensitivity，与 FL 训练一致）")
    p.add_argument("--iterations",    type=int,   default=2000,
                   help="GRNN 攻击迭代次数")
    p.add_argument("--train_iters",   type=int,   default=0,
                   help="攻击前在训练集上预训练模型的步数（0=随机初始化，最易攻击）")
    p.add_argument("--tv_alpha",      type=float, default=1e-3)
    p.add_argument("--data_root",     type=str,   default="./data")
    p.add_argument("--outdir",        type=str,   default="./results/defense")
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--gpu",           type=int,   default=0)
    args = p.parse_args()

    device = (f"cuda:{args.gpu}"
              if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    epsilons    = parse_epsilons(args.epsilon_round)
    sensitivity = cal_client_sensitivity(args.local_lr, args.clip_C, dataset_size=1)

    print(f"\nε_round sweep : {', '.join(eps_label(e) for e in epsilons)}")
    print(f"clip_C={args.clip_C}  local_lr={args.local_lr}  "
          f"sensitivity=cal_client_sensitivity({args.local_lr},{args.clip_C},1)={sensitivity:.4f}"
          f"  δ={DELTA_DP}\n")

    # ---- 模型（Sigmoid：GRNN 二阶反传必需）----
    model = build_lenet(num_classes=10, in_channels=1, act="sigmoid").to(device)

    train_set, _ = get_datasets(args.data_root)

    # ---- 取单张真实图像 ----
    x_real, y_real = next(iter(DataLoader(train_set, batch_size=1, shuffle=True)))
    x_real, y_real = x_real.to(device), y_real.to(device)

    # ---- 可选预训练 ----
    if args.train_iters > 0:
        opt      = torch.optim.SGD(model.parameters(), lr=0.01)
        crit     = nn.CrossEntropyLoss()
        model.train()
        loader_it = iter(DataLoader(train_set, batch_size=64, shuffle=True))
        for step in range(args.train_iters):
            try:
                xb, yb = next(loader_it)
            except StopIteration:
                loader_it = iter(DataLoader(train_set, batch_size=64, shuffle=True))
                xb, yb = next(loader_it)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
            if (step + 1) % 100 == 0:
                print(f"  Pre-training step {step+1}/{args.train_iters}")
        model.eval()
        print(f"Pre-training done ({args.train_iters} steps)\n")

    # ---- 打印真实梯度信息 ----
    true_grad = compute_true_gradient(model, x_real, y_real, device)
    print(f"True label : {y_real.item()}")
    print(f"Gradient   : dim={true_grad.numel()}  norm={true_grad.norm().item():.4f}")
    print(f"Sensitivity: {sensitivity:.4f}  (= 2·{args.local_lr}·{args.clip_C} / 1)\n")

    # ---- DP 参数对照表（对齐官方 noise_scale 公式）----
    print(f"{'='*65}")
    print(f"  {'ε_round':>8}  {'Lap scale(b)':>14}  {'Gau std(σ)':>12}  {'SNR_Lap':>10}")
    print(f"  {'-'*55}")
    for eps in epsilons:
        if math.isinf(eps):
            print(f"  {'∞':>8}  {'0':>14}  {'0':>12}  {'∞':>10}")
        else:
            b     = sensitivity / eps                                  # Laplace scale
            sigma = math.sqrt(2 * math.log(1.25 / DELTA_DP)) * sensitivity / eps
            snr   = true_grad.norm().item() / max(b * math.sqrt(2) * math.sqrt(true_grad.numel()), 1e-12)
            print(f"  {eps:>8g}  {b:>14.6f}  {sigma:>12.6f}  {snr:>10.4f}")
    print(f"{'='*65}\n")

    real_vis = x_real.clamp(0, 1).cpu()

    # ---- 对两种机制逐 ε 发起攻击 ----
    results_all = {}
    psnr_lap, psnr_gau = [], []

    for mech_name, mechanism in [("Laplace", "laplace"), ("Gaussian", "gaussian")]:
        print(f"\n{'='*55}")
        print(f"  Mechanism: {mech_name}")
        print(f"{'='*55}")

        results_by_eps = {}
        for eps in epsilons:
            print(f"\n--- {eps_label(eps)} ---")

            if math.isinf(eps):
                noisy_grad   = true_grad.clone()
                signal_stats = {
                    "grad_norm":  true_grad.norm().item(),
                    "noise_norm": 0.0,
                    "snr":        float("inf"),
                }
            else:
                noisy_grad, signal_stats = add_dp_noise_to_gradient(
                    true_grad, mechanism, eps,
                    args.local_lr, args.clip_C, DELTA_DP,
                )

            print(f"grad_norm={signal_stats['grad_norm']:.4f}  "
                  f"noise_norm={signal_stats['noise_norm']:.4f}  "
                  f"SNR={signal_stats['snr']:.3f}")

            result = grnn_attack(
                model, noisy_grad, batch_size=1,
                num_classes=10, img_size=32, in_channels=1,
                iterations=args.iterations, tv_alpha=args.tv_alpha,
                device=device, seed=args.seed, verbose=True,
            )

            fake_vis = result["fake_img"].clamp(0, 1)
            ps       = psnr(real_vis[0], fake_vis[0])
            print(f"PSNR = {ps:.2f} dB")

            results_by_eps[eps] = {"fake_vis": fake_vis, "psnr": ps, **signal_stats}

        mech_tag = mechanism  # "laplace" or "gaussian"
        save_defense_grid(
            real_vis, results_by_eps, epsilons, mech_name,
            os.path.join(args.outdir, f"defense_{mech_tag}.png"),
        )
        results_all[mechanism] = results_by_eps

        if mechanism == "laplace":
            psnr_lap = [results_by_eps[e]["psnr"] for e in epsilons]
        else:
            psnr_gau = [results_by_eps[e]["psnr"] for e in epsilons]

    save_psnr_bar(psnr_lap, psnr_gau, epsilons, args.clip_C,
                  os.path.join(args.outdir, "psnr_bar.png"))

    # ---- 汇总打印 ----
    print("\n" + "=" * 70)
    print("  Privacy-Utility Summary")
    print(f"  sensitivity = 2·{args.local_lr}·{args.clip_C} = {sensitivity:.4f}")
    print(f"  {'ε_round':>12}  {'Laplace PSNR':>14}  {'Gaussian PSNR':>14}")
    print("-" * 70)
    for eps, pl, pg in zip(epsilons, psnr_lap, psnr_gau):
        print(f"  {eps_label(eps):>12}  {pl:>14.2f} dB  {pg:>14.2f} dB")
    print("=" * 70)

    save_psnr_csv(results_all, epsilons, args.clip_C, args.local_lr,
                  os.path.join(args.outdir, "psnr_data.csv"))
    print("\nAll defense experiments done.")


if __name__ == "__main__":
    main()
