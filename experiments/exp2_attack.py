"""阶段四：复现 GRNN 梯度攻击（无差分隐私，batch=1 起步）。

流程：
  1. 构造一个 Sigmoid 版 LeNet 作为"全局模型"（攻击要求二阶可导）。
     论文指出未收敛模型反而更易攻击，这里默认用随机初始化模型（=第0轮）。
     可用 --train_iters 让模型先训练几步，模拟"训练早期"的全局模型。
  2. 取 batch_size 张真实 MNIST 图，算出真实梯度（这就是被截获的梯度）。
  3. 跑 GRNN 攻击还原图像。
  4. 保存：逐步还原过程图、还原 vs 原图对比图、损失曲线，并打印 PSNR。

运行：
    python -m experiments.exp2_attack --batch_size 1 --iterations 2000
"""

import os
import sys
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import build_lenet
from models.resnet import build_resnet18
from fl.data import get_datasets, make_test_loader, denormalize
from attack.run_attack import compute_true_gradient, grnn_attack, psnr


def save_grid(images, titles, path, nrow=None, cmap="gray"):
    """把若干 (C,H,W) 图像横排保存，images 为 [0,1] 张量列表。"""
    n = len(images)
    nrow = nrow or n
    ncol = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(2.5 * nrow, 3.2 * ncol))
    axes = [axes] if n == 1 else (axes.flatten() if hasattr(axes, "flatten") else axes)
    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            arr = img.mean(0).numpy()
            ax.imshow(arr, cmap="gray")
            ax.set_title(titles[i], fontsize=9, pad=6)
        ax.axis("off")
    plt.tight_layout(pad=1.0)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      type=str,   default="lenet",
                   help="全局模型类型：lenet 或 resnet18")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--train_iters", type=int, default=0,
                   help="攻击前让全局模型先训练几步（0=随机初始化，最易攻击）")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="./results/attack")
    p.add_argument("--tv_alpha", type=float, default=1e-3, help="TV 损失权重")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index to use, -1 for CPU")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    # 1. 全局模型
    if args.model == "resnet18":
        model = build_resnet18(num_classes=10, in_channels=1, act="sigmoid").to(device)
    else:
        model = build_lenet(num_classes=10, in_channels=1, act="sigmoid").to(device)
    print(f"Model: {args.model}")

    train_set, _ = get_datasets(args.data_root)

    # 从训练集取图（FL 中梯度来自客户端训练数据）
    train_loader_single = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    x_real, y_real = next(iter(train_loader_single))
    x_real, y_real = x_real[:args.batch_size].to(device), y_real[:args.batch_size].to(device)

    # 可选：先训练几步，模拟训练早期的全局模型（全量训练集 + SGD）
    if args.train_iters > 0:
        loader = DataLoader(train_set, batch_size=64, shuffle=True)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        model.train()
        it = iter(loader)
        for step in range(args.train_iters):
            try:
                xb, yb = next(it)
            except StopIteration:
                it = iter(loader)
                xb, yb = next(it)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
            if (step + 1) % 100 == 0:
                print(f"  Pre-training step {step+1}/{args.train_iters}")
        model.eval()

    # 2. 计算真实梯度（create_graph=True 以支持二阶反传）
    true_grad = compute_true_gradient(model, x_real, y_real, device)
    print(f"True labels: {y_real.tolist()}  Gradient dim: {true_grad.numel()}")

    # 3. 攻击
    result = grnn_attack(
        model, true_grad, batch_size=args.batch_size,
        num_classes=10, img_size=32, in_channels=1,
        iterations=args.iterations, tv_alpha=args.tv_alpha,
        device=device, seed=args.seed,
    )

    # 4. 保存结果
    x_real_vis = denormalize(x_real).cpu()
    fake_vis = result["fake_img"].clamp(0, 1)  # sigmoid output already in [0,1]

    # 4a. 逐步还原过程（取第 0 个样本）
    snap_iters = sorted(result["snapshots"].keys())
    snap_imgs = [result["snapshots"][i][0:1].clamp(0, 1)[0] for i in snap_iters]
    snap_imgs = [x_real_vis[0]] + snap_imgs
    snap_titles = ["Original"] + [f"iter {i}" for i in snap_iters]
    save_grid(snap_imgs, snap_titles, os.path.join(args.outdir, "recover_process.png"),
              nrow=len(snap_imgs))

    # 4b. 还原 vs 原图对比 + PSNR
    cmp_imgs, cmp_titles = [], []
    for b in range(args.batch_size):
        ps = psnr(x_real_vis[b], fake_vis[b])
        cmp_imgs += [x_real_vis[b], fake_vis[b]]
        cmp_titles += [f"Original #{b}", f"Recovered #{b}\nPSNR={ps:.1f}dB"]
    save_grid(cmp_imgs, cmp_titles, os.path.join(args.outdir, "compare.png"),
              nrow=2)

    # 4c. 损失曲线
    hist = result["loss_history"]
    its = [h[0] for h in hist]
    plt.figure(figsize=(6, 4))
    plt.semilogy(its, [h[2] for h in hist], label="MSE")
    plt.semilogy(its, [h[3] for h in hist], label="WD")
    plt.xlabel("Iteration"); plt.ylabel("Loss (log)"); plt.legend()
    plt.title("GRNN Attack Loss Curve"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "loss_curve.png"), dpi=150)
    plt.close()

    avg_psnr = sum(psnr(x_real_vis[b], fake_vis[b]) for b in range(args.batch_size)) / args.batch_size
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
    print(f"Results saved to: {args.outdir}/  (recover_process.png, compare.png, loss_curve.png)")


if __name__ == "__main__":
    main()
