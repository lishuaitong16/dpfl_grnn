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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lenet import build_lenet
from fl.data import get_datasets, make_test_loader, denormalize
from attack.run_attack import compute_true_gradient, grnn_attack, psnr


def save_grid(images, titles, path, nrow=None, cmap="gray"):
    """把若干 (C,H,W) 图像横排保存，images 为 [0,1] 张量列表。"""
    n = len(images)
    nrow = nrow or n
    ncol = (n + nrow - 1) // nrow
    fig, axes = plt.subplots(ncol, nrow, figsize=(2 * nrow, 2 * ncol))
    axes = [axes] if n == 1 else (axes.flatten() if hasattr(axes, "flatten") else axes)
    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            arr = img[0].numpy() if img.shape[0] == 1 else img.permute(1, 2, 0).numpy()
            ax.imshow(arr, cmap=cmap if img.shape[0] == 1 else None)
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--train_iters", type=int, default=0,
                   help="攻击前让全局模型先训练几步（0=随机初始化，最易攻击）")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--outdir", type=str, default="./results/attack")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index to use, -1 for CPU")
    args = p.parse_args()

    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    # 1. Sigmoid 版全局模型
    model = build_lenet(num_classes=10, in_channels=1, act="sigmoid").to(device)

    _, test_set = get_datasets(args.data_root)
    test_loader = make_test_loader(test_set, batch_size=args.batch_size)

    # 取一个 batch 的真实图像
    x_real, y_real = next(iter(test_loader))
    x_real, y_real = x_real[:args.batch_size].to(device), y_real[:args.batch_size].to(device)

    # 可选：先训练几步，模拟训练早期的全局模型
    if args.train_iters > 0:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        crit = nn.CrossEntropyLoss()
        for _ in range(args.train_iters):
            opt.zero_grad()
            crit(model(x_real), y_real).backward()
            opt.step()

    # 2. 计算真实梯度（create_graph=True 以支持二阶反传）
    true_grad = compute_true_gradient(model, x_real, y_real, device, create_graph=True)
    print(f"True labels: {y_real.tolist()}  Gradient dim: {true_grad.numel()}")

    # 3. 攻击
    result = grnn_attack(
        model, true_grad, batch_size=args.batch_size,
        num_classes=10, img_size=32, in_channels=1,
        iterations=args.iterations, device=device, seed=args.seed,
    )

    # 4. 保存结果
    x_real_vis = denormalize(x_real).cpu()
    fake_vis = result["fake_img"].clamp(-1, 1)
    fake_vis = (fake_vis + 1) / 2  # tanh [-1,1] -> [0,1]

    # 4a. 逐步还原过程（取第 0 个样本）
    snap_iters = sorted(result["snapshots"].keys())
    snap_imgs = [((result["snapshots"][i][0:1].clamp(-1, 1) + 1) / 2)[0] for i in snap_iters]
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
