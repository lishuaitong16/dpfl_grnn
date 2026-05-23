"""GRNN 攻击核心：从（可能带噪的）真实梯度还原私有图像。

对应论文 Algorithm 1：
  冻结全局模型 θ，优化生成器 G，使 G 产生的假数据在 θ 上算出的"假梯度 ĝ"
  逼近截获的"真梯度 g"。收敛后 G 的输出即还原图像。

关键点：
  - 真梯度用 create_graph=True 计算，使后续"损失对 ĝ 的梯度"可二阶反传，
    所以模型激活必须用 Sigmoid（见 models/lenet.py 说明）。
  - 真梯度 g 和假梯度 ĝ 必须用相同参数顺序拉平后比较。
  - 优化器用 RMSprop，lr=1e-4，momentum=0.99（论文设置）。
"""

import os
import math
import torch
import torch.nn as nn

from .grnn import GRNNGenerator
from .losses import grnn_loss


def compute_true_gradient(model, x, y, device, create_graph=False):
    """在冻结的全局模型上对一个 batch 计算梯度（按参数顺序拉平为向量）。

    x: (B,C,H,W) 真实图像（已标准化）；y: (B,) 真实标签。
    返回拉平后的梯度向量。
    """
    model = model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    x, y = x.to(device), y.to(device)
    out = model(x)
    loss = criterion(out, y)
    grads = torch.autograd.grad(loss, list(model.parameters()),
                                create_graph=create_graph)
    flat = torch.cat([g.reshape(-1) for g in grads])
    return flat


def compute_fake_gradient(model, fake_img, fake_label, device):
    """用生成的假图/假软标签计算梯度（保持计算图，供生成器反传）。

    fake_label 为软标签（概率分布），用与 softmax 交叉熵等价的形式：
        loss = - sum(fake_label * log_softmax(logits))
    这样对生成器可微。
    """
    model = model.to(device)
    # tanh [-1,1] -> MNIST normalized space, matching how true gradients are computed
    fake_norm = (fake_img + 1.0) / 2.0
    fake_norm = (fake_norm - 0.1307) / 0.3081
    out = model(fake_norm.to(device))
    log_prob = torch.log_softmax(out, dim=1)
    loss = -(fake_label.to(device) * log_prob).sum(dim=1).mean()
    grads = torch.autograd.grad(loss, list(model.parameters()), create_graph=True)
    flat = torch.cat([g.reshape(-1) for g in grads])
    return flat


def psnr(img1, img2):
    """两张 [0,1] 图像的 PSNR（dB）。越高越相似。"""
    mse = ((img1 - img2) ** 2).mean().item()
    if mse < 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def grnn_attack(
    global_model,
    true_grad,                 # 截获的真实梯度（已拉平），可以是带噪的
    batch_size: int,
    num_classes: int = 10,
    img_size: int = 32,
    in_channels: int = 1,
    latent_dim: int = 1024,
    iterations: int = 2000,
    lr: float = 1e-4,
    momentum: float = 0.99,
    tv_alpha: float = 1e-3,
    device: str = "cuda",
    snapshot_iters=(0, 100, 300, 500, 1000, 1500, 1999),
    seed: int = 0,
    verbose: bool = True,
):
    """运行 GRNN 攻击，返回还原结果与中间快照。

    Returns dict:
        fake_img: 最终还原图像 (B,C,H,W)，已 detach 到 CPU
        snapshots: {iter: 图像张量}  用于展示"逐步还原"过程
        loss_history: [(iter, total, mse, wd, tv), ...]
    """
    torch.manual_seed(seed)
    global_model = global_model.to(device)
    for p in global_model.parameters():
        p.requires_grad_(True)  # 需要对参数求梯度（算 fake 梯度），但不更新它们

    true_grad = true_grad.detach().to(device)

    generator = GRNNGenerator(
        latent_dim=latent_dim, num_classes=num_classes,
        out_channels=in_channels, target_size=img_size,
    ).to(device)

    # 固定输入向量（论文：每个样本一个随机向量，整个攻击过程固定）
    v = torch.randn(batch_size, latent_dim, device=device)

    optimizer = torch.optim.RMSprop(generator.parameters(), lr=lr, momentum=momentum)

    snapshots = {}
    loss_history = []

    for it in range(iterations):
        optimizer.zero_grad()
        fake_img, fake_label = generator(v)
        fake_grad = compute_fake_gradient(global_model, fake_img, fake_label, device)

        total, parts = grnn_loss(true_grad, fake_grad, fake_img, alpha=tv_alpha)
        total.backward()
        optimizer.step()

        loss_history.append((it, total.item(), parts["mse"], parts["wd"], parts["tv"]))

        if it in snapshot_iters:
            snapshots[it] = fake_img.detach().cpu().clone()
        if verbose and (it % 100 == 0 or it == iterations - 1):
            print(f"[iter {it:4d}] total={total.item():.4e} "
                  f"mse={parts['mse']:.3e} wd={parts['wd']:.3e} tv={parts['tv']:.3e}")

    return {
        "fake_img": fake_img.detach().cpu(),
        "fake_label": fake_label.detach().cpu(),
        "snapshots": snapshots,
        "loss_history": loss_history,
    }
