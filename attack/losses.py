"""GRNN 攻击损失：MSE + Wasserstein 距离 + TV 正则。

论文损失：L = MSE(g, ĝ) + WD(g, ĝ) + alpha * TVLoss(x̂)
- MSE：两个梯度向量的欧氏距离平方，衡量数值上接近程度。
- WD ：Wasserstein 距离，衡量两个梯度"分布"的几何差异（与 MSE 等权）。
       这里采用一维近似：对两个向量排序后求 L1 距离，是 1D Wasserstein-1 的精确形式。
- TVLoss：总变差，约束生成图像的空间平滑性，让结果像自然图片而非噪点。
"""

import torch


def grad_mse(g_true, g_fake):
    """两个拉平梯度向量的均方误差。"""
    return ((g_true - g_fake) ** 2).mean()


def grad_wasserstein(g_true, g_fake):
    """一维 Wasserstein-1 距离：排序后的 L1 距离。

    对于一维分布，W1 等于两个排序后序列的逐元素绝对差之和（再取均值）。
    这是论文中用 WD 衡量梯度几何差异的一个轻量可微近似。
    """
    a, _ = torch.sort(g_true)
    b, _ = torch.sort(g_fake)
    return (a - b).abs().mean()


def tv_loss(img):
    """各向同性总变差损失。

    img: 形状 (B, C, H, W)。
    返回相邻像素差的均值，鼓励平滑。
    """
    dh = (img[:, :, 1:, :] - img[:, :, :-1, :]).abs().mean()
    dw = (img[:, :, :, 1:] - img[:, :, :, :-1]).abs().mean()
    return dh + dw


def grnn_loss(g_true, g_fake, fake_img, alpha=1e-3):
    """GRNN 总损失，并返回各分量便于记录曲线。

    Returns:
        total, dict(mse=, wd=, tv=)
    """
    mse = grad_mse(g_true, g_fake)
    wd = grad_wasserstein(g_true, g_fake)
    tv = tv_loss(fake_img)
    total = mse + wd + alpha * tv
    return total, {"mse": mse.item(), "wd": wd.item(), "tv": tv.item()}
