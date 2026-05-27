"""GRNN 攻击损失：L2² + L2-norm + TV 正则。

- L2²：sum((g_fake - g_true)²)，梯度差的平方和
- WD ：‖g_fake − g_true‖₂，梯度差的 L2 范数（位置对齐匹配）
- TVLoss：总变差，约束生成图像的空间平滑性
"""

import torch


def grad_l2sq(g_true, g_fake):
    """梯度差的 L2 平方和（对应官方 loss_f('l2', ...)）。"""
    return ((g_fake - g_true) ** 2).sum()


def grad_wd(g_true, g_fake):
    """梯度差的 L2 范数（‖g_fake − g_true‖₂）。

    梯度每一维对应固定的模型参数，必须位置对齐地匹配，
    不能用排序后的分布距离（K-R WD）——那会忽略维度顺序。
    """
    diff = (g_fake - g_true).view(-1)
    return torch.pow(torch.sum(torch.pow(torch.abs(diff), 2)), 0.5)


def tv_loss(img):
    """L2² 总变差损失，对齐官方实现（平方差之和，除以元素数和 batch size）。"""
    B = img.size(0)
    count_h = img[:, :, 1:, :].numel()
    count_w = img[:, :, :, 1:].numel()
    h_tv = ((img[:, :, 1:, :] - img[:, :, :-1, :]) ** 2).sum()
    w_tv = ((img[:, :, :, 1:] - img[:, :, :, :-1]) ** 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / B


def grnn_loss(g_true, g_fake, fake_img, alpha=1e-3):
    """GRNN 总损失，返回各分量便于记录曲线。

    Args:
        alpha: TV 损失权重（默认 1e-3）
    Returns:
        total, dict(mse=, wd=, tv=)
    """
    mse = grad_l2sq(g_true, g_fake)
    wd  = grad_wd(g_true, g_fake)
    tv  = tv_loss(fake_img) * alpha
    total = mse + wd + tv
    return total, {"mse": mse.item(), "wd": wd.item(), "tv": tv.item()}
