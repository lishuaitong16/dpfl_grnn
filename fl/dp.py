"""差分隐私工具：梯度裁剪 + 拉普拉斯/高斯加噪 + 隐私预算（简单组合定理）。

使用顺序（不可颠倒）：
    1) clip_update：把更新向量的 L2 范数裁到 <= C，使敏感度 = C。
    2) add_*_noise：按敏感度和隐私预算加噪。
没有第 1 步，敏感度不确定，差分隐私不成立。
"""

import math
import torch


# ---------------------------------------------------------------------------
# 梯度/更新裁剪
# ---------------------------------------------------------------------------
def clip_update(vec: torch.Tensor, C: float) -> torch.Tensor:
    """按 L2 范数裁剪：vec * min(1, C / ||vec||_2)。

    裁剪后该向量的 L2 敏感度上界即为 C。
    """
    norm = vec.norm(p=2)
    factor = min(1.0, C / (norm.item() + 1e-12))
    return vec * factor


# ---------------------------------------------------------------------------
# 加噪机制
# ---------------------------------------------------------------------------
def add_laplace_noise(vec: torch.Tensor, sensitivity: float, epsilon: float) -> torch.Tensor:
    """拉普拉斯机制，满足 epsilon-DP（基于 L1 敏感度）。

    噪声尺度 b = sensitivity / epsilon。这里 sensitivity 用裁剪范数 C。
    注：严格 L1 敏感度由 L2 裁剪给出的上界为 sqrt(d)*C，本实验为简化
    采用工程上常见做法直接用 C 作为尺度参数，足以演示噪声对攻防的影响。
    """
    b = sensitivity / max(epsilon, 1e-12)
    # 用两个指数分布之差生成拉普拉斯噪声（PyTorch 无原生 Laplace 采样张量接口时通用）
    lap = torch.distributions.Laplace(loc=0.0, scale=b)
    noise = lap.sample(vec.shape).to(vec.device)
    return vec + noise


def gaussian_sigma(sensitivity: float, epsilon: float, delta: float) -> float:
    """高斯机制的标准差：sigma = sensitivity * sqrt(2 ln(1.25/delta)) / epsilon。"""
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / max(epsilon, 1e-12)


def add_gaussian_noise(vec: torch.Tensor, sensitivity: float, epsilon: float,
                       delta: float = 1e-5) -> torch.Tensor:
    """高斯机制，满足 (epsilon, delta)-DP（基于 L2 敏感度）。"""
    sigma = gaussian_sigma(sensitivity, epsilon, delta)
    noise = torch.randn_like(vec) * sigma
    return vec + noise


# ---------------------------------------------------------------------------
# 隐私预算：简单组合定理
# ---------------------------------------------------------------------------
def per_round_epsilon(total_epsilon: float, T: int) -> float:
    """简单组合定理：T 轮总预算 = T * 单轮预算  =>  单轮 = 总 / T。"""
    return total_epsilon / T


def total_epsilon(per_round_eps: float, T: int) -> float:
    """由单轮预算反推总预算。"""
    return per_round_eps * T


# ---------------------------------------------------------------------------
# 便捷封装：对一个更新向量做"裁剪 + 加噪"
# ---------------------------------------------------------------------------
def privatize(vec: torch.Tensor, mechanism: str, C: float, epsilon: float,
              delta: float = 1e-5) -> torch.Tensor:
    """对单个更新向量执行差分隐私处理。

    Args:
        vec: 待保护的更新/梯度向量（已拉平为 1D）。
        mechanism: "laplace" 或 "gaussian" 或 "none"。
        C: 梯度裁剪范数（同时作为敏感度）。
        epsilon: 该次操作的隐私预算（通常是单轮预算 epsilon_round）。
        delta: 高斯机制用的 delta。
    """
    mechanism = mechanism.lower()
    if mechanism == "none":
        return vec
    clipped = clip_update(vec, C)
    if mechanism == "laplace":
        return add_laplace_noise(clipped, sensitivity=C, epsilon=epsilon)
    if mechanism == "gaussian":
        return add_gaussian_noise(clipped, sensitivity=C, epsilon=epsilon, delta=delta)
    raise ValueError(f"未知机制: {mechanism}")
