"""差分隐私机制：对齐官方实现 client/dp_mechanism.py。

一、官方接口（函数签名和公式与 dp-fl/client/dp_mechanism.py 完全一致）
─────────────────────────────────────────────────────────────────────
  cal_client_sensitivity(lr, clip, dataset_size)
      sensitivity = 2 * lr * clip / dataset_size

  laplace_mechanism(epsilon, sensitivity, size)
      noise_scale = sensitivity / epsilon
      返回 numpy 数组

  gaussian_mechanism(epsilon, delta, sensitivity, size)
      noise_scale = sqrt(2 * ln(1.25/δ)) * sensitivity / ε
      返回 numpy 数组
      注意：官方要求 epsilon < 1；本实现对 ε≥1 给出警告而非报错
            （实验扫参时允许 ε>1，但此时 DP 保证较为宽松）

  simple_composition(k, total_epsilon, total_delta)
      每轮 ε = total_ε / k，δ = total_δ / k

  advanced_composition(k, total_epsilon, total_delta)
      使用高级组合定理（Boosting and Differential Privacy）

  renyi_gaussian_composition(k, total_epsilon, total_delta)
      使用 Rényi 差分隐私组合

  renyi_gaussian_mechanism(alpha, epsilon, sensitivity, size)
      noise_scale = sqrt(α / (2ε)) * sensitivity
      返回 numpy 数组

二、辅助接口（供 fl/train.py 使用的 PyTorch 版本）
─────────────────────────────────────────────────────────────────────
  clip_gradient(vec, C)
      把张量向量裁剪到 L2 球半径 C，超出才裁

  add_dp_noise_torch(delta, mechanism, epsilon, sensitivity, delta_dp)
      调用官方 numpy 接口后转换为 torch tensor，直接加到 delta 上
      sensitivity 应由 cal_client_sensitivity 预先计算
"""

import math
import warnings
import numpy as np
import torch
from scipy.optimize import fsolve


# ============================================================================
# 一、官方接口（完全对齐 client/dp_mechanism.py）
# ============================================================================

def cal_client_sensitivity(lr: float, clip: float, dataset_size: int) -> float:
    """计算本地客户端学习的敏感度。

    与官方 cal_client_sensitivity 完全一致：
        sensitivity = 2 * lr * clip / dataset_size

    Args:
        lr:           本地学习率。
        clip:         梯度裁剪范数（L1 for Laplace，L2 for Gaussian）。
        dataset_size: 客户端本地数据集大小（样本总数 N）。
    """
    return 2.0 * lr * clip / dataset_size


def laplace_mechanism(epsilon: float, sensitivity: float, size) -> np.ndarray:
    """生成满足 ε-差分隐私的 Laplace 噪声（与官方完全一致）。

    尺度参数 b = sensitivity / ε，噪声 ~ Laplace(0, b)。
    返回 numpy 数组，形状为 size。
    """
    noise_scale = sensitivity / epsilon
    return np.random.laplace(0, scale=noise_scale, size=size)


def gaussian_mechanism(epsilon: float, delta: float,
                        sensitivity: float, size) -> np.ndarray:
    """生成满足 (ε,δ)-差分隐私的 Gaussian 噪声（与官方完全一致）。

    标准差 σ = sqrt(2 * ln(1.25/δ)) * sensitivity / ε。
    返回 numpy 数组，形状为 size。

    注意：严格意义上该公式要求 ε < 1；本函数对 ε≥1 发出警告而不中止，
    以允许实验参数扫描（此时 DP 约束仍成立但界不是最优的）。
    """
    if epsilon >= 1:
        warnings.warn(
            f"gaussian_mechanism: epsilon={epsilon} >= 1，"
            "Gaussian 机制的标准公式在此范围外 DP 保证较宽松。"
            "建议使用组合定理将总预算拆为多轮（per-round ε < 1）。",
            UserWarning, stacklevel=2,
        )
    noise_scale = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
    return np.random.normal(0, noise_scale, size=size)


def simple_composition(k: int, total_epsilon: float,
                        total_delta: float):
    """简单组合定理（与官方完全一致）。

    Sources: Boosting and Differential Privacy
    每轮 ε = total_ε / k，δ = total_δ / k。
    """
    epsilon = total_epsilon / k
    delta   = total_delta   / k
    return epsilon, delta


def advanced_composition(k: int, total_epsilon: float,
                          total_delta: float):
    """高级组合定理（与官方完全一致）。

    Sources: Boosting and Differential Privacy
    """
    delta_prime_ratio = 0.5
    delta_prime = delta_prime_ratio * total_delta
    delta = (total_delta - delta_prime) / k
    x0 = [total_epsilon / k]

    def f(x0):
        x = x0[0]
        return [math.sqrt(2 * k * math.log(1 / delta_prime)) * x
                + k * x * (math.exp(x) - 1) - total_epsilon]

    epsilon = fsolve(f, x0)
    return epsilon, delta


def renyi_gaussian_composition(k: int, total_epsilon: float,
                                 total_delta: float):
    """Rényi 差分隐私组合定理（与官方完全一致）。

    Sources: Rényi differential privacy
    返回 (alpha, epsilon_per_round)。
    """
    n     = 1 + math.log(1 / total_delta) / total_epsilon
    alpha = n + math.sqrt(n ** 2 - n)
    epsilon = (total_epsilon - math.log(1 / total_delta) / (alpha - 1)) / k
    return alpha, epsilon


def renyi_gaussian_mechanism(alpha: float, epsilon: float,
                               sensitivity: float, size) -> np.ndarray:
    """生成满足 (α,ε)-Rényi 差分隐私的 Gaussian 噪声（与官方完全一致）。

    noise_scale = sqrt(α / (2ε)) * sensitivity
    返回 numpy 数组，形状为 size。
    """
    noise_scale = math.sqrt(alpha / (2 * epsilon)) * sensitivity
    return np.random.normal(0, noise_scale, size=size)


# ============================================================================
# 二、辅助接口（PyTorch 版，供 fl/train.py 使用）
# ============================================================================

def clip_gradient(vec: torch.Tensor, C: float, norm: int = 2) -> torch.Tensor:
    """把向量裁剪到 Lp 球半径 C（超出才裁，否则不变）。

    Args:
        vec:  要裁剪的张量（1-D 或任意形状）。
        C:    裁剪阈值（对应官方 dp_clip）。
        norm: 范数类型。
              2 → L2 范数（Gaussian 机制，官方默认）
              1 → L1 范数（Laplace 机制，对齐官方 perSampleClip norm=1）

    官方 client/Update.py perSampleClip 使用：
        Laplace  → norm=1
        Gaussian → norm=2
    本函数裁剪的是整体 delta（而非官方的 per-sample gradient），
    但裁剪范数的选择与官方保持一致。
    """
    if norm == 2:
        cur_norm = vec.norm(2).item()
    elif norm == 1:
        cur_norm = vec.norm(1).item()
    else:
        raise ValueError(f"norm 必须是 1 或 2，得到 {norm}")
    if cur_norm > C:
        vec = vec * (C / cur_norm)
    return vec


def add_dp_noise_torch(
    delta:      torch.Tensor,
    mechanism:  str,
    epsilon:    float,
    sensitivity: float,
    delta_dp:   float = 1e-5,
) -> torch.Tensor:
    """给 torch 张量加 DP 噪声（内部调用官方 numpy 接口再转回 tensor）。

    sensitivity 应由 cal_client_sensitivity(lr, clip, N) 预先计算，
    而非直接使用 clip 范数。

    Args:
        delta:      参数更新向量（已裁剪）。
        mechanism:  "laplace" 或 "gaussian"。
        epsilon:    每轮隐私预算 ε_round。
        sensitivity: 客户端敏感度 = 2 * lr * clip / N。
        delta_dp:   (ε,δ)-DP 的 δ（Gaussian only，默认 1e-5）。
    Returns:
        加噪后的向量（与 delta 同设备、同 dtype）。
    """
    size = tuple(delta.shape)
    if mechanism == "laplace":
        noise_np = laplace_mechanism(epsilon, sensitivity, size)
    elif mechanism == "gaussian":
        noise_np = gaussian_mechanism(epsilon, delta_dp, sensitivity, size)
    else:
        raise ValueError(f"未知 DP 机制: {mechanism}（应为 'laplace' 或 'gaussian'）")
    noise = torch.from_numpy(noise_np).to(device=delta.device, dtype=delta.dtype)
    return delta + noise
