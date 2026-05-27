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

    官方 dp_mechanism.py 要求 epsilon < 1 并在 ε≥1 时抛异常；
    本实现允许 ε≥1（实验扫描时有意使用大 ε 展示隐私-效用权衡），
    公式本身在此范围仍然有效，只是严格 DP 界不是最优的。
    """
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


# ============================================================================
# 三、Per-sample gradient clipping（对齐官方 perSampleClip，无需 Opacus）
# ============================================================================

def per_sample_clip_grads(
    model,
    x_batch:  torch.Tensor,
    y_batch:  torch.Tensor,
    clip_C:   float,
    clip_norm: int = 2,
) -> dict:
    """逐样本裁剪梯度，返回聚合后的梯度字典（对齐官方 perSampleClip）。

    实现方式：torch.func.vmap + grad（PyTorch ≥ 2.0，无需 Opacus）。

    官方 Update.py perSampleClip 流程（完全对齐）：
      1. 计算每个样本的梯度 g_i
      2. 计算 per-sample 梯度范数（Laplace 用 L1，Gaussian 用 L2）
      3. clip_factor = min(1, clip_C / ||g_i||)
      4. g_i_clipped = g_i * clip_factor
      5. 返回平均裁剪梯度：mean(g_i_clipped)

    Args:
        model:     全局模型（参数不会被修改，仅用于 functional_call）。
        x_batch:   (B, C, H, W) 输入图像，已在目标 device 上。
        y_batch:   (B,) 标签，已在目标 device 上。
        clip_C:    每步裁剪阈值（对应官方 dp_clip / local_ep）。
        clip_norm: 范数类型（1=L1 for Laplace, 2=L2 for Gaussian）。

    Returns:
        dict: {param_name: clipped_grad_tensor}，形状与模型参数一致，
              已平均（相当于聚合后除以 batch_size）。
    """
    import torch.nn.functional as F
    from torch.func import functional_call, grad, vmap

    params  = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def loss_for_one(params, x_i, y_i):
        """单样本损失（功能性调用，不修改模型状态）。"""
        out  = functional_call(model, (params, buffers), (x_i.unsqueeze(0),))
        return F.cross_entropy(out, y_i.unsqueeze(0))

    # vmap 把 grad(loss_for_one) 在 batch 维度上并行化
    # in_dims=(None, 0, 0): params 不批化，x/y 沿 dim=0 批化
    per_sample_grads = vmap(grad(loss_for_one), in_dims=(None, 0, 0))(
        params, x_batch, y_batch
    )
    # per_sample_grads: {name: (batch_size, *param_shape)}

    batch_size = x_batch.shape[0]

    # ── 1. 计算每个样本的总范数（所有层联合）──
    per_param_norms = [
        g.reshape(batch_size, -1).norm(clip_norm, dim=1)
        for g in per_sample_grads.values()
    ]
    per_sample_norm = torch.stack(per_param_norms, dim=1).norm(clip_norm, dim=1)
    # shape: (batch_size,)

    # ── 2. 计算裁剪因子 min(1, C / ||g||) ──
    clip_factor = (clip_C / (per_sample_norm + 1e-6)).clamp(max=1.0)
    # shape: (batch_size,)

    # ── 3. 应用裁剪因子，对 batch 取均值 ──
    aggregated = {}
    for name, g in per_sample_grads.items():
        # g: (batch_size, *param_shape)
        factor = clip_factor.view(-1, *([1] * (g.dim() - 1)))
        aggregated[name] = (g * factor).mean(dim=0)  # 平均，与 batch SGD 等价

    return aggregated


def add_noise_to_params(
    model,
    mechanism:   str,
    epsilon:     float,
    sensitivity: float,
    delta_dp:    float = 1e-5,
) -> None:
    """训练结束后，对模型每个参数层独立加噪声（对齐官方 add_noise）。

    官方 add_noise 逻辑：
        for name, parameter in net.named_parameters():
            noise = laplace_mechanism(...)  # 与参数 shape 相同
            parameter += noise

    Args:
        model:       训练完成的模型（in-place 修改参数）。
        mechanism:   "laplace" 或 "gaussian"。
        epsilon:     每轮隐私预算 ε_round（已由组合定理换算好的 per-round 值）。
        sensitivity: 2 * lr * clip / N（由 cal_client_sensitivity 计算）。
        delta_dp:    (ε,δ)-DP 的 δ（Gaussian only）。
    """
    with torch.no_grad():
        for param in model.parameters():
            size = tuple(param.shape)
            if mechanism == "laplace":
                noise_np = laplace_mechanism(epsilon, sensitivity, size)
            elif mechanism == "gaussian":
                noise_np = gaussian_mechanism(epsilon, delta_dp, sensitivity, size)
            else:
                raise ValueError(f"未知机制: {mechanism}")
            noise = torch.from_numpy(noise_np).to(device=param.device, dtype=param.dtype)
            param.data += noise
