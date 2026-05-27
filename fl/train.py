"""FedAvg 联邦学习主循环，支持 per-round DP（对齐官方 dp-fl 实现）。

DP 方案（对齐 client/dp_mechanism.py + client/Update.py）：
────────────────────────────────────────────────────────────
  敏感度：sensitivity = 2 * lr * clip / N
      其中 N 为该客户端的本地数据集大小（对齐 cal_client_sensitivity）

  Laplace：noise_scale = sensitivity / ε_round
  Gaussian：noise_std = sqrt(2·ln(1.25/δ))·sensitivity / ε_round

  噪声加在参数更新量 delta = local_params - global_params 上，
  等价于官方把噪声加到模型参数上（delta ≠ 0 时相同效果）。

组合定理说明（sequential composition）：
  总预算 ε_total = T × ε_round（简单组合，T = 通信轮数）
  或使用 fl.dp.simple_composition / advanced_composition 预先换算。

无噪声时（mechanism="none"）即标准 FedAvg 基线。
"""

import copy
import math
import torch
import torch.nn as nn

from .dp import (
    cal_client_sensitivity,
    clip_gradient,
    add_dp_noise_torch,
    renyi_gaussian_mechanism,
)


# ---------------------------------------------------------------------------
# 参数 <-> 向量（加噪、GRNN 都要用一致的顺序）
# ---------------------------------------------------------------------------

def params_to_vector(state_dict) -> torch.Tensor:
    """把 state_dict（有序）拉平为 1-D float 向量。"""
    return torch.cat([v.reshape(-1).float() for v in state_dict.values()])


def vector_to_params(vec: torch.Tensor, reference_state_dict) -> dict:
    """把 1-D 向量按 reference_state_dict 的形状还原为 state_dict。"""
    new_state = {}
    offset = 0
    for k, v in reference_state_dict.items():
        numel = v.numel()
        new_state[k] = vec[offset:offset + numel].reshape(v.shape).to(v.dtype)
        offset += numel
    return new_state


# ---------------------------------------------------------------------------
# 本地训练（纯 SGD，不加噪声）
# ---------------------------------------------------------------------------

def local_train(model, loader, epochs: int, lr: float, device: str):
    """客户端本地训练，返回训练后的 state_dict（deep copy）。"""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())


@torch.no_grad()
def evaluate(model, test_loader, device: str) -> float:
    """返回测试集准确率（0~1）。"""
    model = model.to(device)
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# FedAvg 主循环
# ---------------------------------------------------------------------------

def federated_train(
    global_model,
    client_loaders,
    test_loader,
    rounds:         int   = 30,
    local_epochs:   int   = 1,
    local_lr:       float = 0.01,
    device:         str   = "cuda",
    mechanism:      str   = "none",              # "none" | "laplace" | "gaussian"
    epsilon_round:  float = float("inf"),        # 每轮隐私预算 ε_round；∞ 表示不加噪
    clip_C:         float = 9.0,                 # 梯度裁剪范数（L2），即 Δf 上界
    delta_dp:       float = 1e-5,                # (ε,δ)-DP 的 δ（Gaussian only）
    verbose:        bool  = True,
    # Rényi DP 扩展参数（由 renyi_gaussian_composition 返回后传入）
    dp_composition: str   = "none",              # "none" | "simple" | "advanced" | "renyi"
    dp_alpha:       float = 4.0,                 # Rényi order α（仅 renyi 时使用）
):
    """运行 FedAvg 联邦学习，返回 (global_model, acc_history)。

    DP 应用逻辑（对齐官方 client/Update.py add_noise）：
      1. 客户端本地训练得到 local_state
      2. 计算参数更新量 delta = local_vec - global_vec
      3. 将 delta 裁剪到 L2 球半径 clip_C（bound L2 sensitivity）
      4. 计算敏感度 sensitivity = cal_client_sensitivity(local_lr, clip_C, N)
         其中 N = len(loader.dataset)（客户端本地样本数）
      5. 用 laplace_mechanism 或 gaussian_mechanism 生成噪声并加到 delta
      6. 服务端聚合：θ_global += mean(noisy_delta)

    Args:
        acc_history: 每轮测试精度列表（0~1），用于画"精度 vs 轮数"曲线。
    """
    global_model = global_model.to(device)
    acc_history  = []

    use_dp = (mechanism != "none") and (not math.isinf(epsilon_round))

    if use_dp and verbose:
        # 以第一个 loader 的数据集大小为代表打印参数
        n0 = len(client_loaders[0].dataset)
        s0 = cal_client_sensitivity(local_lr, clip_C, n0)
        print(f"[DP] 机制={mechanism}  ε_round={epsilon_round}  "
              f"ε_total={epsilon_round * rounds:.2f}  C={clip_C}  δ={delta_dp}")
        print(f"     sensitivity(lr={local_lr}, clip={clip_C}, N={n0}) = {s0:.6f}")
    elif verbose and mechanism != "none":
        print(f"[DP] 机制={mechanism}  ε_round=∞（无噪声基线）")

    for r in range(1, rounds + 1):
        global_state = global_model.state_dict()
        global_vec   = params_to_vector(global_state)

        delta_sum = torch.zeros_like(global_vec)

        for loader in client_loaders:
            # ---- 本地训练 ----
            local_model = copy.deepcopy(global_model)
            local_state = local_train(local_model, loader,
                                      local_epochs, local_lr, device)
            delta = params_to_vector(local_state) - global_vec

            # ---- DP 扰动（对齐官方） ----
            if use_dp:
                # 1. 裁剪 delta，限制 L2 灵敏度
                delta = clip_gradient(delta, clip_C)

                # 2. 计算敏感度（对齐 cal_client_sensitivity）
                n_samples   = len(loader.dataset)
                sensitivity = cal_client_sensitivity(local_lr, clip_C, n_samples)

                # 3. 加噪（Laplace / Gaussian / Rényi Gaussian）
                if mechanism == "laplace":
                    delta = add_dp_noise_torch(
                        delta, "laplace", epsilon_round, sensitivity, delta_dp)
                elif mechanism == "gaussian":
                    if dp_composition == "renyi":
                        # Rényi Gaussian 机制（须预先调用 renyi_gaussian_composition）
                        noise_np = renyi_gaussian_mechanism(
                            alpha=dp_alpha, epsilon=epsilon_round,
                            sensitivity=sensitivity,
                            size=tuple(delta.shape))
                        import numpy as _np
                        noise = torch.from_numpy(noise_np).to(
                            device=delta.device, dtype=delta.dtype)
                        delta = delta + noise
                    else:
                        delta = add_dp_noise_torch(
                            delta, "gaussian", epsilon_round, sensitivity, delta_dp)

            delta_sum += delta

        # ---- 服务端聚合 ----
        new_vec = global_vec + delta_sum / len(client_loaders)
        global_model.load_state_dict(vector_to_params(new_vec, global_state))

        # ---- 评估 ----
        acc = evaluate(global_model, test_loader, device)
        acc_history.append(acc)
        if verbose:
            print(f"[Round {r:3d}/{rounds}] test acc = {acc * 100:.2f}%")

    return global_model, acc_history
