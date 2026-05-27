"""FedAvg 联邦学习主循环，支持 per-round DP。

流程（每个通信轮）：
    1. 服务器把当前全局模型下发给各客户端；
    2. 客户端本地训练（纯 SGD），得到训练后参数向量 local_vec；
    3. 计算参数更新量 delta = local_vec - global_vec，服务器对每个 delta
       做 per-round DP：先裁剪到 L2 球 C，再加校准噪声；
    4. 服务器平均所有（加噪后）delta，叠加到全局模型上，作为新的全局模型；
    5. 在测试集上评估全局模型精度。

无噪声时（mechanism="none"）即标准 FedAvg 基线。

DP 参数说明（per-round sequential composition）：
    epsilon_round : 每轮隐私预算 ε_round（∞ 表示无 DP）
    clip_C        : L2 裁剪范数，即灵敏度上界 Δf = C
    delta_dp      : (ε,δ)-DP 的失败概率 δ（Gaussian only）
    总预算        : ε_total = ε_round × rounds
"""

import copy
import math
import torch
import torch.nn as nn

from .dp import privatize, noise_std_from_epsilon


# ---------------------------------------------------------------------------
# 参数 <-> 向量 的拉平/还原（加噪、GRNN 都要用一致的顺序）
# ---------------------------------------------------------------------------
def params_to_vector(state_dict):
    """把一个 state_dict（有序）拉平成 1D 向量。"""
    return torch.cat([v.reshape(-1).float() for v in state_dict.values()])


def vector_to_params(vec, reference_state_dict):
    """把 1D 向量按 reference_state_dict 的形状还原成 state_dict。"""
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
def local_train(model, loader, epochs, lr, device):
    """客户端本地训练，返回训练后的 state_dict。"""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())


@torch.no_grad()
def evaluate(model, test_loader, device):
    """返回测试集准确率（0~1）。"""
    model = model.to(device)
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# FedAvg 主循环
# ---------------------------------------------------------------------------
def federated_train(
    global_model,
    client_loaders,
    test_loader,
    rounds: int = 30,
    local_epochs: int = 1,
    local_lr: float = 0.01,
    device: str = "cuda",
    mechanism: str = "none",              # "none" | "laplace" | "gaussian"
    epsilon_round: float = float("inf"),  # 每轮隐私预算 ε_round；∞ 表示不加噪
    clip_C: float = 9.0,                  # L2 裁剪范数
    delta_dp: float = 1e-5,               # (ε,δ)-DP 的 δ（Gaussian only）
    verbose: bool = True,
):
    """运行联邦学习，返回 (global_model, acc_history)。

    acc_history: 每轮结束后的测试精度列表，用于画"精度 vs 轮数"曲线。

    DP 在 delta（参数更新量 = local_vec - global_vec）上施加，而非完整参数向量。
    服务端平均加噪后的 delta，叠加到全局模型上：
        θ_global = θ_global + mean(privatize(θ_local_i - θ_global))

    DP 参数说明（per-round sequential composition）：
        epsilon_round : 每轮隐私预算 ε_round，总预算 ε_total = rounds × ε_round
        clip_C        : L2 裁剪范数，即灵敏度上界 Δf = C
        delta_dp      : (ε,δ)-DP 的失败概率 δ（Gaussian only）
    """
    global_model = global_model.to(device)
    acc_history = []

    use_dp = (mechanism != "none") and (not math.isinf(epsilon_round))
    if use_dp:
        sigma_lap = noise_std_from_epsilon("laplace",  epsilon_round, clip_C, delta_dp)
        sigma_gau = noise_std_from_epsilon("gaussian", epsilon_round, clip_C, delta_dp)
        if verbose:
            print(f"[DP] 机制={mechanism}  ε_round={epsilon_round}  "
                  f"ε_total={epsilon_round * rounds:.2f}  C={clip_C}  δ={delta_dp}")
            print(f"     σ_Lap={sigma_lap:.4f}  σ_Gau={sigma_gau:.4f}")
    elif verbose and mechanism != "none":
        print(f"[DP] 机制={mechanism}  ε_round=∞（无噪声基线）")

    for r in range(1, rounds + 1):
        global_state = global_model.state_dict()

        global_vec = params_to_vector(global_state)
        delta_sum = torch.zeros_like(global_vec)
        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            local_state = local_train(local_model, loader, local_epochs, local_lr, device)
            delta = params_to_vector(local_state) - global_vec

            if use_dp:
                delta = privatize(delta, mechanism, epsilon_round, clip_C, delta_dp)

            delta_sum += delta

        new_vec = global_vec + delta_sum / len(client_loaders)
        global_model.load_state_dict(vector_to_params(new_vec, global_state))

        acc = evaluate(global_model, test_loader, device)
        acc_history.append(acc)
        if verbose:
            print(f"[Round {r:3d}/{rounds}] test acc = {acc*100:.2f}%")

    return global_model, acc_history
