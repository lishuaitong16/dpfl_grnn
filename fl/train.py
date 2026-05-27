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
import numpy as np
import torch
import torch.nn as nn

from .dp import (
    cal_client_sensitivity,
    clip_gradient,
    add_dp_noise_torch,
    renyi_gaussian_mechanism,
    per_sample_clip_grads,
    add_noise_to_params,
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
    """客户端本地训练（无 DP），返回训练后的 state_dict（deep copy）。"""
    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion  = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()
    return copy.deepcopy(model.state_dict())


def local_train_dp_sgd(
    model,
    loader,
    epochs:     int,
    lr:         float,
    device:     str,
    mechanism:  str,
    epsilon:    float,
    clip_C:     float,
    delta_dp:   float = 1e-5,
    dp_composition: str = "none",
    dp_alpha:   float = 4.0,
) -> dict:
    """DP-SGD 本地训练，完全对齐官方 client/Update.py。

    每个 batch 做 per-sample gradient clipping（使用 torch.func.vmap+grad，
    无需 Opacus），训练结束后对模型参数加一次 DP 噪声。

    对齐官方的关键细节：
      • 每批裁剪阈值 = clip_C / epochs（官方 dp_clip / local_ep）
        确保整个本地训练累积的灵敏度上界为 clip_C
      • Laplace → L1 范数裁剪；Gaussian → L2 范数裁剪
      • 噪声加到每个参数层（独立地，与官方 add_noise 一致）
      • sensitivity = cal_client_sensitivity(lr, clip_C, N)

    Args:
        model:      训练前的全局模型（in-place 训练，不影响原模型，
                    因 federated_train 里已经 deepcopy）。
        loader:     客户端数据 DataLoader。
        epochs:     本地训练轮数。
        lr:         学习率。
        device:     设备字符串。
        mechanism:  "laplace" 或 "gaussian"。
        epsilon:    per-round 隐私预算 ε_round。
        clip_C:     总裁剪阈值（官方 dp_clip）。
        delta_dp:   (ε,δ)-DP 的 δ（Gaussian only）。
        dp_composition: 组合方式（"renyi" 时使用 Rényi Gaussian 噪声）。
        dp_alpha:   Rényi order α（仅 renyi 时使用）。

    Returns:
        训练 + 加噪后的 state_dict（deep copy）。
    """
    from .dp import renyi_gaussian_mechanism as _renyi_mech

    model = model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # 每 epoch 的裁剪阈值 = clip_C / epochs（对齐官方 dp_clip / local_ep）
    clip_per_epoch = clip_C / max(epochs, 1)
    clip_norm = 1 if mechanism == "laplace" else 2

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # ── per-sample 裁剪（对齐官方 perSampleClip）──
            clipped_grads = per_sample_clip_grads(
                model, x, y, clip_per_epoch, clip_norm=clip_norm)

            # ── 把裁剪后的梯度写回模型，执行 SGD step ──
            optimizer.zero_grad()
            for name, param in model.named_parameters():
                param.grad = clipped_grads[name].to(param.dtype)
            optimizer.step()

    # ── 训练结束后对参数加一次噪声（对齐官方 add_noise）──
    n_samples   = len(loader.dataset)
    sensitivity = cal_client_sensitivity(lr, clip_C, n_samples)

    if dp_composition == "renyi":
        # Rényi Gaussian
        with torch.no_grad():
            for param in model.parameters():
                noise_np = _renyi_mech(
                    alpha=dp_alpha, epsilon=epsilon,
                    sensitivity=sensitivity, size=tuple(param.shape))
                noise = torch.from_numpy(noise_np).to(
                    device=param.device, dtype=param.dtype)
                param.data += noise
    else:
        add_noise_to_params(model, mechanism, epsilon, sensitivity, delta_dp)

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
    rounds:           int   = 30,
    local_epochs:     int   = 1,
    local_lr:         float = 0.01,
    device:           str   = "cuda",
    mechanism:        str   = "none",          # "none" | "laplace" | "gaussian"
    epsilon_round:    float = float("inf"),    # 每轮隐私预算；∞ 表示不加噪
    clip_C:           float = 9.0,             # 裁剪阈值（对应官方 dp_clip）
    delta_dp:         float = 1e-5,            # (ε,δ)-DP 的 δ（Gaussian only）
    verbose:          bool  = True,
    dp_composition:   str   = "none",          # "none"|"simple"|"advanced"|"renyi"
    dp_alpha:         float = 4.0,             # Rényi order α
    # ── DP 方式选择 ──────────────────────────────────────────────────────
    per_sample_clip:  bool  = False,
    # False（默认）：delta 整体裁剪（输出扰动，不需要 Opacus）
    # True          ：per-sample gradient clipping（对齐官方，使用 torch.func）
    #                 每 batch 做 per-sample 裁剪，训练后加一次噪声
):
    """运行 FedAvg 联邦学习，返回 (global_model, acc_history)。

    ─── per_sample_clip=False（默认，输出扰动）────────────────────────────
    1. 本地正常训练（无裁剪）→ 得到 local_params
    2. delta = local_params - global_params
    3. 裁剪 delta（Laplace→L1，Gaussian→L2）
    4. 加 DP 噪声（sensitivity = cal_client_sensitivity(lr, C, N)）
    5. 服务端聚合 θ' = θ + mean(noisy_delta)

    ─── per_sample_clip=True（完全对齐官方 client/Update.py）─────────────
    1. 每 batch：per-sample gradient clipping（clip_C/epochs，torch.func vmap）
    2. 训练结束后：add_noise_to_params（每层独立加噪，对齐官方 add_noise）
    3. 服务端聚合 θ' = θ + mean(local_params - global_params)
    """
    global_model = global_model.to(device)
    acc_history  = []

    use_dp = (mechanism != "none") and (not math.isinf(epsilon_round))

    if use_dp and verbose:
        n0   = len(client_loaders[0].dataset)
        s0   = cal_client_sensitivity(local_lr, clip_C, n0)
        mode = "per-sample-clip" if per_sample_clip else "delta-clip"
        print(f"[DP] 机制={mechanism}  ε_round={epsilon_round:.4g}  "
              f"ε_total={epsilon_round * rounds:.2f}  C={clip_C}  δ={delta_dp}")
        print(f"     模式={mode}  "
              f"sensitivity(lr={local_lr},C={clip_C},N={n0})={s0:.2e}")
    elif verbose and mechanism != "none":
        print(f"[DP] 机制={mechanism}  ε_round=∞（无噪声基线）")

    for r in range(1, rounds + 1):
        global_state = global_model.state_dict()
        global_vec   = params_to_vector(global_state)
        delta_sum    = torch.zeros_like(global_vec)

        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)

            # ──────────────────────────────────────────────────────────────
            if use_dp and per_sample_clip:
                # 模式 B：per-sample gradient clipping（完全对齐官方）
                local_state = local_train_dp_sgd(
                    local_model, loader,
                    epochs=local_epochs, lr=local_lr, device=device,
                    mechanism=mechanism, epsilon=epsilon_round,
                    clip_C=clip_C, delta_dp=delta_dp,
                    dp_composition=dp_composition, dp_alpha=dp_alpha,
                )
            else:
                # 模式 A：正常训练
                local_state = local_train(
                    local_model, loader, local_epochs, local_lr, device)
            # ──────────────────────────────────────────────────────────────

            delta = params_to_vector(local_state) - global_vec

            # 模式 A 的 DP 扰动（delta-clip + 噪声）
            if use_dp and not per_sample_clip:
                clip_norm   = 1 if mechanism == "laplace" else 2
                delta       = clip_gradient(delta, clip_C, norm=clip_norm)
                n_samples   = len(loader.dataset)
                sensitivity = cal_client_sensitivity(local_lr, clip_C, n_samples)

                if mechanism == "laplace":
                    delta = add_dp_noise_torch(
                        delta, "laplace", epsilon_round, sensitivity, delta_dp)
                elif mechanism == "gaussian":
                    if dp_composition == "renyi":
                        noise_np = renyi_gaussian_mechanism(
                            alpha=dp_alpha, epsilon=epsilon_round,
                            sensitivity=sensitivity, size=tuple(delta.shape))
                        noise = torch.from_numpy(noise_np).to(
                            device=delta.device, dtype=delta.dtype)
                        delta = delta + noise
                    else:
                        delta = add_dp_noise_torch(
                            delta, "gaussian", epsilon_round, sensitivity, delta_dp)

            delta_sum += delta

        # 服务端聚合
        new_vec = global_vec + delta_sum / len(client_loaders)
        global_model.load_state_dict(vector_to_params(new_vec, global_state))

        acc = evaluate(global_model, test_loader, device)
        acc_history.append(acc)
        if verbose:
            print(f"[Round {r:3d}/{rounds}] test acc = {acc * 100:.2f}%")

    return global_model, acc_history
