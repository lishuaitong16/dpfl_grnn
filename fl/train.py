"""FedAvg 联邦学习主循环，支持 DP-SGD 风格的梯度噪声注入。

流程（每个通信轮）：
    1. 服务器把当前全局模型下发给各客户端；
    2. 客户端本地训练：每步 loss.backward() 后、optimizer.step() 前对梯度加噪；
    3. 客户端上传 delta = 训练后参数 - 全局参数；
    4. 服务器平均所有 delta，更新全局模型；
    5. 在测试集上评估全局模型精度。

无噪声时（mechanism="none"）即标准 FedAvg 基线。
"""

import copy
import torch
import torch.nn as nn

from .dp import privatize


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
# 本地训练（含可选梯度噪声）
# ---------------------------------------------------------------------------
def local_train(model, loader, epochs, lr, device, mechanism="none", noise_std=0.0):
    """客户端本地训练，每步梯度算完后可选加噪再 step。"""
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
            if mechanism != "none":
                flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
                flat = privatize(flat, mechanism=mechanism, noise_std=noise_std)
                offset = 0
                for p in model.parameters():
                    numel = p.grad.numel()
                    p.grad.data = flat[offset:offset + numel].reshape(p.grad.shape)
                    offset += numel
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
    mechanism: str = "none",   # "none" | "laplace" | "gaussian"
    noise_std: float = 0.0,    # 注入到每步梯度上的物理噪声标准差
    verbose: bool = True,
):
    """运行联邦学习，返回 (global_model, acc_history)。

    acc_history: 每轮结束后的测试精度列表，用于画"精度 vs 轮数"曲线。
    """
    global_model = global_model.to(device)
    acc_history = []

    if mechanism != "none" and verbose:
        print(f"[Noise] 机制={mechanism}  grad_noise_std={noise_std:g}  不裁剪")

    for r in range(1, rounds + 1):
        global_state = global_model.state_dict()
        global_vec = params_to_vector(global_state)

        delta_sum = torch.zeros_like(global_vec)
        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            local_state = local_train(local_model, loader, local_epochs, local_lr, device,
                                      mechanism=mechanism, noise_std=noise_std)
            local_vec = params_to_vector(local_state)
            delta_sum += local_vec - global_vec

        new_vec = global_vec + delta_sum / len(client_loaders)
        global_model.load_state_dict(vector_to_params(new_vec, global_state))

        acc = evaluate(global_model, test_loader, device)
        acc_history.append(acc)
        if verbose:
            print(f"[Round {r:3d}/{rounds}] test acc = {acc*100:.2f}%")

    return global_model, acc_history
