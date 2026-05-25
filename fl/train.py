"""FedAvg 联邦学习主循环，支持可选差分隐私。

流程（每个通信轮）：
    1. 服务器把当前全局模型下发给各客户端；
    2. 每个客户端在本地数据上训练 E 个 epoch，得到本地模型；
    3. 计算本地"模型更新" delta = local_params - global_params；
    4. (可选 DP) 对 delta 做裁剪 + 加噪；
    5. 服务器对所有客户端的 delta 求平均，加回全局模型；
    6. 在测试集上评估全局模型精度。

无 DP 时（mechanism="none"）即标准 FedAvg 基线。
"""

import copy
import torch
import torch.nn as nn

from .dp import privatize, per_round_epsilon


# ---------------------------------------------------------------------------
# 参数 <-> 向量 的拉平/还原（DP 加噪、GRNN 都要用一致的顺序）
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
# 本地训练
# ---------------------------------------------------------------------------
def local_train(model, loader, epochs, lr, device):
    """在单个客户端上本地训练若干 epoch，返回训练后的 state_dict（深拷贝）。"""
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


def compute_client_gradient(model, loader, device):
    """取客户端一个样本计算梯度，返回拉平的 1D 向量（不更新模型）。

    与 attack/run_attack.py 中 compute_true_gradient 语义一致（batch_size=1），
    保证 FL 传输的梯度范数（~4.3）与攻击实验一致，C 可统一设置。
    """
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    x, y = next(iter(loader))
    x, y = x[:1].to(device), y[:1].to(device)  # 单样本，与攻击实验对齐
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss = criterion(model(x), y)
    loss.backward()
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


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
    # ---- 差分隐私参数（mechanism="none" 即关闭 DP）----
    mechanism: str = "none",      # "none" | "laplace" | "gaussian"
    total_epsilon: float = None,  # 总体隐私预算；按简单组合定理分摊到每轮
    clip_C: float = 1.0,          # 梯度裁剪范数（= 敏感度）
    dp_delta: float = 1e-5,       # 高斯机制用的 delta
    verbose: bool = True,
):
    """运行联邦学习，返回 (global_model, acc_history)。

    acc_history: 每轮结束后的测试精度列表，用于画"精度 vs 轮数"曲线。
    """
    global_model = global_model.to(device)
    acc_history = []

    # 简单组合定理：把总预算分摊到每一轮
    eps_round = None
    if mechanism != "none":
        assert total_epsilon is not None, "开启 DP 时必须给定 total_epsilon"
        eps_round = per_round_epsilon(total_epsilon, rounds)
        if verbose:
            print(f"[DP] 机制={mechanism}  总预算ε_total={total_epsilon}  "
                  f"轮数T={rounds}  单轮ε_round={eps_round:.4f}  裁剪C={clip_C}")

    for r in range(1, rounds + 1):
        global_state = global_model.state_dict()
        global_vec = params_to_vector(global_state)

        # 收集各客户端的"更新向量" delta，并做平均
        delta_sum = torch.zeros_like(global_vec)
        for loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            local_state = local_train(local_model, loader, local_epochs, local_lr, device)
            local_vec = params_to_vector(local_state)
            update = local_vec - global_vec  # 本地模型相对全局的更新

            if mechanism != "none":
                update = privatize(update, mechanism=mechanism, C=clip_C,
                                   epsilon=eps_round, delta=dp_delta)
            delta_sum += update

        avg_delta = delta_sum / len(client_loaders)
        new_vec = global_vec + avg_delta
        global_model.load_state_dict(vector_to_params(new_vec, global_state))

        acc = evaluate(global_model, test_loader, device)
        acc_history.append(acc)
        if verbose:
            print(f"[Round {r:3d}/{rounds}] test acc = {acc*100:.2f}%")

    return global_model, acc_history
