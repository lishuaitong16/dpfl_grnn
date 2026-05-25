"""MNIST 数据加载与联邦切分。

- 统一 pad 到 32x32（与 LeNet 输入、GRNN 上采样块对齐）。
- 提供 IID 切分：把训练集平均、随机地分给 N 个客户端。
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 统一变换：pad 到 32x32，单通道，像素值 [0,1]
_transform = transforms.Compose([
    transforms.Pad(2),       # 28 -> 32
    transforms.ToTensor(),   # 1ch, [0,1]
])


def get_datasets(root: str = "./data"):
    """下载并返回 (train_set, test_set)。首次运行会自动下载到 root。"""
    train_set = datasets.MNIST(root, train=True, download=True, transform=_transform)
    test_set = datasets.MNIST(root, train=False, download=True, transform=_transform)
    return train_set, test_set


def split_iid(train_set, num_clients: int, seed: int = 0):
    """把训练集 IID（独立同分布）均分给 num_clients 个客户端。

    返回一个长度为 num_clients 的列表，每个元素是该客户端的 Subset。
    IID 指每个客户端的数据是从全体里随机抽的，类别分布大体一致。
    """
    g = torch.Generator().manual_seed(seed)
    n = len(train_set)
    perm = torch.randperm(n, generator=g).tolist()
    per = n // num_clients
    client_subsets = []
    for i in range(num_clients):
        idx = perm[i * per:(i + 1) * per]
        client_subsets.append(Subset(train_set, idx))
    return client_subsets


def make_client_loaders(client_subsets, batch_size: int = 64):
    """为每个客户端构造 DataLoader。"""
    return [
        DataLoader(s, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
        for s in client_subsets
    ]


def make_test_loader(test_set, batch_size: int = 256):
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)


def denormalize(x):
    """像素值已是 [0,1]，直接截断返回（保留接口兼容性）。"""
    return x.clamp(0, 1)
