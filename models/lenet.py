"""LeNet-5 用于 MNIST 10 分类。

关键设计：激活函数可在 ReLU / Sigmoid 之间切换。
- 联邦学习正常训练时用 ReLU（收敛快、精度高）。
- GRNN 攻击时必须用 Sigmoid：攻击要算"损失对梯度的梯度"（二阶导），
  ReLU 在 0 点不二阶可导，会导致攻击不稳定甚至报错。这一点与 DLG/iDLG 一致。

输入约定：MNIST 原始为 1x28x28，本项目统一 pad 到 1x32x32，
方便 GRNN 生成器用 4->8->16->32 三个上采样块对齐分辨率。
"""

import torch.nn as nn


def _act(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"未知激活函数: {name}")


class LeNet(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1, act: str = "relu"):
        """
        Args:
            num_classes: 类别数，MNIST 为 10。
            in_channels: 输入通道，MNIST 灰度图为 1。
            act: 激活函数，"relu"（训练）或 "sigmoid"（攻击）。
        """
        super().__init__()
        self.act_name = act
        # 输入 1x32x32
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=0),  # -> 6x28x28
            _act(act),
            nn.MaxPool2d(kernel_size=2, stride=2),                          # -> 6x14x14
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),           # -> 16x10x10
            _act(act),
            nn.MaxPool2d(kernel_size=2, stride=2),                          # -> 16x5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            _act(act),
            nn.Linear(120, 84),
            _act(act),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


def build_lenet(num_classes: int = 10, in_channels: int = 1, act: str = "relu") -> LeNet:
    """工厂函数，方便实验脚本统一构造模型。"""
    return LeNet(num_classes=num_classes, in_channels=in_channels, act=act)
