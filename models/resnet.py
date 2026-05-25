"""ResNet-18 用于 MNIST / 灰度图分类。

与 LeNet 保持相同接口：
  - in_channels: 输入通道数（MNIST 灰度=1）
  - act: "relu"（训练）或 "sigmoid"（GRNN 攻击，需要二阶可导）

参考官方 GRNN Backbone/resnet.py，调整为单通道输入并统一激活函数。
"""

import torch.nn as nn
from models.lenet import _act


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, act="sigmoid"):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            _act(act),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = _act(act)

    def forward(self, x):
        return self.act(self.residual_function(x) + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, in_channels=1, act="sigmoid"):
        super().__init__()
        self.in_ch = 64
        self.act_name = act

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            _act(act),
        )
        self.conv2_x = self._make_layer(64,  2, stride=1, act=act)
        self.conv3_x = self._make_layer(128, 2, stride=2, act=act)
        self.conv4_x = self._make_layer(256, 2, stride=2, act=act)
        self.conv5_x = self._make_layer(512, 2, stride=2, act=act)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride, act):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_ch, out_channels, stride=s, act=act))
            self.in_ch = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_resnet18(num_classes=10, in_channels=1, act="sigmoid"):
    return ResNet18(num_classes=num_classes, in_channels=in_channels, act=act)
