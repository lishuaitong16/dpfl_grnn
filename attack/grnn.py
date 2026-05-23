"""GRNN 生成器：从随机向量同时生成"假图像"和"假标签"。

结构（论文 Figure 2 + Table 2）：
  输入 v ~ N(0,1)，维度 latent_dim。
  ┌─ 图像支（top branch / fake-data generator）
  │    FC: v -> 4x4 特征图 (起始通道 base_ch)
  │    若干 UpsamplingBlock：最近邻上采样x2 -> Conv(3,1,1) -> BN -> GLU
  │      4 -> 8 -> 16 -> 32（三个块，对应目标 32x32）
  │    末端 Conv 输出 out_channels 通道、Tanh 归一到 [-1,1]
  └─ 标签支（bottom branch / fake-label generator）
       FC -> softmax，输出 num_classes 维概率（fake label）

注意 GLU：nn.functional.glu 在通道维把特征切两半 a,b，输出 a * sigmoid(b)，
通道数减半。因此每个上采样块里 Conv 要输出 2*out_ch，GLU 后变 out_ch。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsamplingBlock(nn.Module):
    """最近邻上采样x2 -> Conv -> BN -> GLU。GLU 会把通道减半。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        # Conv 输出 2*out_ch，GLU 后变 out_ch
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch * 2)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        x = F.glu(x, dim=1)  # 通道维减半
        return x


class GRNNGenerator(nn.Module):
    def __init__(self, latent_dim=1024, num_classes=10, out_channels=1,
                 base_ch=128, target_size=32):
        """
        Args:
            latent_dim: 随机输入向量维度。
            num_classes: 标签类别数（MNIST=10）。
            out_channels: 生成图像通道（MNIST 灰度=1）。
            base_ch: 4x4 起始特征图的通道数。
            target_size: 目标图像边长（32 -> 三个上采样块）。
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.target_size = target_size

        # 需要的上采样块数：从 4 翻倍到 target_size
        n_blocks = 0
        s = 4
        while s < target_size:
            s *= 2
            n_blocks += 1
        assert s == target_size, "target_size 必须是 4 的 2 的幂次倍（如 16/32/64）"

        # ---- 图像支 ----
        self.fc_img = nn.Linear(latent_dim, base_ch * 4 * 4)
        blocks = []
        ch = base_ch
        for _ in range(n_blocks):
            out_ch = max(ch // 2, 16)
            blocks.append(UpsamplingBlock(ch, out_ch))
            ch = out_ch
        self.up_blocks = nn.Sequential(*blocks)
        self.to_img = nn.Conv2d(ch, out_channels, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

        # ---- 标签支 ----
        self.fc_label = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, v):
        # 图像支
        x = self.fc_img(v).view(v.size(0), -1, 4, 4)
        x = self.up_blocks(x)
        x = self.to_img(x)
        img = self.tanh(x)  # [-1, 1]

        # 标签支
        logits = self.fc_label(v)
        label = F.softmax(logits, dim=1)  # 软标签（概率分布）
        return img, label
