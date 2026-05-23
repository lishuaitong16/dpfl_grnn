# 基于差分隐私的联邦学习 & GRNN 梯度攻击

大数据安全大作业（选项二）核心代码。当前已实现**联邦学习基线**和 **GRNN 梯度攻击**两块；差分隐私工具模块已就绪，扫参与防御实验脚本待补（见下方 TODO）。

## 目录结构

```
dpfl_grnn/
├── models/lenet.py            # LeNet-5，激活可切 ReLU/Sigmoid（攻击需 Sigmoid 保证二阶可导）
├── fl/
│   ├── data.py                # MNIST 加载 + pad 到 32x32 + IID 切分
│   ├── dp.py                  # 梯度裁剪 + 拉普拉斯/高斯加噪 + 简单组合定理预算
│   └── train.py               # FedAvg 主循环（可选 DP）+ 参数拉平/还原工具
├── attack/
│   ├── grnn.py                # GRNN 生成器（图像支 + 标签支）
│   ├── losses.py              # MSE + Wasserstein + TVLoss
│   └── run_attack.py          # 攻击主循环（Algorithm 1）+ PSNR
├── experiments/
│   ├── exp0_fl_baseline.py    # 阶段一：FL 基线（无 DP）
│   └── exp2_attack.py         # 阶段四：复现 GRNN 攻击
├── results/                   # 输出图像/曲线（截图来源）
└── requirements.txt
```

## 环境

```bash
pip install -r requirements.txt
# 5090 是 Blackwell 架构，torch 需装 CUDA 12.x 的较新版本，按 pytorch.org 官网命令装
```

## 运行

所有命令在 `dpfl_grnn/` 目录下执行（用 `-m` 以便正确 import）。

### 1. 联邦学习基线（无 DP）

```bash
python -m experiments.exp0_fl_baseline --clients 5 --rounds 30
```

输出：每轮测试精度 + `results/baseline_acc.png`（精度 vs 轮数曲线）。预期收敛到 ~99%。

### 2. GRNN 梯度攻击（先用 batch=1 跑通）

```bash
python -m experiments.exp2_attack --batch_size 1 --iterations 2000
```

输出到 `results/attack/`：

- `recover_process.png`：原图 vs 不同迭代步的还原图（看图像从噪点逐步成形）；
- `compare.png`：还原图 vs 原图并排 + PSNR；
- `loss_curve.png`：MSE/WD 损失下降曲线。

跑通后可加大 batch 观察还原质量随 batch 增大而下降：

```bash
python -m experiments.exp2_attack --batch_size 4 --iterations 3000
python -m experiments.exp2_attack --batch_size 8 --iterations 4000
```

> 默认攻击随机初始化（未收敛）的全局模型 —— 论文指出这种状态反而最易攻击。
> 加 `--train_iters 10` 可让模型先训练几步，模拟训练早期的全局模型。

## 关键实现说明

- **二阶可导**：攻击要算"损失对梯度的梯度"，故全局模型用 Sigmoid 激活，
  真梯度用 `create_graph=True` 计算。
- **梯度向量一致性**：真梯度与假梯度都按 `model.parameters()` 顺序拉平，保证可比。
- **GLU 通道**：上采样块里 Conv 输出 2×目标通道，GLU 后减半。
- **差分隐私顺序**：先按 L2 裁剪到 C（敏感度=C），再按机制加噪（见 `fl/dp.py`）。
- **简单组合定理**：`ε_total = T · ε_round`，开 DP 时由总预算分摊到每轮。

## 待补（TODO，对应 plan 阶段三/五）

- `experiments/exp1_dp_params.py`：扫 ε / C / 客户端数 → 精度曲线（阶段三）。
  直接复用 `federated_train(..., mechanism="laplace"/"gaussian", total_epsilon=...)`。
- `experiments/exp3_defense.py`：对真梯度用 `fl.dp.privatize` 加噪后再跑 `grnn_attack`，
  对比不同 ε 下的还原效果（阶段五），画攻击防御对比大图。

这两个脚本所需的底层函数（DP 加噪、攻击入口）都已就绪，可按需要我继续补全。

```

```
