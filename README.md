# DP-FL + GRNN 梯度反演攻击实验

研究差分隐私（DP）对联邦学习（FL）模型效用的影响，以及 DP 噪声对 GRNN 梯度反演攻击的防御效果。

**数据集**：MNIST（pad 至 32×32）｜**模型**：LeNet-5（61,706 参数）｜**框架**：FedAvg，5 客户端 IID，T=30 轮

---

## 主要结论

| 实验 | 结论 |
|------|------|
| DP-FL 精度 | 拉普拉斯 ε_round=50 精度 96.8%（基线 97.3%，仅降 0.5%）；相同 ε 下高斯噪声约大 3.4×，影响更显著 |
| GRNN 攻击 | 无 DP 时 PSNR ≈ 40 dB，人眼无法区分还原图与原图；攻击效果在训练早期（0~10步）最强 |
| DP 防御 | 任意有限 DP 噪声均可使 PSNR 降至 < 10 dB（攻击彻底失败），根本原因是高维梯度空间噪声范数放大 √d ≈ 248 倍 |
| 双赢区间 | 拉普拉斯 ε_round ∈ [25, 100]（C=1.0）：FL 精度 93%~97%，GRNN 攻击完全失败 |

---

## 环境配置

```bash
pip install -r requirements.txt
```

依赖：`torch` · `torchvision` · `numpy` · `matplotlib` · `tqdm` · `pillow` · `scipy`

MNIST 数据集首次运行时自动下载至 `./data/`。

---

## 快速运行

五个实验按顺序执行，所有命令在 `dpfl_grnn/` 目录下运行：

```bash
bash run_fl.sh          # 步骤一：FL 基线（无 DP）
bash run_dpfl.sh        # 步骤二：DP 噪声扫描（ε_round × 2 种机制）
bash run_attack.sh      # 步骤三：GRNN 攻击基线（无 DP）
bash run_defense.sh     # 步骤四：DP 防御 GRNN
bash run_train_stage.sh # 步骤五：训练阶段 vs 攻击效果
```

也可以直接调用单个实验脚本：

```bash
python -m experiments.exp0_fl_baseline --clients 5 --rounds 30 --gpu 0
python -m experiments.exp1_dp_params   --epsilon_round 10,25,50,75,100,inf --clip_C 1.0 --gpu 0
python -m experiments.exp2_attack      --batch_size 1 --iterations 2000 --gpu 0
python -m experiments.exp3_defense     --epsilon_round 100,500,1000,2000,5000,inf --clip_C 0.05 --gpu 0
python -m experiments.exp4_train_stage --train_epochs_list 0,1,3,5,10,20,30,50 --gpu 0
```

---

## 输出文件

```
results/
├── baseline_acc.png          # 步骤一：FedAvg 精度曲线
├── dp_laplace_acc.png        # 步骤二：Laplace DP 精度曲线
├── dp_gaussian_acc.png       # 步骤二：Gaussian DP 精度曲线
├── dp_acc_data.csv           # 步骤二：精度原始数据
├── attack/
│   ├── recover_process.png   # 步骤三：还原过程（iter 0/100/300/1000/2000）
│   ├── compare.png           # 步骤三：原图 vs 还原图对比
│   └── loss_curve.png        # 步骤三：MSE/WD/TV 损失曲线
├── defense/
│   ├── defense_laplace.png   # 步骤四：Laplace 防御对比图
│   ├── defense_gaussian.png  # 步骤四：Gaussian 防御对比图
│   ├── psnr_bar.png          # 步骤四：PSNR 柱状图
│   └── psnr_data.csv         # 步骤四：PSNR 原始数据
└── train_stage/
    ├── recovered.png         # 步骤五：各训练阶段还原图
    ├── stage_curve.png       # 步骤五：梯度范数 & PSNR 双轴曲线
    └── stage_data.csv        # 步骤五：原始数据
```

---

## 项目结构

```
dpfl_grnn/
├── models/
│   └── lenet.py          # LeNet-5，激活函数可切换（ReLU / Sigmoid）
├── fl/
│   ├── data.py           # MNIST 加载 + pad 28→32 + IID 切分
│   ├── dp.py             # 梯度裁剪 + Laplace/Gaussian 加噪 + ε→σ 推导
│   └── train.py          # FedAvg 主循环（支持 per-round DP）
├── attack/
│   ├── grnn.py           # GRNN 生成器（图像支 GLU 上采样 + 标签支 Softmax）
│   ├── losses.py         # 攻击损失：MSE + WD + TV 正则
│   └── run_attack.py     # 攻击主循环 + PSNR 评估
├── experiments/
│   ├── exp0_fl_baseline.py
│   ├── exp1_dp_params.py
│   ├── exp2_attack.py
│   ├── exp3_defense.py
│   └── exp4_train_stage.py
├── run_fl.sh / run_dpfl.sh / run_attack.sh / run_defense.sh / run_train_stage.sh
└── requirements.txt
```

---

## 关键参数说明

### DP 参数（步骤二）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epsilon_round` | `10,25,50,75,100,inf` | 每轮隐私预算 ε_round，`inf` 为无 DP 基线 |
| `--clip_C` | `1.0` | L2 裁剪范数，灵敏度上界 Δf = C |
| `--delta` | `1e-5` | (ε,δ)-DP 的 δ（Gaussian 机制） |

噪声标准差推导：拉普拉斯 `σ = C√2 / ε_round`，高斯 `σ = C√(2·ln(1.25/δ)) / ε_round`；总预算 `ε_total = T × ε_round`（顺序合成定理，T=30）。

### 防御实验（步骤四）

使用 `--clip_C 0.05`（略大于实际 delta 范数 ≈ 0.044），ε_round 扫描大范围（100~5000）以覆盖从强保护到极弱保护的全区间。

### GRNN 攻击参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--iterations` | `2000` | 优化迭代次数 |
| `--tv_alpha` | `1e-3` | TV 正则权重 |
| `--batch_size` | `1` | 攻击目标的 batch 大小 |

优化器：RMSprop，lr=1e-4，momentum=0.99。

---

## 实现注意事项

**激活函数选择**：FL 训练（步骤一/二）使用 `act="relu"` 以获得更好的收敛性；GRNN 攻击（步骤三/四/五）使用 `act="sigmoid"`，因为攻击需要计算梯度的梯度（`create_graph=True`），要求激活函数处处二阶可导，ReLU 在 x=0 处不满足此条件。

**梯度拉平顺序**：`fl/train.py` 的 `params_to_vector()` 与 `attack/run_attack.py` 的梯度拼接均按 `model.parameters()` 的固定顺序迭代，两侧必须保持一致，否则攻击失败。

**DP 加噪位置**：噪声在服务器聚合前施加到每个客户端的参数差值 `Δ_k = θ_local - θ_global` 上（per-round DP），而非每步 SGD 梯度上（per-step DP）。

---

## 参考文献

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017
- Ren et al., *GRNN: Generative Regression Neural Network — A Data Leakage Attack for Federated Learning*, ACM TIST 2022
- Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016
- Dwork et al., *Calibrating Noise to Sensitivity in Private Data Analysis*, TCC 2006
