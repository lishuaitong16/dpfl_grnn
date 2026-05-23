# 实验操作指南

> 本文件说明每一步需要运行什么、保存哪些文件、截哪些图，以及这些内容在报告中对应哪个章节。
> 所有命令均在 `dpfl_grnn/` 目录下执行，使用 `dpfl` Conda 环境。

---

## 概览：四个脚本与产出

| 步骤 | 脚本 | 关键产出文件 |
|------|------|-------------|
| 1 | `run_fl.sh` | `results/baseline_acc.png` |
| 2 | `run_dpfl.sh` | `results/dp_laplace_acc.png`、`results/dp_gaussian_acc.png`、`results/dp_clip_acc.png`、`results/dp_acc_data.csv` |
| 3 | `run_attack.sh` | `results/attack/recover_process.png`、`results/attack/compare.png`、`results/attack/loss_curve.png` |
| 4 | `run_defense.sh` | `results/defense/defense_laplace.png`、`results/defense/defense_gaussian.png`、`results/defense/psnr_bar.png`、`results/defense/psnr_data.csv` |

---

## 步骤一：传统联邦学习基线

```bash
bash run_fl.sh
```

### 需要保存的内容

**① 截图：终端输出（每轮精度日志）**
- 终端会打印 30 行，格式为 `[Round  X/30] test acc = XX.XX%`
- 截取完整的 30 轮日志，以及最后一行 `最终精度: XX.XX%`
- 用途：报告"实验环境与参数"章节，证明模型可以正常收敛

**② 保存图片：`results/baseline_acc.png`**
- 精度 vs 通信轮数曲线
- 用途：作为后续所有 DP 实验的**对照基线曲线**，报告"联邦学习基线"章节

### 需要记录的数据

| 参数 | 值 | 说明 |
|------|-----|------|
| 客户端数 N | 5 | 写入报告参数表 |
| 通信轮数 T | 30 | 写入报告参数表 |
| 本地训练轮 E | 1 | 写入报告参数表 |
| 本地学习率 lr | 0.01 | 写入报告参数表 |
| 最终测试精度 | 从终端读取 | 写入报告结果表，预期 ≥ 95% |

---

## 步骤二：差分隐私联邦学习参数扫描

```bash
bash run_dpfl.sh
```

> 这是耗时最长的步骤（13 组实验），建议在独立终端运行。

### 需要保存的内容

**① 截图：终端开头的"简单组合定理预算分配表"**
```
========================================================
  简单组合定理预算分配表（T=30 轮）
  ...
```
- 用途：报告"差分隐私机制实现"章节，展示 ε_round = ε_total / T 的推算

**② 截图：终端结尾的"最终精度汇总表"**
```
  ε_total    Laplace Final Acc    Gaussian Final Acc
  ...
```
- 用途：报告"DP 参数对 FL 性能的影响"章节，以表格形式对比各 ε 下精度

**③ 保存图片：`results/dp_laplace_acc.png`**
- 拉普拉斯机制下，不同 ε_total 的精度曲线对比
- 用途：报告图 —— DP 参数影响分析

**④ 保存图片：`results/dp_gaussian_acc.png`**
- 高斯机制下，不同 ε_total 的精度曲线对比
- 用途：报告图 —— DP 参数影响分析，与拉普拉斯对比

**⑤ 保存图片：`results/dp_clip_acc.png`**
- 不同裁剪范数 C（0.5 / 1.0 / 4.0）的精度曲线对比
- 用途：报告图 —— 裁剪范数 C 对精度的影响分析

**⑥ 保留文件：`results/dp_acc_data.csv`**
- 包含所有组合的逐轮精度原始数据
- 用途：若需要重新画图或调整样式，直接读 CSV，无需重跑实验

### 需要记录的数据

| 对比维度 | 参数组合 | 报告结论方向 |
|---------|---------|------------|
| 主实验：ε_total 影响 | 0.1 / 0.5 / 1.0 / 5.0 / 10.0 / ∞ | ε 越小精度越低，收敛越慢 |
| 机制对比 | Laplace vs Gaussian | 同等 ε 下高斯通常精度略高 |
| 副实验：C 影响 | C = 0.5 / 1.0 / 4.0（固定 ε=1） | C 太小过度裁剪，太大噪声大 |

---

## 步骤三：联邦学习 + GRNN 梯度攻击

```bash
bash run_attack.sh
```

### 需要保存的内容

**① 截图：终端输出（损失下降过程 + PSNR）**
- 每 100 步打印一次损失值，格式：`[iter XXXX] total=... mse=... wd=... tv=...`
- 最后一行：`平均 PSNR: XX.XX dB`
- 截取终端的**前几行 + 最后几行**（展示损失从高到低），以及 PSNR 数值
- 用途：报告"GRNN 攻击复现"章节，证明攻击收敛且 PSNR 有效

**② 保存图片：`results/attack/recover_process.png`**
- 一行图：原始图 → iter 0 → iter 100 → iter 300 → ... → iter 1999
- 用途：报告核心图 —— 展示"从噪声逐步还原出清晰图像"的过程，这是最直观的攻击效果展示

**③ 保存图片：`results/attack/compare.png`**
- 左：真实图；右：最终还原图（含 PSNR 数值标注）
- 用途：报告对比图 —— 最终还原质量展示

**④ 保存图片：`results/attack/loss_curve.png`**
- MSE 和 Wasserstein 损失随迭代次数下降的曲线（对数坐标）
- 用途：报告图 —— 证明攻击优化过程正常收敛

### 需要记录的数据

| 指标 | 来源 | 用途 |
|------|------|------|
| 最终 PSNR (dB) | 终端输出 | 报告结果表，作为无 DP 时攻击成功的量化基准 |
| 真实标签 | 终端第一行 `真实标签: [X]` | 报告附注，说明攻击的是哪个类别 |
| 最终 MSE / WD 损失值 | 终端最后一次打印 | 报告附注 |

---

## 步骤四：联邦学习 + 差分隐私 + GRNN 防御对比

```bash
bash run_defense.sh
```

> 这一步会跑 10 组攻击（5 个 ε × 2 种机制），耗时较长。

### 需要保存的内容

**① 截图：终端开头的"简单组合定理预算分配表"**
```
  ε_total    ε_round=ε_total/T    δ_total=T×δ
  ...
```
- 用途：报告"简单组合定理分析"章节，直接作为表格截图或手动整理成表

**② 保存图片：`results/defense/defense_laplace.png`**
- 多行对比大图：每行 = [真实图 | 该 ε 下的还原图 + PSNR]
- 行顺序：无DP → ε=10 → ε=1 → ε=0.5 → ε=0.1（拉普拉斯）
- 用途：报告**核心结论图** —— 直观展示 DP 保护效果

**③ 保存图片：`results/defense/defense_gaussian.png`**
- 同上，高斯机制版本
- 用途：同上，与拉普拉斯对比

**④ 保存图片：`results/defense/psnr_bar.png`**
- 柱状图：横轴 = ε_total，纵轴 = PSNR，蓝柱=拉普拉斯、橙柱=高斯
- 用途：报告图 —— 量化展示"ε 越小、PSNR 越低、保护越好"

**⑤ 截图：终端结尾的"Trade-off 总结表"**
```
  ε_total    Laplace PSNR    Gaussian PSNR
  ...
```
- 用途：报告"隐私-效用 Trade-off"章节，结合步骤二的精度数据分析

**⑥ 保留文件：`results/defense/psnr_data.csv`**
- 所有机制 × ε 组合的 PSNR 原始数值
- 用途：若需要自己重新画图，读 CSV 即可

### 需要记录的数据

| 对比维度 | 关注点 | 报告结论方向 |
|---------|--------|------------|
| PSNR vs ε_total | 各 ε 下的 PSNR 数值 | ε↓ → PSNR↓ → 攻击还原图越模糊 |
| Laplace vs Gaussian | 相同 ε 下 PSNR 差异 | 分析两种机制的防御强度差异 |
| 与步骤三对比 | 无DP 的 PSNR vs 有DP 的 PSNR | 量化 DP 带来的防护提升 |

---

## 报告用图汇总清单

最终报告中需要用到的所有图，按章节整理：

| 报告章节 | 使用的图/截图 |
|---------|-------------|
| 实验环境与参数 | 步骤一终端日志截图（显示 30 轮精度） |
| 联邦学习基线 | `baseline_acc.png` |
| 差分隐私机制实现 | `dp.py` 代码截图（可在 IDE 中截）+ 步骤二的组合定理表 |
| DP 参数对 FL 性能的影响 | `dp_laplace_acc.png`、`dp_gaussian_acc.png`、`dp_clip_acc.png`、精度汇总表截图 |
| GRNN 攻击复现 | `recover_process.png`（最重要）、`compare.png`、`loss_curve.png`、PSNR 终端截图 |
| DP 防御 GRNN 效果 | `defense_laplace.png`、`defense_gaussian.png`、`psnr_bar.png`、Trade-off 表截图 |
| 结论 | 无需新图，引用前面的图即可 |

---

## 快速检查清单

运行完所有脚本后，确认以下文件均存在：

```
results/
├── baseline_acc.png          ✓ 步骤一
├── dp_laplace_acc.png        ✓ 步骤二
├── dp_gaussian_acc.png       ✓ 步骤二
├── dp_clip_acc.png           ✓ 步骤二
├── dp_acc_data.csv           ✓ 步骤二
├── attack/
│   ├── recover_process.png   ✓ 步骤三
│   ├── compare.png           ✓ 步骤三
│   └── loss_curve.png        ✓ 步骤三
└── defense/
    ├── defense_laplace.png   ✓ 步骤四
    ├── defense_gaussian.png  ✓ 步骤四
    ├── psnr_bar.png          ✓ 步骤四
    └── psnr_data.csv         ✓ 步骤四
```
