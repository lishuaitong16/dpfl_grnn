# GRNN 攻击数学推导

## 符号定义

| 符号 | 含义 |
|---|---|
| $\theta$ | 全局模型参数（冻结，不更新） |
| $\phi$ | 生成器参数（被优化） |
| $v$ | 固定随机输入向量 |
| $x_{real}$ | 受害者真实图像 |
| $y_{real}$ | 受害者真实标签 |
| $f_\theta$ | 全局模型（LeNet + Sigmoid） |
| $G_\phi$ | 生成器网络 |

---

## 攻击流程

### 第一步：计算真实梯度（被服务器截获）

$$g_{true} = \nabla_{\theta}\ \mathcal{L}\bigl(f_\theta(x_{real}),\ y_{real}\bigr)$$

- 对**模型参数 $\theta$** 求导，结果是一个固定的常量向量（61706 维）
- 算完之后 `.detach()`，后续不再对它求导
- 不需要 `create_graph=True`

---

### 第二步：生成器产出假数据

$$(\hat{x},\ \hat{y}) = G_\phi(v)$$

- $\hat{x} \in [0,1]^{1 \times 32 \times 32}$：生成的假图像
- $\hat{y} \in \Delta^{10}$：生成的假软标签（softmax 输出）

---

### 第三步：计算假梯度

先对假图像做归一化，与真实梯度的计算空间对齐：

$$\tilde{x} = \frac{\hat{x} - 0.1307}{0.3081}$$

再计算假梯度：

$$g_{fake} = \nabla_{\theta}\ \mathcal{L}\bigl(f_\theta(\tilde{x}),\ \hat{y}\bigr)$$

- 同样是对**模型参数 $\theta$** 求导
- 但 $\hat{x}$ 是生成器的输出，所以 $g_{fake}$ 是 $\phi$ 的函数
- **必须使用 `create_graph=True`**，原因见第五步

---

### 第四步：攻击损失

$$\mathcal{L}_{attack} = \underbrace{\|g_{fake} - g_{true}\|^2}_{\text{MSE}} + \underbrace{\|g_{fake} - g_{true}\|_2}_{\text{L2 norm}} + \underbrace{\alpha \cdot TV(\hat{x})}_{\text{平滑正则}}$$

---

### 第五步：对生成器求导（链式法则展开）

优化目标是生成器参数 $\phi$，由链式法则：

$$\frac{\partial \mathcal{L}_{attack}}{\partial \phi} = \frac{\partial \mathcal{L}_{attack}}{\partial g_{fake}} \cdot \frac{\partial g_{fake}}{\partial \hat{x}} \cdot \frac{\partial \hat{x}}{\partial \phi}$$

其中中间项展开为：

$$\frac{\partial g_{fake}}{\partial \hat{x}} = \frac{\partial}{\partial \hat{x}} \nabla_\theta\ \mathcal{L}\bigl(f_\theta(\hat{x}),\ \hat{y}\bigr) = \frac{\partial^2 \mathcal{L}}{\partial \theta\ \partial \hat{x}}$$

这是关于 $\theta$ 和 $\hat{x}$ 的**混合二阶偏导数**。

PyTorch 要计算这一项，必须知道 $g_{fake}$ 关于 $\hat{x}$ 的计算图。  
因此计算 $g_{fake}$ 时必须指定 `create_graph=True`，否则计算图被释放，梯度无法继续往回传。

---

## 反传路径图

```
L_attack
    ↓  ∂L/∂g_fake
g_fake
    ↓  ∂g_fake/∂x̂  =  ∂²L/∂θ∂x̂   ← 二阶导，需要 create_graph=True
x̂  (fake_img)
    ↓  ∂x̂/∂ϕ
生成器参数 ϕ  ← optimizer.step() 更新这里
```

全局模型 $\theta$ 出现在链式法则的中间，但**它不是求导的目标**，梯度流过它但不更新它。

---

## 各变量求导汇总

| 量 | 对谁求导 | 用途 | create_graph |
|---|---|---|---|
| $g_{true}$ | 对 $\theta$ 求导 | 作为攻击目标（常量） | 不需要 |
| $g_{fake}$ | 对 $\theta$ 求导 | 需继续对 $\hat{x}$ 求导 | **需要** |
| $\mathcal{L}_{attack}$ | 对 $\phi$ 求导 | 更新生成器 | 自动（`.backward()`） |

---

## 为什么模型必须用 Sigmoid 而不能用 ReLU

二阶导数 $\dfrac{\partial^2 \mathcal{L}}{\partial \theta\ \partial \hat{x}}$ 要求模型的激活函数**处处二阶可导**。

- **Sigmoid**：$\sigma(x) = \frac{1}{1+e^{-x}}$，处处光滑，二阶导数存在
- **ReLU**：$\max(0, x)$，在 $x=0$ 处不可导，大量神经元梯度为零（死区），二阶导数几乎处处为零，梯度无法有效回传
