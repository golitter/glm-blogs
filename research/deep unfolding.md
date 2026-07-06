# Deep Unfolding

**Deep unfolding**：把“迭代算法”变成“深度网络”，把原本手工设定的更新规则、步长、阈值或先验模块变成可学习组件，从而获得兼具可解释性、速度和性能的模型驱动深度学习方法。

更直观地说：

```text
传统迭代算法：
x0 → x1 → x2 → ... → xK

Deep Unfolding：
Layer 1 → Layer 2 → Layer 3 → ... → Layer K
```

传统算法中，每一步更新通常是人工设计的；deep unfolding 则把每一步更新展开成神经网络的一层，并让其中一部分参数通过数据学习。

## 和其他方法的优缺点

| 方法 | 优点 | 缺点 |
|---|---|---|
| 传统优化算法 | 可解释性强；有明确数学结构；能嵌入物理模型和约束 | 迭代慢；步长、阈值、正则强度通常要手调；表达能力有限 |
| 普通深度网络 | 表达能力强；推理速度快；端到端训练方便 | 黑盒性强；需要较多数据；对分布变化和物理约束可能不稳定 |
| Deep Unfolding | 保留迭代算法结构；参数可学习；推理速度快；比纯黑盒更可解释；适合小数据和有物理模型的问题 | 依赖所选择的底层算法；层数固定；可学习参数可能破坏原算法的收敛性；设计复杂度高 |

因此，deep unfolding 可以看成传统优化和深度学习之间的折中：

```text
传统优化：一步步“算”答案
普通深度学习：直接“学”输入到输出
Deep Unfolding：学习如何更聪明地“迭代”
```

## 一个简单例子：把 ISTA 展开成 LISTA

考虑稀疏编码问题：

$$
\min_x \frac{1}{2}\|y - Dx\|_2^2 + \lambda \|x\|_1
$$

其中：

- $y$ 是观测信号；
- $D$ 是字典矩阵；
- $x$ 是希望恢复的稀疏表示；
- $\lambda \|x\|_1$ 用来鼓励 $x$ 稀疏。

传统 ISTA 的一次迭代为：

$$
x^{k+1}
=
\text{soft-threshold}
\left(
x^k + \tau D^T(y - Dx^k),
\lambda\tau
\right)
$$

这里有两个手工设定的重要部分：

- 步长 $\tau$；
- 阈值 $\lambda\tau$。

Deep unfolding 的做法是把 $K$ 次 ISTA 迭代展开成 $K$ 层网络：

```text
y → x0 → 第 1 层 ISTA 更新 → 第 2 层 ISTA 更新 → ... → 第 K 层 ISTA 更新 → xK
```

在 LISTA 中，可以把更新写成：

$$
x^{k+1}
=
\text{soft-threshold}_{\theta_k}
\left(
W_e y + W_s x^k
\right)
$$

其中：

- $W_e$ 负责从观测 $y$ 提取信息；
- $W_s$ 负责利用上一层的估计 $x^k$；
- $\theta_k$ 是第 $k$ 层的可学习阈值。

这里的 $W_e$ 和 $W_s$ 并不是随便引入的两个矩阵。它们可以看成从 ISTA 更新式里的 $\tau D^T$ 和 $I - \tau D^T D$ 演化而来：原始 ISTA 中这些矩阵由字典 $D$ 和步长 $\tau$ 固定决定，而 LISTA 把它们放宽成可学习参数，让网络从数据中学到更有效的更新方向。

这时，原本由人工推导和手工调参得到的更新规则，变成了可以训练的网络层。网络仍然像 ISTA 一样逐步迭代，但每一步会通过数据学习出更合适的更新方向和阈值。

可以把它理解成：

```text
ISTA：
固定公式 + 手工参数 + 多次迭代

LISTA / Deep Unfolding：
算法结构 + 可学习参数 + 固定层数前向传播
```

这个例子说明了 deep unfolding 的核心价值：它不是完全丢掉传统算法，而是把传统算法变成一个可训练、可解释、速度更快的网络。

## 去雨 demo 例子：把去雨优化过程展开成网络


把有雨图记为 $y$，干净图记为 $x$，雨层记为 $r$。一个最简单的观测模型是：

$$
y = x + r
$$

也就是说，一张有雨图可以粗略看成：

```text
有雨图 = 干净图 + 雨层
```

> 需要注意的是，$y = x + r$ 是一个教学用的简化退化模型，方便说明 deep unfolding 的基本思想。真实雨图退化可能还包含遮挡、散射、亮度变化、背景纹理混淆等因素，不一定能被严格写成干净图和雨层的简单相加。

于是去雨任务可以理解成：给定 $y$，同时估计干净图 $x$ 和雨层 $r$。

代码中每个 stage 近似对应下面这个优化问题：

$$
\min_{x,r}
\frac{1}{2}\|y - x - r\|_2^2
+ \Phi_{\text{clean}}(x)
+ \Phi_{\text{rain}}(r)
$$

其中：

- $\frac{1}{2}\|y - x - r\|_2^2$ 是数据一致性项，要求 $x + r$ 能重新合成有雨图 $y$；
- $\Phi_{\text{clean}}(x)$ 是干净图像先验，鼓励 $x$ 像自然图像；
- $\Phi_{\text{rain}}(r)$ 是雨层先验，鼓励 $r$ 像雨纹结构。

如果只看数据一致性项：

$$
\frac{1}{2}\|y - x - r\|_2^2
$$

它的残差可以写成：

$$
\text{residual} = x + r - y
$$

对 $x$ 和 $r$ 分别做一次梯度下降：

$$
\bar{x} = x - \alpha_x (x + r - y)
$$

$$
\bar{r} = r - \alpha_r (x + r - y)
$$

这里的 $\alpha_x$ 和 $\alpha_r$ 就是步长。在普通优化算法里，它们通常需要人工设定；在这个 demo 里，它们是可学习参数。

接下来，传统优化方法可能会接一个手工设计的 proximal operator。但在 deep unfolding 中，proximal operator 被替换成可训练 CNN：

严格来说，这里的 CNN 更像是 learned proximal / denoiser / prior module。它借用了 proximal operator 的位置和作用，但不一定满足传统 proximal operator 的数学性质。因此这种设计增强了表达能力和任务适应性，也通常会削弱原优化算法里严格的收敛保证。

$$
x^{k+1}
=
\bar{x}^k
+
\text{CNN}_{\text{clean}}
\left(
[y, \bar{x}^k, \bar{r}^k]
\right)
$$

$$
r^{k+1}
=
\bar{r}^k
+
\text{CNN}_{\text{rain}}
\left(
[y, \bar{x}^k, \bar{r}^k]
\right)
$$

对应到代码中的流程就是：

```text
residual = clean + rain - rainy
clean_bar = clean - step_clean * residual
rain_bar = rain - step_rain * residual

features = concat([rainy, clean_bar, rain_bar])
clean = clean_bar + CNN_clean(features)
rain = rain_bar + CNN_rain(features)
```

把这个 stage 堆叠 $K$ 次，就得到一个去雨 deep unfolding 网络：

```text
rainy image y
  ↓
初始化：clean^0 = y, rain^0 = 0
  ↓
Stage 1：数据一致性更新 + CNN 先验修正
  ↓
Stage 2：数据一致性更新 + CNN 先验修正
  ↓
...
  ↓
Stage K：数据一致性更新 + CNN 先验修正
  ↓
输出：clean^K, rain^K
```

![e216f5ecbb717d8e3e2001dd312596f7](./deep%20unfolding.assets/e216f5ecbb717d8e3e2001dd312596f7.png)

和代码模块的对应关系：

| 代码模块 | deep unfolding 含义 |
|---|---|
| `DerainUnfoldingStage` | 一次展开迭代，也就是一个 stage |
| `residual = clean + rain - rainy` | 数据一致性残差 |
| `raw_clean_step` / `raw_rain_step` | 可学习步长 |
| `clean_bar` / `rain_bar` | 梯度下降后的中间结果 |
| `ProximalCNN` | 学习到的 proximal mapping / 图像先验 |
| `DeepUnfoldingDerainNet` | 多个 stage 堆叠成完整网络 |

![4a575f56d9e49f8ec70ce6ac56a0dc91](./deep%20unfolding.assets/4a575f56d9e49f8ec70ce6ac56a0dc91.png)

这个去雨例子比 LISTA 更接近图像恢复论文里的 deep unfolding 思路：

```text
LISTA：
稀疏编码问题 → ISTA 迭代 → 可学习阈值和矩阵

去雨 demo：
图像退化模型 y = x + r
→ 数据一致性梯度步
→ CNN 学习 clean/rain 先验
→ 多个 stage 堆叠
```

它的关键点不是让 CNN 直接从有雨图“猜”干净图，而是让网络每一层都沿着一个可解释的恢复过程前进：

```text
先让 clean + rain 更接近 rainy
再用 CNN 修正 clean 和 rain 的结构
重复多次
```



代码：

```python
"""A compact deep unfolding network for image deraining.

核心想法:
    有雨图 y 可以被粗略写成 y = x + r

其中:
    x 是干净图像 clean image
    r 是雨纹 / 雨层 rain layer

我们用 K 个 stage 展开一个近似优化过程。每个 stage 包含两部分:
    1. 数据一致性更新: 让当前估计满足 x + r ≈ y
    2. 学习到的近端映射: 用小 CNN 学习自然图像先验和雨纹先验

这不是论文级大模型，而是一个便于学习 deep unfolding 的最小可训练版本。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """A tiny residual CNN block.

    残差块学习的是“修正量”，不是从零生成整张图。图像恢复任务中这种形式
    通常更稳定，也更符合 unfolding 里“每一步逐渐修正”的直觉。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ProximalCNN(nn.Module):
    """Learned proximal mapping.

    传统优化算法里，prox 通常是手写函数，例如 soft-thresholding。
    在 deep unfolding 中，我们把 prox 换成一个可训练 CNN，让网络自己学习
    什么样的图像更像 clean、什么样的结构更像 rain。
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

        # 最后一层从 0 附近开始，训练初期每个 stage 更接近“优化迭代的一小步”。
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class DerainUnfoldingStage(nn.Module):
    """One unrolled deraining stage.

    我们最小化一个示意目标:

        1/2 ||y - x - r||_2^2 + Phi_clean(x) + Phi_rain(r)

    对数据一致性项 1/2 ||y - x - r||_2^2 做一次梯度下降:

        residual = x + r - y
        x_bar = x - step_x * residual
        r_bar = r - step_r * residual

    然后用 CNN 近端映射分别修正 x_bar 和 r_bar。
    """

    def __init__(self, hidden_channels: int) -> None:
        super().__init__()

        # softplus(raw_step) 保证真正使用的步长为正数。
        # 初始 softplus(0) ≈ 0.693，小于该二次项的稳定步长上界 1。
        self.raw_clean_step = nn.Parameter(torch.tensor(0.0))
        self.raw_rain_step = nn.Parameter(torch.tensor(0.0))

        # 输入特征拼接 [有雨图 y, 当前 clean 估计 x_bar, 当前 rain 估计 r_bar]。
        # 输出是对 clean / rain 的 residual correction。
        self.clean_prox = ProximalCNN(
            in_channels=9,
            hidden_channels=hidden_channels,
            out_channels=3,
        )
        self.rain_prox = ProximalCNN(
            in_channels=9,
            hidden_channels=hidden_channels,
            out_channels=3,
        )

    def forward(
        self,
        rainy: torch.Tensor,
        clean: torch.Tensor,
        rain: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---------- 1. 显式的数据一致性梯度步 ----------
        residual = clean + rain - rainy
        clean_step = F.softplus(self.raw_clean_step)
        rain_step = F.softplus(self.raw_rain_step)

        clean_bar = clean - clean_step * residual
        rain_bar = rain - rain_step * residual

        # ---------- 2. 学习到的近端/先验修正 ----------
        features = torch.cat([rainy, clean_bar, rain_bar], dim=1)
        clean = clean_bar + self.clean_prox(features)
        rain = rain_bar + self.rain_prox(features)

        return clean, rain


class DeepUnfoldingDerainNet(nn.Module):
    """Stack multiple deraining stages into a trainable unfolding network."""

    def __init__(self, num_stages: int = 4, hidden_channels: int = 32) -> None:
        super().__init__()
        if num_stages <= 0:
            raise ValueError("num_stages must be positive.")

        self.stages = nn.ModuleList(
            DerainUnfoldingStage(hidden_channels=hidden_channels)
            for _ in range(num_stages)
        )

    def forward(self, rainy: torch.Tensor, return_all: bool = True) -> dict[str, object]:
        """Run the unrolled network.

        初始值:
            clean^0 = rainy
            rain^0  = 0

        这样第一步从“直接把有雨图当作干净图”开始，后续 stage 逐步估计并移除雨层。
        """

        clean = rainy
        rain = torch.zeros_like(rainy)

        clean_stages: list[torch.Tensor] = []
        rain_stages: list[torch.Tensor] = []

        for stage in self.stages:
            clean, rain = stage(rainy=rainy, clean=clean, rain=rain)
            clean_stages.append(clean)
            rain_stages.append(rain)

        if return_all:
            return {
                "clean": clean,
                "rain": rain,
                "clean_stages": clean_stages,
                "rain_stages": rain_stages,
            }

        return {"clean": clean, "rain": rain}


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters for logging."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)

```

