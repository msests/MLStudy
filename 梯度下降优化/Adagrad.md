## 原理

Adagrad的核心思想是为每个参数维护一个历史梯度平方和，并基于这个累积值来调整学习率。对于具有较大历史梯度平方和的参数，其学习率会减小；而对于那些历史梯度平方和较小的参数，则允许有较大的学习率。

## 梯度更新

给定时间步$t$时的参数$\theta_t$，在$t+1$时刻的更新规则如下：
$$\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t, ii} + \epsilon}} \cdot g_{t, i} $$
- $g_{t, i}$ 表示在第$t$次迭代时对参数$\theta_i$计算得到的梯度。
- $G_t \in \mathbb{R}^{d \times d}$ 是一个对角矩阵，其每个对角元素$i, i$是直到时间步$t$为止所有先前梯度关于$\theta_i$的平方和。更新公式$G_t = G_{t - 1}+\left(\nabla_{\theta_t}L\right)^2$
- $\eta$ 是初始学习率。
- $\epsilon$ 是为了数值稳定性而添加的小常数（通常设置为$10^{-8}$），以避免除零错误。

## 优点

- **无需手动设置学习率**：Adagrad自动调整学习率，这减少了调参的工作量。

- **适合稀疏数据**：对于稀疏特征的数据集表现良好，因为它能够给予罕见但信息量大的特征更大的更新步骤。

## 缺点

- **学习率单调递减**：由于Adagrad使用了所有先前梯度的平方和，这可能导致学习率过快地变得太小，以至于最终几乎停止学习。

- **全局学习率的影响**：尽管Adagrad能自适应地调整每个参数的学习率，但它仍然依赖于全局学习率$\eta$的选择。如果$\eta$选择不当，可能会影响性能。