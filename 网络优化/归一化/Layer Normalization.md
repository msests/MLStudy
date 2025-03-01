## 背景介绍

在深度神经网络训练中，**内部协变量偏移**（Internal Covariate Shift）会导致梯度消失或爆炸，从而降低训练效率。Batch Normalization（BN）通过标准化每一层的输入来缓解这一问题，但其依赖于小批量数据统计量，在以下场景中表现受限：
- 小批量或在线学习（如RNN）
- 动态网络结构（如Transformer）

Layer Normalization（LN）由[Ba et al., 2016]提出，通过对单个样本的层内神经元输出进行标准化，摆脱对批大小的依赖，特别适合序列模型和动态网络。

## 原理细节

LN和BN在理论上没有什么差别，主要是实现上的不同。LN用于解决BN无法使用在RNN/Transformer中。
### 标准化过程

给定输入向量$\mathbf{x} \in \mathbb{R}^H$（H为层中神经元数量），LN按如下步骤处理：

1. **计算均值与方差**  
$$\mu = \frac{1}{H}\sum_{i=1}^{H}x_i$$
$$\sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(x_i - \mu)^2 + \epsilon$$
   （$\epsilon$为数值稳定项，通常取$1e-5$）

2. **归一化**  
   $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2}}$$
3. **仿射变换**  
   $$y_i = \gamma \hat{x}_i + \beta$$  
   （$\gamma$和$\beta$为可学习的缩放与平移参数）

### 图示说明
- **BN**：对同一特征跨样本标准化
- **LN**：对同一样本跨特征标准化

## 梯度计算与反向传播
### 关键梯度

1. **对输入$x_i$的梯度**  
   $$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial y_i} \cdot \frac{\gamma}{\sqrt{\sigma^2}} + \frac{\partial \mathcal{L}}{\partial \mu} \cdot \frac{1}{H} + \frac{\partial \mathcal{L}}{\partial \sigma^2} \cdot \frac{2(x_i - \mu)}{H}$$

2. **对参数$\gamma$和$\beta$的梯度**  
$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{H} \frac{\partial \mathcal{L}}{\partial y_i} \hat{x}_i$$
$$\frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{H} \frac{\partial \mathcal{L}}{\partial y_i}$$

### 反向传播特点

- 梯度计算依赖单个样本的统计量，与批大小无关
- 归一化操作使梯度更稳定，缓解梯度消失/爆炸


