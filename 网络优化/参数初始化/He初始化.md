He初始化由何凯明（Kaiming He）等人提出，专为**ReLU及其变体（如Leaky ReLU、PReLU）​**设计，解决了Xavier初始化在非线性激活函数中方差衰减的问题。其核心是通过修正ReLU的负半轴抑制特性，保持前向和反向传播的方差稳定。

## ​**1. 核心假设**

- ​**非线性激活假设**：激活函数为ReLU（负半轴置零，正半轴线性），破坏了Xavier的对称性假设。

- ​**独立同分布**：权重和输入数据独立且均值为0。

- ​**单边方差修正**：需补偿ReLU导致的**方差减半**效应。

## ​**2. 前向传播的方差分析**

设全连接层的输入为 $\mathbf{x} \in \mathbb{R}^{n_{\text{in}}}$，权重矩阵 $\mathbf{W} \in \mathbb{R}^{n_{\text{out}} \times n_{\text{in}}}$，输出为 $\mathbf{y} = \mathbf{W} \mathbf{x}$，激活后为 $\mathbf{a} = \text{ReLU}(\mathbf{y})$。

- ​**ReLU的方差影响**：  

  ReLU将负值置零，使得激活后的输出 $a_i = \max(0, y_i)$ 的方差为原值的一半：
$$
  \text{Var}(a_i) = \frac{1}{2} \cdot \text{Var}(y_i).
  $$

- ​**保持方差稳定**：  

  为使 $\text{Var}(a_i) = \text{Var}(x_j)$，需满足：
$$
  \frac{1}{2} \cdot n_{\text{in}} \cdot \text{Var}(W) \cdot \text{Var}(x_j) = \text{Var}(x_j).
  $$
  解得：
$$
  \text{Var}(W) = \frac{2}{n_{\text{in}}}.
  $$

## ​**3. 反向传播的方差分析**

反向传播时，梯度通过激活函数的正半轴反向传播。由于ReLU的导数为0或1，梯度方差同样需修正：

- ​**梯度方差修正**：  

  反向传播的梯度方差为：
$$
  \text{Var}\left(\frac{\partial L}{\partial x_j}\right) = \frac{1}{2} \cdot n_{\text{out}} \cdot \text{Var}(W) \cdot \text{Var}\left(\frac{\partial L}{\partial y_i}\right).
  $$
  为使梯度方差稳定，仍需满足：
$$
  \text{Var}(W) = \frac{2}{n_{\text{out}}}.
  $$

- ​**折中策略**：  

  He初始化选择**仅优化前向传播**​（因反向传播实际效果较弱），直接取：
$$
  \text{Var}(W) = \frac{2}{n_{\text{in}}}.
  $$

## ​**4. 初始化公式**

- ​**正态分布**：  
  权重从 $\mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$ 采样。

- ​**均匀分布**：  
  权重在 $\left[-\sqrt{\frac{6}{n_{\text{in}}}}, +\sqrt{\frac{6}{n_{\text{in}}}}\right]$ 内均匀采样。

## ​**5. 针对卷积层的扩展**

对于卷积层，输入维度 $n_{\text{in}}$ 定义为：
$$
n_{\text{in}} = \text{kernel\_width} \times \text{kernel\_height} \times \text{in\_channels}.
$$
直接代入公式 $\text{Var}(W) = \frac{2}{n_{\text{in}}}$。

## ​**6. 变体：Leaky ReLU/PReLU修正**

若激活函数为Leaky ReLU（负半轴斜率为 $\alpha$）或PReLU，方差修正因子调整为：
$$
\text{Var}(W) = \frac{2}{(1 + \alpha^2) \cdot n_{\text{in}}}.
$$
其中 $\alpha$ 为负半轴斜率（如Leaky ReLU通常取 $\alpha=0.01$）。

## ​**7. 与Xavier初始化的对比**
| ​**特性**​       | He初始化                       | Xavier初始化                                    |
| -------------- | --------------------------- | -------------------------------------------- |
| ​**适用激活函数**​   | ReLU、Leaky ReLU等非线性激活函数     | Sigmoid、Tanh等近似线性激活函数                        |
| ​**方差约束**​     | $ \frac{2}{n_{\text{in}}} $ | $ \frac{2}{n_{\text{in}} + n_{\text{out}}} $ |
| ​**反向传播分析**​   | 仅优化前向传播，忽略反向传播约束            | 同时考虑前向和反向传播                                  |
| ​**激活函数方差修正**​ | 补偿ReLU的方差减半效应               | 无修正（假设线性对称）                                  |

## ​**8. 总结**

- ​**优点**：  
  - 显著提升ReLU网络的训练稳定性，适用于深层网络（如ResNet、VGG）。
  - 修正了ReLU导致的方差衰减问题。

- ​**局限性**：  
  - 对Sigmoid/Tanh等对称激活函数效果不如Xavier。
  - 未显式考虑反向传播约束（但实践中表现良好）。