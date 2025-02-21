## 原理细节

Softmax回归（多类逻辑回归）用于多分类问题。给定输入特征向量 $x \in \mathbb{R}^n$ 和类别标签 $y \in \{1,2,...,K\}$，假设函数定义为：
$$
\mathbf{h_\theta(x)} = \begin{bmatrix}
P(y=1|x;\theta) \\
P(y=2|x;\theta) \\
\vdots \\
P(y=K|x;\theta)
\end{bmatrix} = \frac{1}{\sum_{k=1}^K e^{\theta_k^T x}} \begin{bmatrix}
e^{\theta_1^T x} \\
e^{\theta_2^T x} \\
\vdots \\
e^{\theta_K^T x}
\end{bmatrix}
$$
其中 $\theta = [\theta_1, \theta_2, ..., \theta_K]^T$ 是权重矩阵，$\theta_k \in \mathbb{R}^n$ 对应第 $k$ 类的参数向量。

## 损失函数

Softmax回归一般使用交叉熵损失函数，带L2正则化的损失公式如下：
$$
J(\theta) = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log h_\theta(x^{(i)})_k + \frac{\lambda}{2m} \sum_{k=1}^K \sum_{j=1}^n \theta_{kj}^2
$$
其中$m$是样本个数，$k$是类别个数， $y_k^{(i)}$ 是示性函数（当样本 $i$ 属于类 $k$ 时为1），$\lambda$ 是正则化系数。

## 求导过程

对单个样本 $(x^{(i)}, y^{(i)})$ 的损失求导：

1. 计算预测概率：
$$
\frac{\partial h_k}{\partial \theta_j} = h_k(\delta_{kj} - h_j)x
$$
其中 $\delta_{kj}$ 是Kronecker delta函数

2. 交叉熵损失对 $\theta_j$ 的梯度：
$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})_j - y_j^{(i)})x^{(i)} + \frac{\lambda}{m}\theta_j
$$

## 权重参数更新

采用梯度下降法更新参数：
$$
\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)})_j - y_j^{(i)})x^{(i)} + \frac{\lambda}{m}\theta_j \right)
$$
其中 $\alpha$ 是学习率

## 应用场景

1. 多类别分类问题（K ≥ 3）

2. 图像分类（如MNIST手写数字识别）

3. 自然语言处理中的标签预测

4. 推荐系统中的多类别物品推荐

## 优点

1. 输出概率解释性强

2. 可处理线性可分或近似线性可分的数据

3. 计算效率高，适合大规模数据

4. 天然支持多分类，无需组合多个二分类器

## 缺点

1. 只能学习线性决策边界

2. 假设各类别之间互斥

3. 对特征间的复杂非线性关系建模能力有限

4. 类别数量极大时计算成本显著增加

## 参考资料
5. 《Deep Learning》Ian Goodfellow et al., Chapter 3.12
6. 《Pattern Recognition and Machine Learning》Christopher Bishop, Chapter 4.3
7. 《机器学习》周志华, 第3章
8. [CS229 Notes (Stanford)](http://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes1.pdf)