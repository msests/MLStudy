## 原理细节

1. **残差学习**：ResNet 的核心思想是学习残差函数 $F(x) = H(x) + x$，其中 $H(x)$ 是目标映射，$x$ 是输入。通过跳跃连接（Shortcut Connection）实现恒等映射。

2. **网络结构**：

   - 基础块包含卷积层、批量归一化（BatchNorm）和激活函数（ReLU）。
   - 残差块分为两种：
     - **标准残差块**（BasicBlock）：包含两个 $3 \times 3$ 卷积层，用于浅层网络（如 ResNet-18/34）。
     - **瓶颈残差块**（BottleneckBlock）：包含 $1 \times 1$（降维）、$3 \times 3$ 和 $1 \times 1$（升维）卷积层，用于深层网络（如 ResNet-50/101/152）。

3. **跳跃连接**：若输入输出维度不同，使用 $1 \times 1$ 卷积调整维度（Projection Shortcut）；否则直接相加。

## 梯度计算与反向传播

**梯度公式**：假设残差块输出为 $H(x) = F(x, \{W_i\}) + x$，反向传播时梯度为：
$$\begin{aligned} \frac{\partial \mathcal{L}}{\partial x} &= \frac{\partial \mathcal{L}}{\partial H} \cdot \frac{\partial H}{\partial x} \\
&= \frac{\partial \mathcal{L}}{\partial H} \cdot \left( \frac{\partial F}{\partial x} + 1 \right)
\end{aligned}$$
   其中$\frac{\partial \mathcal{L}}{\partial H}$是输入梯度。

**优势**：

   - 跳跃连接的导数项 $1$ 确保梯度直接回传，缓解梯度消失。
   - 即使中间层梯度 $\frac{\partial F}{\partial x}$ 接近零，总梯度仍可通过 $1$ 有效传播。

## 优点

1. **解决退化问题**：允许训练极深层网络（如 152 层），避免准确率饱和或下降。

2. **训练加速**：跳跃连接简化优化目标，使网络更容易收敛。

3. **灵活性**：残差块可堆叠，适配不同深度需求。

4. **迁移学习友好**：预训练模型在多种任务（分类、检测、分割）中表现优异。

## 缺点

1. **计算成本**：深层 ResNet（如 ResNet-152）参数量大，推理速度较慢。

2. **过拟合风险**：极深网络在小数据集上可能依赖强正则化。

3. **冗余结构**：部分残差块在训练后贡献微弱，存在参数冗余。

4. **内存占用**：跳跃连接的加法操作需保存输入特征图，增加显存消耗。