
## 基础Seq2Seq架构

- ​**编码器**：将输入序列 $\mathbf{X} = (x_1, ..., x_T)$ 编码为上下文向量 $\mathbf{c}$
$$h_t = f_{enc}(x_t, h_{t-1}),\quad \mathbf{c} = q(h_1,...,h_T)$$
   通常取最后一个隐藏状态 $\mathbf{c} = h_T$

- ​**解码器**：基于 $\mathbf{c}$ 生成输出序列 $\mathbf{Y} = (y_1,...,y_{T'})$
$$s_i = f_{dec}(y_{i-1}, s_{i-1}, \mathbf{c})$$
$$P(y_i|y_{<i}, \mathbf{X}) = g(y_{i-1}, s_i, \mathbf{c})$$

## 注意力机制动机

- ​**传统模型缺陷**：
	1. 上下文向量成为信息瓶颈
	2. 长序列信息丢失严重
	3. 缺乏对输入序列的动态关注

- ​**核心思想**：解码时动态关注输入序列的不同部分

## 注意力机制原理

### 计算注意力权重

对于解码器时刻 $i$：
$$e_{ij} = a(s_{i-1}, h_j)$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^T \exp(e_{ik})}$$

其中：
- $a$ 是注意力评分函数
- 常用评分方式：
  - 加性注意力：$a(s,h) = \mathbf{v}^\top \tanh(\mathbf{W}_1 s + \mathbf{W}_2 h)$
  - 乘性注意力：$a(s,h) = s^\top \mathbf{W} h$

### 生成上下文向量
$$\mathbf{c}_i = \sum_{j=1}^T \alpha_{ij} h_j$$

### 解码器改进
$$s_i = f_{dec}(y_{i-1}, s_{i-1}, \mathbf{c}_i)$$
$$P(y_i|y_{<i}, \mathbf{X}) = g(y_{i-1}, s_i, \mathbf{c}_i)$$

## 注意力机制优势

1. 解决信息瓶颈问题
2. 支持显式的对齐学习
3. 提升长序列处理能力
4. 提供可解释的注意力分布

## 常见变体

| 类型 | 公式 | 特点 |
|------|------|------|
| 加性注意力 | $e_{ij} = \mathbf{v}^\top \tanh(\mathbf{W}_1 s_{i-1} + \mathbf{W}_2 h_j)$ | 计算稳定 |
| 乘性注意力 | $e_{ij} = s_{i-1}^\top \mathbf{W} h_j$ | 计算高效 |
| 缩放点积 | $e_{ij} = \frac{s_{i-1}^\top h_j}{\sqrt{d}}$ | Transformer使用 |
