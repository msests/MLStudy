## 网络结构概述

ELMo 是一种基于双向LSTM的深度上下文词表示模型，核心结构包含**字符编码层**和**多层双向语言模型**。其网络结构分为三部分：
$$
\text{ELMo}_k = \gamma \sum_{j=0}^L s_j \cdot \mathbf{h}_{k,j}
$$
- $L$: 双向LSTM层数  
- $s_j$: 可训练的任务特定权重  
- $\gamma$: 全局缩放系数

## 1. 字符编码层（Character Encoding）

- ​**输入**：原始字符序列（如单词拆分为字符）

- ​**结构**：
  - ​**字符卷积层**  
    - 卷积核：$2048$个过滤器，尺寸为$1 \times 7$（1D卷积）
    - 步长(stride)=1，无填充
  - ​**Highway层**  
    - 2层Highway网络，增强梯度流动
  - ​**线性投影**  
    - 将输出投影到$d=512$维空间
- ​**输出**：单词的字符级表示 $\mathbf{x}_k \in \mathbb{R}^{512}$。

## 2. 双向语言模型（Bidirectional Language Model）

### 前向LSTM

- ​**输入**：字符编码 $\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_n$

- ​**隐藏层**：$L$层LSTM（通常$L=2$）
	- 每层隐藏单元数：$4096$  
	- 投影输出到$512$维（减少参数）

- ​**目标函数**：最大化前向对数似然  
$$\sum_{k=1}^n \log p(t_k | t_1, ..., t_{k-1})$$

### 后向LSTM

- ​**输入**：字符编码 $\mathbf{x}_n, \mathbf{x}_{n-1}, ..., \mathbf{x}_1$（反向序列）

- ​**隐藏层**：与前向结构对称

- ​**目标函数**：最大化后向对数似然  
  $$\sum_{k=1}^n \log p(t_k | t_{k+1}, ..., t_n)$$

### 双向联合训练

- ​**总损失函数**：  
$$\mathcal{L} = -\sum_{k=1}^n \left( \log p(t_k | t_1, ..., t_{k-1}) + \log p(t_k | t_{k+1}, ..., t_n) \right)$$

## 3. 多层表示组合

ELMo 融合各层隐藏状态生成最终词表示：

- ​**层输出**：
  - $\mathbf{h}_{k,0}$: 字符编码层输出  
  - $\mathbf{h}_{k,1}$: 第一层双向LSTM输出  
  - $\mathbf{h}_{k,2}$: 第二层双向LSTM输出

- ​**参数化组合**：  
  $$\mathbf{ELMo}_k = \gamma \left( s_0 \cdot \mathbf{h}_{k,0} + s_1 \cdot \mathbf{h}_{k,1}^{forward} + s_1 \cdot \mathbf{h}_{k,1}^{backward} + s_2 \cdot \mathbf{h}_{k,2}^{forward} + s_2 \cdot \mathbf{h}_{k,2}^{backward} \right)$$
  - $s_j$: Softmax归一化的任务特定权重  
  - $\gamma$: 预训练后固定或微调