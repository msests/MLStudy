## 原理

Sigmoid函数是深度学习中常用的一种激活函数，其数学表达式为：
$$f(x) = \frac{1}{1 + e^{-x}}$$
该函数将输入值映射到(0, 1)区间内，常用于二分类问题中作为输出层的激活函数，模拟概率输出。
![](SigmodDiagram.png)
## 求导

Sigmoid函数的数学表达式为：
$$f(x)=\frac{1}{1+e^{-x}}$$
对Sigmoid函数求导，我们首先应用链式法则。Sigmoid函数的导数可以表示为：
$$f'(x)=\frac{d}{dx}\left(\frac{1}{1+e^{-x}}\right)$$
为了简化这个过程，我们可以使用以下技巧：将原函数$f(x)$重写为$f(x)=(1+e^{-x})^{-1}$，然后对其求导。根据链式法则，我们得到：
$$f'(x)=-1\cdot(1+e^{-x})^{-2}\cdot(-e^{-x})$$
这可以进一步简化为：
$$f'(x)=\frac{e^{-x}}{(1+e^{-x})^2}$$
注意到原始的Sigmoid函数$f(x)=\frac{1}{1+e^{-x}}$，我们可以将其导数表示为：
$$f'(x)=f(x)\cdot(1-f(x))$$
这是因为：
$$f(x)\cdot(1-f(x))=\frac{1}{1+e^{-x}}\cdot\left(1-\frac{1}{1+e^{-x}}\right)=\frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}=\frac{e^{-x}}{(1+e^{-x})^2}$$
因此，Sigmoid函数的导数具有一个非常简洁的形式：
$$f'(x)=f(x)\cdot(1-f(x))$$
这意味着一旦你知道了某个点的Sigmoid函数值，你就可以直接计算出该点的导数值，而无需再次执行指数运算。这种性质使得Sigmoid函数在神经网络中的反向传播算法中特别有用。

![](SigmodBackPropagation.drawio.svg)
## 优缺点

### 优点

- 输出值介于0和1之间，非常适合用于二分类问题。

- 函数及其导数易于理解和实现。

### 缺点

- **梯度消失问题**：当输入远离原点时，Sigmoid函数的梯度接近零，导致权重更新缓慢或停止更新，即所谓的“梯度消失”问题。

- **非零中心化输出**：Sigmoid函数的输出始终为正，这可能会影响后续层的训练。

- **计算量较大**：由于指数运算的存在，相较于其他简单的激活函数（如ReLU），Sigmoid的计算成本较高。