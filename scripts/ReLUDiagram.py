# 绘制ReLU函数图像

import numpy as np
import matplotlib.pyplot as plt

# 定义ReLU函数


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# 生成数据点
x = np.linspace(-6, 6, 100)  # 在区间(-6, 6)之间生成100个点
y = relu(x)
y_derivative = relu_derivative(x)

# 创建图形
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

# 加粗坐标轴
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

plt.xlim(-6, 6)
plt.ylim(-1, 6)

plt.plot(x, y, label='Activation', color='#E63829', linewidth=3)

# 添加标题和标签
plt.title('ReLU Function')
plt.legend()

# 显示网格
# 辅助网格线
plt.grid(True,
         linestyle=':',
         alpha=0.6,
         color=(0.4, 0.4, 0.4))

plt.subplot(1, 2, 2)

# 加粗坐标轴
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

plt.xlim(-6, 6)
plt.ylim(-1, 6)

plt.plot(x, y_derivative, label='Derivative', color='#2E5C87', linewidth=3)

# 添加标题和标签
plt.title('Derivative of ReLU')
plt.legend()

# 辅助网格线
plt.grid(True,
         linestyle=':',
         alpha=0.6,
         color=(0.4, 0.4, 0.4))

# 展示图形
plt.savefig("Activation/ReLUDiagram.png")
