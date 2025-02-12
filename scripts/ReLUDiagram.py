# 绘制ReLU函数图像

import numpy as np
import matplotlib.pyplot as plt

# 定义ReLU函数
def relu(x):
    return np.maximum(0, x)

# 生成数据点
x = np.linspace(-6, 6, 100)  # 在区间(-6, 6)之间生成100个点
y = relu(x)

# 创建图形
plt.figure(figsize=(8, 6))

# 加粗坐标轴
plt.axhline(0, color='black', lw=2)
plt.axvline(0, color='black', lw=2)

plt.plot(x, y, label='ReLU', color='b')

# 添加标题和标签
plt.title('ReLU Function')
plt.xlabel('x')
plt.ylabel('relu(x)')
plt.legend()

# 显示网格
plt.grid(True)

# 展示图形
plt.show()