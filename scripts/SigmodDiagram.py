import numpy as np
import matplotlib.pyplot as plt

# 定义Sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成数据点
x = np.linspace(-6, 6, 100)  # 在区间(-6, 6)之间生成100个点
y = sigmoid(x)

# 创建图形
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Sigmoid', color='b')

# 加粗坐标轴
plt.axhline(0, color='black', lw=2)
plt.axvline(0, color='black', lw=2)

# 添加标题和标签
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('sigmoid(x)')
plt.legend()

# 显示网格
plt.grid(True)

# 展示图形
plt.show()