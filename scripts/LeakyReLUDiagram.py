import matplotlib.pyplot as plt
import numpy as np

# 定义Leaky ReLU函数
def leaky_relu(x, alpha=0.05):
    return np.where(x >= 0, x, alpha * x)

# 生成x值
x = np.linspace(-10, 10, 400)

# 计算对应的y值
y = leaky_relu(x)

# 创建图像
plt.figure(figsize=(8, 6))

# 加粗坐标轴
plt.axhline(0, color='black', lw=2)
plt.axvline(0, color='black', lw=2)

plt.plot(x, y, label=r'$\alpha = 0.05$', linewidth=2, color='b')

# 添加标题和标签
plt.title('Leaky ReLU Function')
plt.xlabel('x')
plt.ylabel('Leaky ReLU(x)')
plt.legend()

# 显示网格
plt.grid(True)

# 展示图像
plt.show()