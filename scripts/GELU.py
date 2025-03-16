import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# 定义GELU函数


def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

# 定义GELU的导数


def gelu_derivative(x):
    phi_x = 0.5 * (1 + erf(x / np.sqrt(2)))  # 正态分布的CDF
    pdf_x = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)  # 正态分布的PDF
    return phi_x + x * pdf_x


# 生成x轴数据
x = np.linspace(-4, 4, 1000)

# 计算GELU和导数值
y_gelu = gelu(x)
y_derivative = gelu_derivative(x)

# 创建图像
plt.figure(figsize=(12, 5))

# 绘制GELU函数
plt.subplot(1, 2, 1)
plt.plot(x, y_gelu, label='GELU', color='blue')
plt.title('GELU Activation Function')
plt.xlabel('x')
plt.ylabel('GELU(x)')
plt.grid(True)
plt.legend()

# 绘制导数图像
plt.subplot(1, 2, 2)
plt.plot(x, y_derivative, label='Derivative', color='red')
plt.title('Derivative of GELU')
plt.xlabel('x')
plt.ylabel('d/dx GELU(x)')
plt.grid(True)
plt.legend()

# 调整布局并显示
plt.tight_layout()
plt.savefig("GELU.png")
