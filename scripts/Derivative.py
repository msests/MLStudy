import numpy as np
import matplotlib.pyplot as plt

# 定义x的取值范围
x = np.linspace(-10, 10, 400)
# 计算y=x**2
y = x**2
# 计算y=x**2的导数dy/dx=2*x
dy_dx = 2*x

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制y=x**2
plt.subplot(1, 2, 1)
plt.plot(x, y, label='y = x^2')
plt.title('Graph of y=x^2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 绘制y=x**2的导数dy/dx=2*x
plt.subplot(1, 2, 2)
plt.plot(x, dy_dx, label="dy/dx = 2x", color='red')
plt.title('Derivative of y=x^2')
plt.xlabel('x')
plt.ylabel('dy/dx')
plt.legend()
plt.grid(True)

# 显示图像
plt.tight_layout()
plt.show()
