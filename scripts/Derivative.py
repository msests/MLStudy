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
plt.plot(x, y, label=r'$y = x^2$')
plt.title(r'Graph of $y=x^2$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

x_p = np.array([3, 5])
y_p = x_p**2

plt.plot(x_p, y_p, color='red', marker='o', markersize=3)

x_vert = np.array([5, 5])
y_vert = np.array([9, 25])
plt.plot(x_vert, y_vert, color='green', linestyle='--')
plt.text(5.2, 14.5, r'$\Delta y = f(x+\Delta x) - f(x)$',
         fontsize=12, verticalalignment='bottom')

x_horz = np.array([3, 5])
y_horz = np.array([9, 9])
plt.plot(x_horz, y_horz, color='green', linestyle='--')
plt.text(3.5, 5.5, r'$\Delta x$', fontsize=12, horizontalalignment='left')

plt.text(2, 7.5, r'$(x_0,y_0)$', fontsize=12, horizontalalignment='left')

plt.xlim(-10, 10)
plt.ylim(0, 50)

ax = plt.gca()
ax.set_xticklabels([])
ax.set_yticklabels([])

# 显示图像
plt.tight_layout()
plt.savefig('Calculus/Derivative.png')
