import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成随机数据点
np.random.seed(0)
X = 2 * np.random.rand(100, 2)

# 假设真实的平面方程为 y = 4 + 3*X1 + 5*X2 + 噪声
y = 4 + 3 * X[:, 0] + 5 * X[:, 1] + np.random.randn(100)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 获取最佳拟合平面的系数（斜率）和截距
coefficients = model.coef_
intercept = model.intercept_

# 创建一个网格来表示拟合平面
x1_grid = np.linspace(0, 2, 100)
x2_grid = np.linspace(0, 2, 100)
x1_grid, x2_grid = np.meshgrid(x1_grid, x2_grid)

# 根据拟合平面的方程计算Z值
z_grid = coefficients[0] * x1_grid + coefficients[1] * x2_grid + intercept

# 绘制原始数据点和拟合平面
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color='blue', s=30, label='Data Points')  # 原始数据点
ax.plot_surface(x1_grid, x2_grid, z_grid, color='red', alpha=0.5)  # 拟合平面

# 添加自定义图例条目
scatter_proxy = plt.Line2D([0], [0], linestyle="none", marker='o', color='blue')
surface_proxy = plt.Line2D([0], [0], linestyle="none", marker='s', color='red', alpha=0.5)
ax.legend([scatter_proxy, surface_proxy], ['Data Points', 'Best Fit Plane'])

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.title('Linear Regression in 3D Space Illustration')
plt.savefig('LinearRegression3D.png')