import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 生成随机数据点
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 使用模型预测
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Data Points')  # 原始数据点
plt.plot(X_new, y_predict, "r-", label='Best Fit Line')  # 最佳拟合直线
plt.title('Linear Regression Illustration')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('LinearRegression2D.png', dpi=300, transparent=True)