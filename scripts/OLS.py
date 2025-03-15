import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据点
np.random.seed(0)
# 生成一个10*1的列向量，从0到10步长为10
x = np.linspace(0, 10, 10).reshape(-1, 1)

print(x)

# y = 2x + 1 + 噪声
y = 1.5 * x + 1 + np.random.randn(10, 1) * 1  # y = 2x + 1 加上一些噪声

y_hat = 1.5 * x + 1  # 真实的y值

y_fake = 2 * x - 1

# 绘图
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

axs[0].scatter(x, y, color='blue', label='Data points')  # 绘制原始数据点
axs[0].plot(x, y_hat, color='green', label='True line')  # 绘制真实的直线
# 绘制以原始数据点和真实点之间距离为边长的正方形
dis = np.abs(y - y_hat)
for i in range(len(x)):
    residual_square_X = [x[i], x[i], x[i] + dis[i], x[i] + dis[i], x[i]]
    residual_square_Y = [y[i], y_hat[i], y_hat[i], y[i], y[i]]
    axs[0].fill(residual_square_X, residual_square_Y, color='red', alpha=0.3)

axs[0].set_xlim(0, 20)
axs[0].set_ylim(0, 20)
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('OLS: Small Error Fit Line')

axs[1].scatter(x, y, color='blue', label='Data points')  # 绘制原始数据点
axs[1].plot(x, y_fake, color='green', label='True line')  # 绘制真实的直线
# 绘制以原始数据点和真实点之间距离为边长的正方形
dis = np.abs(y - y_fake)
for i in range(len(x)):
    if x[i] + dis[i]/2 > 10:
        dis[i] = -dis[i]
    residual_square_X = [x[i], x[i], x[i] + dis[i], x[i] + dis[i], x[i]]
    residual_square_Y = [y[i], y_fake[i], y_fake[i], y[i], y[i]]
    axs[1].fill(residual_square_X, residual_square_Y, color='red', alpha=0.3)

axs[1].set_xlim(0, 20)
axs[1].set_ylim(0, 20)
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_title('OLS: Large Error Fit Line')

plt.savefig('OLS.png')
