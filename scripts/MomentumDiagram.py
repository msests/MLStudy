import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# 定义目标函数和梯度
def f(x, y):
    return x**2 + 10*y**2  # 明显椭圆形的函数，用于展示方向差异

def grad_f(x, y):
    return np.array([2*x, 20*y])

# 传统SGD优化算法
def sgd(start_point, learning_rate=0.1, num_steps=50):
    x, y = start_point
    trajectory = [(x, y)]
    
    for _ in range(num_steps):
        g = grad_f(x, y)
        x -= learning_rate * g[0]
        y -= learning_rate * g[1]
        trajectory.append((x, y))
    return np.array(trajectory)

# Momentum优化算法
def momentum(start_point, learning_rate=0.1, beta=0.9, num_steps=50):
    x, y = start_point
    trajectory = [(x, y)]
    v = np.zeros(2)
    
    for _ in range(num_steps):
        g = grad_f(x, y)
        v = beta * v + learning_rate * g
        x -= v[0]
        y -= v[1]
        trajectory.append((x, y))
    return np.array(trajectory)

# 生成演示数据
start_point = (-4, 4.5)
trajectory_sgd = sgd(start_point, learning_rate=0.1, num_steps=30)
trajectory_momentum = momentum(start_point, learning_rate=0.1, beta=0.9, num_steps=15)

# 创建网格用于绘制等高线
x = np.linspace(-5, 5, 400)
y = np.linspace(-5, 5, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 创建可视化图形
plt.figure(figsize=(10, 6))
ax = plt.axes()

# 绘制等高线图
levels = np.logspace(0, 3, 20)
ax.contour(X, Y, Z, levels=levels, norm=LogNorm(), cmap='coolwarm', alpha=0.5)

# 绘制优化轨迹
ax.plot(trajectory_sgd[:, 0], trajectory_sgd[:, 1], 'o-', markersize=4, 
        color='blue', linewidth=2, markeredgecolor='red', markeredgewidth=0.5)
ax.plot(trajectory_momentum[:, 0], trajectory_momentum[:, 1], 'o-', markersize=4,
        color='orange', linewidth=2, markeredgecolor='red', markeredgewidth=0.5)

# 添加箭头显示优化方向
for i in range(1, len(trajectory_sgd)):
    ax.annotate('', xy=trajectory_sgd[i], xytext=trajectory_sgd[i-1],
                arrowprops=dict(arrowstyle='->', color='black', lw=1, alpha=0.7))

for i in range(1, len(trajectory_momentum)):
    ax.annotate('', xy=trajectory_momentum[i], xytext=trajectory_momentum[i-1],
                arrowprops=dict(arrowstyle='->', color='black', lw=1, alpha=0.7))

# 标注起点和终点
ax.scatter(*start_point, c='green', s=80, label='Start', zorder=5)
ax.scatter(0, 0, c='red', s=80, label='Minimum', zorder=5)
# 绿色为普通梯度下降，蓝色为RMSProp优化
ax.plot([], [], 'o-', markersize=4, color='blue', linewidth=2, label='SGD')
ax.plot([], [], 'o-', markersize=4, color='orange', linewidth=2, label='Momentum')

# 添加图例和标注
ax.set_title('Momentum Optimization Path', fontsize=14)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.text(-5, -5, f'Learning rate: 0.15\nBeta: 0.9\n', 
        bbox=dict(facecolor='white', alpha=0.8))
ax.legend()

# 设置坐标轴范围
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# 显示网格
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()