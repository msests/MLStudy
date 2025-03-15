import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time

# 创建3D画布
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_zlim(2, 5)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# 定义x, y的范围
x_base = np.linspace(-5, 5, 10)
y_base = np.linspace(-5, 5, 10)

# 创建网格数据
X_base, Y_base = np.meshgrid(x_base, y_base)

# 定义Z轴，即平面的方程。
# 这里我们定义为Z=2，创建一个平行于xy平面的平面
Z_base = 2 * np.ones_like(X_base)

x_conv_5x5 = np.linspace(-5, 0, 6)
y_conv_5x5 = np.linspace(-5, 0, 6)

print(x_conv_5x5)

X_conv_5x5, Y_conv_5x5 = np.meshgrid(x_conv_5x5, y_conv_5x5)

Z_conv_5x5 = 2.5 * np.ones_like(X_conv_5x5)

x_conv_feat = np.linspace(-3, -2, 2)
y_conv_feat = np.linspace(-3, -2, 2)

X_conv_feat, Y_conv_feat = np.meshgrid(x_conv_feat, y_conv_feat)

Z_conv_feat = 3 * np.ones_like(X_conv_feat)

# 修改绘图顺序：从低到高绘制
# 1. 先绘制base表面 (Z=2)
ax.plot_surface(X_base, Y_base, Z_base, alpha=1,
                color='blue', edgecolor='lightgray')

print(Z_base)

# 2. 然后绘制第一个conv表面 (Z=2.5)
ax.plot_surface(X_conv_5x5, Y_conv_5x5, Z_conv_5x5,
                alpha=1, color='red', edgecolor='lightgray')

print(Z_conv_5x5)

# 3. 最后绘制第二个conv表面 (Z=3)
ax.plot_surface(X_conv_feat, Y_conv_feat,
                Z_conv_feat, alpha=1, color='orange', edgecolor='lightgray')

X_feat_map = np.array([-5, -3])
Y_feat_map = np.array([-5, -3])
Z_feat_map = np.array([2.5, 3])
ax.plot(X_feat_map, Y_feat_map, Z_feat_map, alpha=1, color='green')

X_feat_map = np.array([-5, -3])
Y_feat_map = np.array([0, -2])
ax.plot(X_feat_map, Y_feat_map, Z_feat_map, alpha=1, color='green')

X_feat_map = np.array([0, -2])
Y_feat_map = np.array([0, -2])
ax.plot(X_feat_map, Y_feat_map, Z_feat_map, alpha=1, color='green')

X_feat_map = np.array([0, -2])
Y_feat_map = np.array([-5, -3])
ax.plot(X_feat_map, Y_feat_map, Z_feat_map, alpha=1, color='green')

# 调整视角

# for i in range(36):
#     ax.view_init(elev=40, azim=i * 10)
#     plt.title(f'Stacked 3x3 Convolution Layers, Azimuth={i*10}')
#     plt.draw()
#     plt.pause(1)
ax.view_init(elev=30, azim=180)

ax.axis('off')

plt.title('Stacked 3x3 Convolution Layers')
# plt.tight_layout()
plt.show()
