import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(121, projection='3d')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-16, 5)

ax.quiver(-5, 0, 0, 10, 0, 0, color='red', length=1, arrow_length_ratio=0.1)
ax.quiver(0, -5, 0, 0, 10, 0, color='green', length=1, arrow_length_ratio=0.1)
ax.quiver(0, 0, -16, 0, 0, 21, color='blue', length=1, arrow_length_ratio=0.1)

ax.text(5, 0, -1, 'x', color='black', fontsize=12)
ax.text(0, 5, -1, 'y', color='black', fontsize=12)
ax.text(-2, -2, 2, 'z', color='black', fontsize=12)

ax.view_init(30, 30)

x = np.linspace(-5, 1, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

Z = -X**2 - Y**2 + 2
ax.plot_surface(X, Y, Z, alpha=0.3, color='gray', cmap='viridis',  linewidth=0)

# 创建x=2的切面
y_sec_plane = np.linspace(-5, 5, 100)
z_sec_plane = np.linspace(-25, 5, 100)
Y, Z = np.meshgrid(y_sec_plane, z_sec_plane)
X = 2*np.ones_like(Y)
ax.plot_surface(X, Y, Z, alpha=0.2, color='red', linewidth=0)
ax.text(2, -4, 0, 'x=1', color='red', fontsize=12)

x = np.linspace(1, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)

Z = -X**2 - Y**2 + 2
ax.plot_surface(X, Y, Z, alpha=0.8, cmap='viridis', color='gray', linewidth=0)

plt.gca().set_box_aspect([1, 1, 1])

ax = fig.add_subplot(122, projection='3d')

ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-16, 5)

ax.quiver(-5, 0, 0, 10, 0, 0, color='red', length=1, arrow_length_ratio=0.1)
ax.quiver(0, -5, 0, 0, 10, 0, color='green', length=1, arrow_length_ratio=0.1)
ax.quiver(0, 0, -16, 0, 0, 21, color='blue', length=1, arrow_length_ratio=0.1)

ax.text(5, 0, -1, 'x', color='black', fontsize=12)
ax.text(0, 5, -1, 'y', color='black', fontsize=12)
ax.text(-2, -2, 2, 'z', color='black', fontsize=12)

# 创建x=2的切面
x_sec = 1
y_sec = np.linspace(-5, 5, 100)
z_sec = -x_sec**2 - y_sec**2 + 2
ax.plot(x_sec*np.ones_like(y_sec), y_sec, z_sec, color='red', linestyle='--')

y_sec_plane = np.linspace(-5, 5, 100)
z_sec_plane = np.linspace(-25, 5, 100)
Y, Z = np.meshgrid(y_sec_plane, z_sec_plane)
X = 2*np.ones_like(Y)
ax.plot_surface(X, Y, Z, alpha=0.2, color='red')
ax.text(2, -4, 0, 'x=1', color='red', fontsize=12)

y_sec_line = np.linspace(-5, 5, 100)
z_sec_line = -4 * y_sec_line + 4
ax.plot(x_sec*np.ones_like(y_sec_line), y_sec_line, z_sec_line, color='red')

ax.text(1, 1, 0, r'$\frac{\partial z}{\partial y}$',
        color='black', fontsize=18)

ax.view_init(30, 30)

plt.title('Partial Derivative')

plt.savefig('Calculus/PartialDerivative.png')

plt.show()
