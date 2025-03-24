import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.set_zlim(-1, 5)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

x_conv_input = np.linspace(-5, 5, 11)
y_conv_input = np.linspace(-5, 5, 11)
X, Y = np.meshgrid(x_conv_input, y_conv_input)
Z = 0 * np.ones_like(X)

ax.plot_surface(X, Y, Z, alpha=1, color='blue',
                edgecolor='lightgray', zorder=0)

x_conv_kernel = np.linspace(-4, 0, 5)
y_conv_kernel = np.linspace(-4, 0, 5)
X, Y = np.meshgrid(x_conv_kernel, y_conv_kernel)
Z = 1 * np.ones_like(X)

ax.plot_surface(X, Y, Z, alpha=1, color='red', edgecolor='lightgray', zorder=1)

# for i in range(36):
#     ax.view_init(elev=40, azim=i * 10)
#     plt.title(f'Stacked 3x3 Convolution Layers, Azimuth={i*10}')
#     plt.draw()
#     plt.pause(1)
ax.view_init(elev=20, azim=180)
ax.legend()

plt.show()
