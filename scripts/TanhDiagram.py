# Tanh函数图像绘制指南

# 使用Matplotlib绘制tanh函数及其导数图像
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-6, 6, 500)
tanh = np.tanh(x)
derivative = 1 - tanh**2  # 导数公式

# 创建画布
plt.figure(figsize=(12, 5), dpi=100)

plt.subplot(1, 2, 1)

# 辅助网格线
plt.grid(True,
         linestyle=':',
         alpha=0.6,
         color=(0.4, 0.4, 0.4))

# 加粗坐标轴
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

plt.xlim(-6, 6)
plt.ylim(-2, 5)

# 绘制主函数
plt.plot(x, tanh,
         color='#E63829',
         linewidth=3,
         label='tanh(x)')


# 标签和图例
plt.title('Tanh Activation', pad=20, fontsize=14)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.legend(loc='upper left', fontsize=12)


plt.subplot(1, 2, 2)

# 辅助网格线
plt.grid(True,
         linestyle=':',
         alpha=0.6,
         color=(0.4, 0.4, 0.4))

# 加粗坐标轴
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)

plt.xlim(-6, 6)
plt.ylim(-2, 5)

# 绘制导数曲线
plt.plot(x, derivative,
         color='#2E5C87',
         linestyle='--',
         linewidth=3,
         label='Derivative')

# 标签和图例
plt.title('Derivative of Tanh', pad=20, fontsize=14)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.legend(loc='upper left', fontsize=12)

# 保存和显示
plt.savefig('Activation/TanhDiagram.png')
