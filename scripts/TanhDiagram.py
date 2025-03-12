# Tanh函数图像绘制指南

## 使用Matplotlib绘制tanh函数及其导数图像
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-5, 5, 500)
tanh = np.tanh(x)
derivative = 1 - tanh**2  # 导数公式

# 创建画布
plt.figure(figsize=(10, 6), dpi=100)

# 绘制主函数
plt.plot(x, tanh, 
         color='#E63829', 
         linewidth=3, 
         label='tanh(x)')

# 绘制导数曲线
plt.plot(x, derivative, 
         color='#2E5C87', 
         linestyle='--', 
         linewidth=3,
         label='Derivative')

# 坐标轴设置
plt.axhline(0, color='gray', linewidth=1)  # x轴
plt.axvline(0, color='gray', linewidth=1)  # y轴
plt.axis([-5, 5, -1.2, 1.2])  # 坐标范围

# 标签和图例
plt.title('Tanh Activation Function and its Derivative', pad=20, fontsize=14)
plt.xlabel('Input', fontsize=12)
plt.ylabel('Output', fontsize=12)
plt.legend(loc='upper left', fontsize=12)

# 辅助网格线
plt.grid(True, 
        linestyle=':', 
        alpha=0.6, 
        color=(0.8,0.8,0.8))

# 保存和显示
plt.savefig('TanhDiagram.png', dpi=300, transparent=True)