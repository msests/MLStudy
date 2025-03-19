import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(12, 6))

x = np.linspace(-6, 6, 100)
y = x**2

plt.plot(x, y, label=r'$y = x^2$', color='black', linewidth=1)

x_tangent = np.array([3, 4])
y_tangent = x_tangent**2
plt.plot(x_tangent, y_tangent, label='Tangent line at x=3',
         color='red', linewidth=1)

x_dy = np.array([4, 4])
y_dy = np.array([9, 16])
plt.plot(x_dy, y_dy, color='blue', linewidth=1)
plt.text(4.2, 12, r'$dy$', fontsize=12)

plt.title(r'$y = x^2$')
plt.xlabel('x')
plt.ylabel('y')

plt.xlim(-6, 6)
plt.ylim(0, 40)

plt.grid(True)

plt.savefig('Calculus/Differential.png')
