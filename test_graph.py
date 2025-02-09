import matplotlib.pyplot as plt
import numpy as np

# 示例数据
x = np.linspace(0, 10, 10)
y = np.sin(x)

# 定义一组标记形状
markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

plt.figure(figsize=(8, 5))

# 先绘制连接线
plt.plot(x, y, linestyle='-', color='gray', label='连接线')

# 循环绘制每个数据点，使用不同的标记
for i in range(len(x)):
    plt.plot(x[i], y[i], marker=markers[i % len(markers)], markersize=8,
             label='数据点' if i == 0 else None)  # 仅给第一个点添加图例，避免重复

plt.xlabel("x")
plt.ylabel("sin(x)")
plt.title("折线图示例：每个点使用不同的标记形状")

plt.grid(True, linestyle='--', alpha=0.6)  # 增加网格线
plt.legend()
plt.show()
