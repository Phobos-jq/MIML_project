import matplotlib.pyplot as plt
import numpy as np

# 模拟数据
x = np.linspace(0.2, 0.8, 10)  # 横坐标为训练数据比例
y_data = [np.tanh(10 * (x - 0.4) + i) * 50 + 50 for i in range(9)]  # 示例数据生成
titles = [
    "Full batch Adam",
    "Minibatch Adam",
    "Full batch Adam, update noise",
    "Dropout 0.1, Adam",
    "AdamW, weight decay 1",
    "AdamW, weight decay to init",
    "Adam, 0.3x baseline LR",
    "Adam, 3x baseline LR",
    "Gaussian weight noise + Adam",
]

# 创建图形和子图
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True, sharey=True)
axes = axes.ravel()  # 将3x3网格展平成一维

# 绘制每个子图
for i, ax in enumerate(axes):
    y = y_data[i] + np.random.randn(10) * 5  # 加入随机噪声
    ax.plot(x, y, marker='o', label="Best validation accuracy")
    ax.set_title(titles[i])
    ax.set_ylim(0, 100)
    ax.set_xlabel("Training data fraction")
    ax.set_ylabel("Best validation accuracy")
    ax.grid(True)

# 调整子图间距
plt.tight_layout()
plt.show()
