#!/usr/bin/env python3
"""
Plot Value vs Step with TensorBoard-style smoothing and aesthetics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ====== CSV 文件路径 ======
csv_file = "/home/song/cpg_jump/deploy/csv/Sep13_16-02-26_high_jump_vel.csv"
save_path = "/home/song/cpg_jump/deploy/figures/jump_fraction_over_training.png"

# ====== 读取数据 ======
df = pd.read_csv(csv_file)

# 有些 tensorboard 导出的 csv 可能不是严格按 Step 排序，这里保证一下
df = df.sort_values("Step").reset_index(drop=True)

# ====== EMA 平滑 ======
def smooth(y, weight=0.93):
    """
    Exponential moving average smoothing.
    weight 越接近 1 越平滑（TensorBoard风格 ~0.9-0.95）。
    """
    smoothed = []
    last = y[0]
    for val in y:
        last = last * weight + (1 - weight) * val
        smoothed.append(last)
    return np.array(smoothed)

# ====== 数据提取 & 平滑 ======
x = df["Step"].to_numpy()
y = df["Value"].to_numpy()
y_smooth = smooth(y)

# ====== 绘图（TensorBoard风格） ======
plt.figure(figsize=(7, 5))

# 原始曲线（淡）
plt.plot(
    x, y,
    color="#1f77b4",
    alpha=0.3,
    linewidth=1.5,
    label="Raw"
)

# 平滑曲线（主曲线）
plt.plot(
    x, y_smooth,
    color="#1f77b4",
    linewidth=2.5,
    label="Smoothed"
)

# 可选：给一点淡淡的填充区域来提升可读性（视觉类似 TB 的不确定带）
plt.fill_between(
    x,
    y_smooth,
    y,
    color="#1f77b4",
    alpha=0.08,
    linewidth=0
)

# 轴标签和标题
plt.xlabel("Iteration", fontsize=14)
plt.ylabel("Jump fraction", fontsize=14)
plt.title("Jump fraction over training iterations", fontsize=15)

# 样式
plt.legend(fontsize=12, frameon=False)
plt.grid(True, linestyle="--", alpha=0.3)

# 去掉顶部和右侧边框
ax = plt.gca()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 紧凑布局 + 保存
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Saved TensorBoard-style plot to: {save_path}")
