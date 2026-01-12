#!/usr/bin/env python3
"""
Plot height vs. time, with shaded regions indicating jump_signal == 1 (Jump phases).
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------- CSV 文件路径 ----------
gait = "WALK"  # 修改为你的 gait 名称
CSV_FILE = f"/home/song/cpg_jump/deploy/csv/height_jump_5s_{gait}.csv"

# ---------- 检查文件 ----------
if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")

# ---------- 读取数据 ----------
df = pd.read_csv(CSV_FILE)

# ---------- 提取时间、高度与跳跃信号 ----------
time = df["Time (s)"].to_numpy()
height = df["height"].to_numpy()
jump_signal = df["jump_signal"].to_numpy().astype(int)

# ---------- 寻找 jump_signal == 1 的区间 ----------
def get_intervals(time, signal):
    """返回 (start, end) 区间列表，表示 signal==1 的连续段"""
    intervals = []
    in_jump = False
    start_t = None
    for t, s in zip(time, signal):
        if s == 1 and not in_jump:
            start_t = t
            in_jump = True
        elif s == 0 and in_jump:
            end_t = t
            intervals.append((start_t, end_t))
            in_jump = False
    if in_jump and start_t is not None:
        intervals.append((start_t, time[-1]))
    return intervals

jump_intervals = get_intervals(time, jump_signal)

# ---------- 绘图 ----------
plt.figure(figsize=(10, 5))
plt.plot(time, height, color="#1f77b4", linewidth=3, label="Base Height")

# 在 jump_signal == 1 的区间加背景阴影并添加文字说明
for (start, end) in jump_intervals:
    plt.axvspan(start, end, color="#FFD580", alpha=0.35)
    mid_t = (start + end) / 2
    plt.text(mid_t, np.max(height) * 0.95, "jump_signal = 1",
             ha='center', va='top', fontsize=12, color='black', alpha=0.8,
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))

# ---------- 图形美化 ----------
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Time (s)", fontsize=15)
plt.ylabel("Height (m)", fontsize=15)
plt.legend(fontsize=15)
plt.grid(True, linestyle="--", alpha=0.6)

# ---------- 保存图像 ----------
save_path = f"/home/song/cpg_jump/deploy/figures/height_with_jump_shading_{gait}.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Saved height plot with jump shading to: {save_path}")
