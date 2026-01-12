#!/usr/bin/env python3
"""
Plot the first velocity data (vel_0 vs target_vel_0) with automatic time alignment.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# ---------- CSV 文件路径 ----------
gait="WALK"  # 修改为你的 gait 名称
CSV_VEL_FILE = f"/home/song/cpg_jump/deploy/csv/velocities_10s_{gait}.csv"
CSV_TARGET_VEL_FILE = f"/home/song/cpg_jump/deploy/csv/target_velocities_10s_{gait}.csv"

# ---------- 检查文件 ----------
if not os.path.exists(CSV_VEL_FILE):
    raise FileNotFoundError(f"Velocity CSV not found: {CSV_VEL_FILE}")
if not os.path.exists(CSV_TARGET_VEL_FILE):
    raise FileNotFoundError(f"Target velocity CSV not found: {CSV_TARGET_VEL_FILE}")

# ---------- 读取数据 ----------
vel_df = pd.read_csv(CSV_VEL_FILE)
target_df = pd.read_csv(CSV_TARGET_VEL_FILE)

# ---------- 提取时间与数据 ----------
time_actual = vel_df["Time (s)"].to_numpy()
actual_vel = vel_df["vel_0"].to_numpy()

time_target = target_df["Time (s)"].to_numpy()
target_vel_raw = target_df["target_vel_0"].to_numpy()
print(f"Actual vel data points: {len(actual_vel)}, Target vel data points: {len(target_vel_raw)}")
# ---------- 对齐时间：使用插值 ----------
# 将 target_vel 插值到 actual_vel 的时间轴上
target_vel_interp = np.interp(time_actual, time_target, target_vel_raw)

# ---------- 绘图 ----------
plt.figure(figsize=(10, 5))
plt.plot(time_actual, actual_vel, label="Actual Velocity", linewidth=3)
plt.plot(time_actual, target_vel_interp, '--', label="Target Velocity", linewidth=3)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Time (s)", fontsize=15)
plt.ylabel("Velocity (m/s)", fontsize=15)
# plt.title(f"Velocity tracking performance during the {gait} gait", fontsize=14)
plt.legend(fontsize=15)
plt.grid(True, linestyle="--", alpha=0.6)

# ---------- 保存图像 ----------
save_path = f"/home/song/cpg_jump/deploy/figures/vel_{gait}.png"
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()

print(f"✅ Saved aligned velocity comparison plot to: {save_path}")
