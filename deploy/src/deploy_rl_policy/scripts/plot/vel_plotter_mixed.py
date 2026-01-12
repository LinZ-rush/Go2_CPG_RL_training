#!/usr/bin/env python3
"""
Plot vel_0 vs time with target_vel_0 (interpolated),
mark gait transitions with bold vertical lines, and
shade each gait interval with a light background.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ========= 文件路径 =========
CSV_FILE  = "/home/song/cpg_jump/deploy/csv/vel_target_gait_30s_mixed.csv"
save_path = "/home/song/cpg_jump/deploy/figures/vel_with_gait_shading_mixed.png"

# ========= 步态配置 =========
GAIT_NAME = {0: "Trot", 1: "Walk", 2: "Bound", 3: "Pronk"}
# 背景色：浅而有对比度，不与折线冲突
GAIT_BG   = {
    0: "#cfe3ff",   # Trot  浅蓝
    1: "#ffe0b3",   # Walk  浅橙
    2: "#d1f0d1",   # Bound 浅绿
    3: "#ffd4dc",   # Pronk 浅粉
}
BG_ALPHA = 0.22          # 背景透明度
LINE_COLOR = "red"       # 竖线颜色
LINE_WIDTH = 1.0         # 竖线粗细
TEXT_COLOR = "#b30000"   # 标注文字颜色

def main():
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(CSV_FILE)

    # 读取
    df = pd.read_csv(CSV_FILE)

    # 需要的列转数值
    for col in ["Time (s)", "vel_0", "target_vel_0", "GaitID"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 丢掉没有时间戳的
    df = df.dropna(subset=["Time (s)"]).sort_values("Time (s)").reset_index(drop=True)

    # 时间轴
    t = df["Time (s)"].to_numpy()

    # 实际速度 vel_0
    if "vel_0" not in df.columns:
        raise ValueError("vel_0 column is missing.")
    v = df["vel_0"].to_numpy()

    # 目标速度 target_vel_0（可能有 NaN）
    tv = None
    if "target_vel_0" in df.columns:
        tv_raw = df["target_vel_0"].to_numpy()
        valid = ~np.isnan(tv_raw)
        if valid.any():
            # 插值到 t 轴
            tv = np.interp(t, t[valid], tv_raw[valid])
        else:
            tv = None

    # GaitID：-1 或 NaN 视为未知，不用于切换检测
    gid_raw = df["GaitID"].to_numpy() if "GaitID" in df.columns else None
    if gid_raw is None:
        print("[WARN] No GaitID column; will plot velocities without shading/markers.")
        gid_clean = None
    else:
        gid_clean = gid_raw.astype(float)
        gid_clean[gid_clean < 0] = np.nan  # -1 视为 NaN

    # ====== 计算切换时刻与区间 ======
    transitions = []   # [(t_change, prev_id, new_id), ...]
    intervals   = []   # [(t_start, t_end, gid), ...]

    if gid_clean is not None:
        # 找到有效区间（连续有效 GaitID 的段）
        valid = ~np.isnan(gid_clean)
        if valid.any():
            idx = np.where(valid)[0]
            start_idx = idx[0]
            prev_id = int(gid_clean[start_idx])

            for i in idx[1:]:
                cur_id = int(gid_clean[i])
                if cur_id != prev_id:
                    # 记录上一个区间
                    intervals.append((t[start_idx], t[i], prev_id))
                    # 记录切换
                    transitions.append((t[i], prev_id, cur_id))
                    # 新区间开始
                    start_idx = i
                    prev_id = cur_id
            # 最后一个区间
            intervals.append((t[start_idx], t[valid][-1], prev_id))

    # ====== 绘图 ======
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # 先画折线（确保自动确定 ylims）
    ax.plot(t, v, label="Actual Velocity", linewidth=2.0, zorder=5)
    if tv is not None:
        ax.plot(t, tv, "--", label="Target Velocity", linewidth=2.0, zorder=5)

    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Velocity (m/s)", fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)

    # 现在拿到 ylims，用于竖线贯穿全图
    ymin, ymax = ax.get_ylim()

    # 步态区间背景
    if intervals:
        for (ts, te, gid) in intervals:
            if te <= ts:
                continue
            color = GAIT_BG.get(gid, "#eeeeee")
            ax.axvspan(ts, te, ymin=0.0, ymax=1.0, facecolor=color, alpha=BG_ALPHA, zorder=2)

    # 竖线与文字标注
    if transitions:
        for (tc, prev_id, new_id) in transitions:
            # 加粗竖线
            ax.plot([tc, tc], [ymin, ymax],
                    color=LINE_COLOR, linewidth=LINE_WIDTH, linestyle='--', alpha=0.95,
                    zorder=10, solid_capstyle='butt', clip_on=False)
            # 顶部文字
            prev_name = GAIT_NAME.get(prev_id, str(prev_id))
            new_name  = GAIT_NAME.get(new_id,  str(new_id))
            ax.text(tc, ymax, f"{prev_name} → {new_name}",
                    color="black", fontsize=12, fontweight="bold",
                    ha="center", va="bottom",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2),
                    zorder=11)

    # 背景图例（可选）
    legend_patches = []
    seen = set()
    for (_, _, gid) in intervals:
        if gid in seen:
            continue
        seen.add(gid)
        legend_patches.append(Patch(facecolor=GAIT_BG.get(gid, "#eeeeee"),
                                    edgecolor='none',
                                    alpha=BG_ALPHA,
                                    label=GAIT_NAME.get(gid, f"Gait {gid}")))
    if legend_patches:
        pass
        # leg1 = ax.legend(handles=legend_patches, title="Gait intervals", loc="upper left", fontsize=10)
        # ax.add_artist(leg1)
    ax.legend(loc="lower right", fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved to: {save_path}")

if __name__ == "__main__":
    main()
