#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ===== 参数 =====
CONTACT_THRESHOLD = 20.0
DT_RESAMPLE = 0.01            # 10 ms 分辨率
PRE_SEC = 2.0                 # 切换前窗口
POST_SEC = 2.0                # 切换后窗口
OUT_DIR = "/home/song/cpg_jump/deploy/figures"
GAIT_NAME = {
    0: "Trot",
    1: "Walk",
    2: "Bound",
    3: "Pronk"
}
# ===== 工具函数（保持你原有风格）=====
def _intervals_from_bool_series(time, bool_series):
    time = np.asarray(time)
    b = np.asarray(bool_series).astype(bool)
    if time.shape[0] == 0:
        return []
    intervals = []
    i = 0
    n = len(b)
    while i < n:
        if b[i]:
            start = time[i]
            j = i + 1
            while j < n and b[j]:
                j += 1
            end = time[j] if j < n else time[-1]
            duration = end - start
            if duration <= 0:
                duration = 1e-3
            intervals.append((start, duration))
            i = j
        else:
            i += 1
    return intervals

def _intervals_from_two_bool_series_overlap(time, a_series, b_series):
    both = np.logical_and(np.asarray(a_series).astype(bool), np.asarray(b_series).astype(bool))
    return _intervals_from_bool_series(time, both)

def load_and_make_contact(df, contact_threshold=CONTACT_THRESHOLD):
    """
    读取并标准化列，计算接触状态；若包含 GaitID 列则一并保留。
    """
    cols = [str(c).strip() for c in df.columns.tolist()]
    df.columns = cols

    # 支持两种输入：带表头（Time/FL_Fz...）或 raw 5 列
    expected_fz = ['FL_Fz', 'FR_Fz', 'RL_Fz', 'RR_Fz']
    if 'Time (s)' in cols and all(c in cols for c in expected_fz):
        pass
    else:
        if df.shape[1] >= 5:
            new_names = ['Time (s)', 'FL_Fz', 'FR_Fz', 'RL_Fz', 'RR_Fz'] + \
                        [f'col_{i}' for i in range(5, df.shape[1])]
            df.columns = new_names[:df.shape[1]]
        else:
            raise ValueError("CSV lacks required columns. Need >=5: Time + 4 Fz.")

    # 转数值
    df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
    for c in expected_fz:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # 可选：GaitID
    has_gait = 'GaitID' in df.columns
    if has_gait:
        df['GaitID'] = pd.to_numeric(df['GaitID'], errors='coerce')

    # 清 NaN 时间戳
    df = df.dropna(subset=['Time (s)']).reset_index(drop=True)

    # 接触：注意你之前数据是负为压地，所以用 < -threshold
    df['FL_Contact_Status'] = (df['FL_Fz'] < -contact_threshold).astype(int)
    df['FR_Contact_Status'] = (df['FR_Fz'] < -contact_threshold).astype(int)
    df['RL_Contact_Status'] = (df['RL_Fz'] < -contact_threshold).astype(int)
    df['RR_Contact_Status'] = (df['RR_Fz'] < -contact_threshold).astype(int)

    return df

def resample_to_fixed_dt(df, dt=DT_RESAMPLE):
    """
    统一采样到固定 dt。接触状态用最近邻（>0.5→1），GaitID 也用最近邻（round）。
    """
    t_min = df['Time (s)'].min()
    t_max = df['Time (s)'].max()
    time_fixed = np.arange(t_min, t_max + dt/2, dt)
    out = pd.DataFrame({'Time (s)': time_fixed})

    # 接触状态最近邻
    for leg in ['FL', 'FR', 'RL', 'RR']:
        col = f'{leg}_Contact_Status'
        out[col] = np.interp(time_fixed, df['Time (s)'], df[col], left=0, right=0)
        out[col] = (out[col] > 0.5).astype(int)

    # 若有 GaitID，则最近邻 + round 成 int
    if 'GaitID' in df.columns:
        gait_interp = np.interp(time_fixed, df['Time (s)'], df['GaitID'].fillna(method='ffill').fillna(method='bfill'), left=np.nan, right=np.nan)
        # 边界 NaN 用最邻近填充
        if np.isnan(gait_interp[0]):
            first_valid = np.flatnonzero(~np.isnan(gait_interp))
            if first_valid.size > 0:
                gait_interp[:first_valid[0]] = gait_interp[first_valid[0]]
        if np.isnan(gait_interp[-1]):
            last_valid = np.flatnonzero(~np.isnan(gait_interp))
            if last_valid.size > 0:
                gait_interp[last_valid[-1]:] = gait_interp[last_valid[-1]]
        out['GaitID'] = np.rint(gait_interp).astype(int)

    return out

def find_gait_transitions(df_resampled):
    """
    从重采样后的数据里找 GaitID 的变化点（上升/下降沿）。
    返回列表 [(t_change, gait_prev, gait_new), ...]
    """
    if 'GaitID' not in df_resampled.columns:
        return []

    g = df_resampled['GaitID'].to_numpy()
    t = df_resampled['Time (s)'].to_numpy()
    if len(g) < 2:
        return []

    changes = np.where(g[1:] != g[:-1])[0] + 1  # 变化发生在 i (与 i-1 不同)
    transitions = []
    for idx in changes:
        gait_prev = int(g[idx-1])
        gait_new = int(g[idx])
        t_change = float(t[idx])
        transitions.append((t_change, gait_prev, gait_new))
    return transitions

def plot_contact_window(df_resampled, t0, t1, save_path=None, title=None, transitions_in_window=None):
    """画 [t0, t1] 的接触条形图，并在 gait 变化时刻画垂直竖线标注。"""
    df_win = df_resampled[(df_resampled['Time (s)'] >= t0) & (df_resampled['Time (s)'] <= t1)].copy()
    if df_win.empty:
        print(f"[WARN] empty window [{t0:.3f}, {t1:.3f}]")
        return

    time = df_win['Time (s)'].to_numpy()
    legs = ['FL', 'FR', 'RL', 'RR']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=120)
    y_height = 0.8
    y_gap = 0.2
    y_positions = {
        'FL': 3 * (y_height + y_gap),
        'FR': 2 * (y_height + y_gap),
        'RL': 1 * (y_height + y_gap),
        'RR': 0 * (y_height + y_gap),
    }

    # 绘制接触条形图
    for leg, color in zip(legs, colors):
        col = f'{leg}_Contact_Status'
        intervals = _intervals_from_bool_series(time, df_win[col].to_numpy())
        ax.broken_barh(intervals, (y_positions[leg], y_height),
                       facecolors=color, edgecolors='k', linewidth=0.5, alpha=0.9, label=leg)

    # 坐标轴设置
    ax.set_xlim(t0, t1)
    ax.set_ylim(-0.2, 4 * (y_height + y_gap))
    ax.set_yticks([y_positions[l] + y_height / 2 for l in legs])
    ax.set_yticklabels(legs)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xlabel('Time (s)', fontsize=14)
    if title:
        ax.set_title(title, fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=1)

    # 图例
    from matplotlib.patches import Patch
    patches = [Patch(facecolor=colors[i], edgecolor='k', label=legs[i]) for i in range(len(legs))]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1.0), fontsize=12, loc='upper left')

    # =========== ⭐ 关键部分：绘制 gait_id 变化时刻竖线 ===========
    if transitions_in_window:
        y_top_base = 4 * (y_height + y_gap)
        y_text = y_top_base + 0.15
        ax.set_ylim(-0.2, y_text + 0.35)

        # 再取最终的 y 轴范围，用数据坐标画线
        ymin, ymax = ax.get_ylim()

       # ========== ⭐ Gait 转换标注：背景 + 加粗竖线 + 文字 ==========
        PRE_HL  = 2.0   # 切换前高亮时长（秒）
        POST_HL = 2.0   # 切换后高亮时长（秒）

        for (t_change, g_prev, g_new) in transitions_in_window or []:
            if not (t0 <= t_change <= t1):
                continue

            g_prev_name = GAIT_NAME.get(g_prev, str(g_prev))
            g_new_name  = GAIT_NAME.get(g_new, str(g_new))

            # ✅ 高亮背景区域（前后）
            ax.axvspan(t_change - PRE_HL, t_change, ymin=0.0, ymax=1.0,
                    facecolor='#7fa6ff', alpha=0.15, zorder=2)  # 比原先略深、稍更饱和
            ax.axvspan(t_change, t_change + POST_HL, ymin=0.0, ymax=1.0,
                    facecolor='#ff9ca3', alpha=0.15, zorder=2)

            # ✅ 加粗竖线（红色）
            ymin, ymax = ax.get_ylim()
            ax.plot([t_change, t_change], [ymin, ymax],
                    color='red', linestyle='--', linewidth=2.0, alpha=0.95,
                    zorder=10, solid_capstyle='butt', clip_on=False)

            # ✅ 文字标注
            ax.text(
                t_change, ymax - 0.1,   # 顶部留点空白
                f"{g_prev_name} → {g_new_name}",
                color='black', fontsize=13, fontweight='bold',
                ha='center', va='bottom',
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=2),
                zorder=11
            )

    print(f"[MARK] {g_prev_name}->{g_new_name} at {t_change:.3f}s")



    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] saved: {save_path}")
    plt.show()
    plt.close(fig)



def main():
    # 你的混合步态 CSV（含 GaitID）
    csv_to_plot = "/home/song/cpg_jump/deploy/csv/gait_contact_forces_30s_mixed_gaits.csv"
    if not os.path.exists(csv_to_plot):
        print("CSV not found:", csv_to_plot); return

    # 读取
    df_raw = pd.read_csv(csv_to_plot, header=0)
    df = load_and_make_contact(df_raw, contact_threshold=CONTACT_THRESHOLD)

    # 有 GaitID 才能做切换分析
    if 'GaitID' not in df.columns:
        print("[ERR] CSV has no GaitID column. Abort."); return

    # 重采样（含 GaitID 最近邻）
    df_rs = resample_to_fixed_dt(df, dt=DT_RESAMPLE)
    df_rs = df_rs.sort_values('Time (s)').reset_index(drop=True)

    # 找切换
    transitions = find_gait_transitions(df_rs)
    if not transitions:
        print("[INFO] no gait transition found.")
        return

    # 对每次切换，画 [t-1, t+1]
    os.makedirs(OUT_DIR, exist_ok=True)
    for (t_change, g_prev, g_new) in transitions:
        t0 = max(df_rs['Time (s)'].min(), t_change - PRE_SEC)
        t1 = min(df_rs['Time (s)'].max(), t_change + POST_SEC)
        # title = f"Gait transition {g_prev} → {g_new} at t={t_change:.2f}s"
        g_prev_name = GAIT_NAME.get(g_prev, str(g_prev))
        g_new_name  = GAIT_NAME.get(g_new, str(g_new))
        save_path = os.path.join(OUT_DIR, f"contact_transition_{g_prev_name}_to_{g_new_name}_{t_change:.2f}s.png")
        plot_contact_window(df_rs, t0, t1, save_path=save_path,transitions_in_window=[(t_change, g_prev, g_new)])

if __name__ == "__main__":
    main()
