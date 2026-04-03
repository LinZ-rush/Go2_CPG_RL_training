#!/usr/bin/env python3
"""
步态接触相位图工具
用法：
    python gait_contact_plotter.py                        # 使用默认CSV路径
    python gait_contact_plotter.py --csv /path/to/file.csv --start 2.0 --duration 3.0
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ===== 参数 =====
CONTACT_THRESHOLD = 1.0   # N，IsaacGym仿真接触力阈值


def _intervals_from_bool_series(time, bool_series):
    time = np.asarray(time)
    b = np.asarray(bool_series).astype(bool)
    if len(time) == 0:
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
            duration = max(end - start, 1e-4)
            intervals.append((start, duration))
            i = j
        else:
            i += 1
    return intervals


def _phase_stats(time, contact_log):
    """
    从接触序列估算步态周期和对角腿相位差。
    contact_log: [N, 4]  FL=0, FR=1, RL=2, RR=3
    """
    def rising_edges(arr):
        idx = np.where(np.diff(arr.astype(int)) == 1)[0]
        return time[idx + 1] if len(idx) else np.array([])

    fl_t = rising_edges(contact_log[:, 0])
    fr_t = rising_edges(contact_log[:, 1])
    rl_t = rising_edges(contact_log[:, 2])
    rr_t = rising_edges(contact_log[:, 3])

    # 估算周期：取FL+RR触地事件合并排序后的平均间隔
    all_t = np.sort(np.concatenate([fl_t, rr_t])) if len(fl_t) and len(rr_t) else np.array([])
    T = float(np.mean(np.diff(all_t))) * 2 if len(all_t) >= 3 else None

    results = {}
    if T and T > 1e-4:
        results['period_ms'] = T * 1000

        def mean_phase(ref_times, tgt_times, period):
            offsets = []
            for t0 in ref_times:
                diffs = tgt_times - t0
                diffs = diffs[(diffs > 0) & (diffs < period)]
                if len(diffs):
                    offsets.append(diffs[0] / period)
            return float(np.mean(offsets)) if offsets else float('nan')

        results['FL_FR_phase'] = mean_phase(fl_t, fr_t, T)   # TROT理想=0.5
        results['FL_RL_phase'] = mean_phase(fl_t, rl_t, T)   # TROT理想=0.5
        results['FL_RR_phase'] = mean_phase(fl_t, rr_t, T)   # TROT理想=0.0 (同相)
    return results


def plot_gait(csv_path, start_time=0.0, duration=3.0, save_path=None):
    if not os.path.exists(csv_path):
        print(f'[Error] CSV not found: {csv_path}')
        return

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
    for leg in ['FL', 'FR', 'RL', 'RR']:
        df[f'{leg}_Fz'] = pd.to_numeric(df[f'{leg}_Fz'], errors='coerce')
    df = df.dropna().reset_index(drop=True)

    # 接触状态（IsaacGym中Fz>0表示接触）
    for leg in ['FL', 'FR', 'RL', 'RR']:
        df[f'{leg}_contact'] = (df[f'{leg}_Fz'] > CONTACT_THRESHOLD).astype(int)

    # 裁剪时间窗口
    t_end = start_time + duration
    win = df[(df['Time (s)'] >= start_time) & (df['Time (s)'] <= t_end)].copy()
    if win.empty:
        print(f'[Error] 时间窗口 [{start_time}, {t_end}] s 内无数据')
        print(f'  可用范围: {df["Time (s)"].min():.2f} ~ {df["Time (s)"].max():.2f} s')
        return

    time = win['Time (s)'].to_numpy()
    contact_log = np.stack([win[f'{leg}_contact'].to_numpy() for leg in ['FL', 'FR', 'RL', 'RR']], axis=1)

    # ===== 绘图 =====
    legs   = ['FL', 'FR', 'RL', 'RR']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    y_h, y_gap = 0.7, 0.3
    y_pos = {leg: (3 - i) * (y_h + y_gap) for i, leg in enumerate(legs)}

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={'height_ratios': [3, 1]})

    # --- 上图：接触甘特图 ---
    ax = axes[0]
    for leg, color in zip(legs, colors):
        idx = legs.index(leg)
        intervals = _intervals_from_bool_series(time, contact_log[:, idx])
        ax.broken_barh(intervals, (y_pos[leg], y_h),
                       facecolors=color, edgecolors='k', linewidth=0.4, alpha=0.85, label=leg)

    # 对角对同时接触高亮
    fl_rr = contact_log[:, 0] & contact_log[:, 3]
    fr_rl = contact_log[:, 1] & contact_log[:, 2]
    for s, d in _intervals_from_bool_series(time, fl_rr):
        ax.axvspan(s, s + d, alpha=0.12, color='blue', label='_nolegend_')
    for s, d in _intervals_from_bool_series(time, fr_rl):
        ax.axvspan(s, s + d, alpha=0.12, color='red', label='_nolegend_')

    ax.set_ylim(-0.2, 4 * (y_h + y_gap))
    ax.set_yticks([y_pos[l] + y_h / 2 for l in legs])
    ax.set_yticklabels(legs, fontsize=13)
    ax.set_xlabel('Time (s)', fontsize=13)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    patches = [mpatches.Patch(color=colors[i], label=legs[i]) for i in range(4)]
    ax.legend(handles=patches, loc='upper right', fontsize=11)

    # 相位统计写入标题
    stats = _phase_stats(time, contact_log)
    if stats:
        T_ms  = stats.get('period_ms', float('nan'))
        ph_fr = stats.get('FL_FR_phase', float('nan'))
        ph_rl = stats.get('FL_RL_phase', float('nan'))
        ph_rr = stats.get('FL_RR_phase', float('nan'))
        trot_ok = 0.4 < ph_fr < 0.6 and 0.4 < ph_rl < 0.6
        verdict = '✓ TROT' if trot_ok else '✗ 非标准TROT'
        title = (f'步态接触图  T={T_ms:.1f}ms  '
                 f'FL→FR={ph_fr:.3f}  FL→RL={ph_rl:.3f}  FL→RR={ph_rr:.3f}  '
                 f'(理想TROT: 0.500 / 0.500 / 0.000)   {verdict}')
        ax.set_title(title, fontsize=11)
        print(f'\n[GaitPlot] 步态周期   = {T_ms:.1f} ms')
        print(f'[GaitPlot] FL→FR 相位  = {ph_fr:.3f}  (TROT理想 0.500)')
        print(f'[GaitPlot] FL→RL 相位  = {ph_rl:.3f}  (TROT理想 0.500)')
        print(f'[GaitPlot] FL→RR 相位  = {ph_rr:.3f}  (TROT理想 0.000 同相)')
        print(f'[GaitPlot] 判定: {verdict}')
    else:
        ax.set_title('步态接触图（数据不足以估算相位）', fontsize=12)

    # --- 下图：每时刻接触脚数 ---
    ax2 = axes[1]
    total = contact_log.sum(axis=1)
    ax2.plot(time, total, color='k', linewidth=0.8)
    ax2.axhline(2, color='green', linestyle='--', linewidth=1, label='TROT理想(2脚)')
    ax2.set_ylim(-0.2, 4.5)
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_xlabel('Time (s)', fontsize=13)
    ax2.set_ylabel('接触脚数', fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(linestyle='--', alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'[GaitPlot] 图片已保存 → {save_path}')
    plt.show()


if __name__ == '__main__':
    # 默认路径与play.py保持一致
    default_csv = os.path.join(
        os.path.dirname(__file__), '..', '..', 'logs', 'gait_contact.csv')
    default_csv = os.path.abspath(default_csv)

    parser = argparse.ArgumentParser(description='TROT步态接触相位图')
    parser.add_argument('--csv',      default=default_csv, help='CSV文件路径')
    parser.add_argument('--start',    type=float, default=0.0,  help='时间窗口起点(s)')
    parser.add_argument('--duration', type=float, default=3.0,  help='时间窗口长度(s)')
    parser.add_argument('--save',     default=None, help='图片保存路径(可选)')
    args = parser.parse_args()

    plot_gait(args.csv, start_time=args.start, duration=args.duration, save_path=args.save)
