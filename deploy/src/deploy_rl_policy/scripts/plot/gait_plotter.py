#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Threshold for contact (absolute Fz)
CONTACT_THRESHOLD = 30.0

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
            if j < n:
                end = time[j]
            else:
                end = time[-1]
            duration = end - start
            if duration <= 0:
                duration = 1e-3
            intervals.append((start, duration))
            i = j
        else:
            i += 1
    return intervals
def resample_to_fixed_dt(df, dt=0.001):
    """
    Resample dataframe to fixed dt (seconds) using nearest neighbor contact status.
    Assumes Time (s) column exists.
    """
    t_min = df['Time (s)'].min()
    t_max = df['Time (s)'].max()
    time_fixed = np.arange(t_min, t_max + dt/2, dt)  # include last point
    df_resampled = pd.DataFrame({'Time (s)': time_fixed})
    # 对每条 contact 列使用 nearest 填充
    for leg in ['FL', 'FR', 'RL', 'RR']:
        col = f'{leg}_Contact_Status'
        df_resampled[col] = np.interp(time_fixed, df['Time (s)'], df[col], left=0, right=0)
        # 将小数转成 0/1
        df_resampled[col] = (df_resampled[col] > 0.5).astype(int)
    return df_resampled
def _intervals_from_two_bool_series_overlap(time, a_series, b_series):
    both = np.logical_and(np.asarray(a_series).astype(bool), np.asarray(b_series).astype(bool))
    return _intervals_from_bool_series(time, both)

def load_and_make_contact(df, contact_threshold=CONTACT_THRESHOLD):
    """
    Ensure dataframe has Time (s) and contact status columns.
    If df contains Fz columns (FL_Fz, FR_Fz, RL_Fz, RR_Fz) or raw 5-column data,
    compute contact status columns using abs(Fz) > contact_threshold.
    """
    # If Time column absent but first column exists, try to infer
    cols = df.columns.tolist()

    # Normalize column names by stripping whitespace
    cols_stripped = [str(c).strip() for c in cols]

    # Case A: Has header with FL_Fz etc.
    expected_fz = ['FL_Fz', 'FR_Fz', 'RL_Fz', 'RR_Fz']
    if 'Time (s)' in cols_stripped and all(c in cols_stripped for c in expected_fz):
        # rename stripped columns mapping if needed
        df.columns = cols_stripped
        # ensure numeric
        df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
        for c in expected_fz:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    else:
        # Case B: maybe no header, raw 5 columns: time, f1, f2, f3, f4
        # Try to coerce all to numeric and assign names
        if df.shape[1] >= 5:
            # rename first five columns
            new_names = ['Time (s)', 'FL_Fz', 'FR_Fz', 'RL_Fz', 'RR_Fz'] + \
                        [f'col_{i}' for i in range(5, df.shape[1])]
            df.columns = new_names[:df.shape[1]]
            df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
            for c in ['FL_Fz', 'FR_Fz', 'RL_Fz', 'RR_Fz']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            raise ValueError("CSV does not contain enough columns. Expect at least 5 columns: time + 4 Fz values.")

    # Drop rows with NaN timestamp
    df = df.dropna(subset=['Time (s)']).reset_index(drop=True)

    # Compute contact status based on threshold
    df['FL_Contact_Status'] = (df['FL_Fz']<- contact_threshold).astype(int)
    df['FR_Contact_Status'] = (df['FR_Fz']<- contact_threshold).astype(int)
    df['RL_Contact_Status'] = (df['RL_Fz']<- contact_threshold).astype(int)
    df['RR_Contact_Status'] = (df['RR_Fz']<- contact_threshold).astype(int)

    return df

def plot_gait_bars(file_name, start_time=0.0, duration=1.0, contact_threshold=CONTACT_THRESHOLD):
    """
    Read CSV of raw Fz data (or precomputed contact columns), compute contact boolean by threshold,
    and plot contact-phase bars for each leg for a specific time window.
    Args:
        file_name (str): CSV path.
        start_time (float): window start time in seconds.
        duration (float): window duration in seconds.
        contact_threshold (float): threshold for |Fz| to consider contact.
    """
    if not os.path.exists(file_name):
        print(os.getcwd())
        print(f"Error: file '{file_name}' not found. Make sure the recording script generated it.")
        return

    try:
        # Try reading with header; if no header, pandas will still read as columns [0..]
        df = pd.read_csv(file_name, header=0)
    except pd.errors.EmptyDataError:
        print(f"Error: file '{file_name}' is empty.")
        return

    # Prepare dataframe and compute contact columns
    try:
        df = load_and_make_contact(df, contact_threshold=contact_threshold)
        df = resample_to_fixed_dt(df, dt=0.01)
    except ValueError as e:
        print(f"Error processing CSV: {e}")
        return

    # Ensure time column is numeric and sorted
    df['Time (s)'] = pd.to_numeric(df['Time (s)'], errors='coerce')
    df = df.sort_values('Time (s)').reset_index(drop=True)

    # Crop to requested time window
    window_end = start_time + duration
    df_win = df[(df['Time (s)'] >= start_time) & (df['Time (s)'] <= window_end)].copy()

    if df_win.empty:
        print(f"Warning: No data in the time window [{start_time}, {window_end}] s. Available time range: "
              f"{df['Time (s)'].min()} - {df['Time (s)'].max()} s")
        return
    else:
        actual_start = df_win['Time (s)'].min()
        actual_end = df_win['Time (s)'].max()
        if actual_start > start_time or actual_end < window_end:
            print(f"Note: Window clipped to available data: [{actual_start:.3f}, {actual_end:.3f}] s")

    time = df_win['Time (s)'].to_numpy()
    legs = ['FL', 'FR', 'RL', 'RR']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Prepare figure
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=100)
    y_height = 0.8
    y_gap = 0.2
    y_positions = {
        'FL': 3 * (y_height + y_gap),
        'FR': 2 * (y_height + y_gap),
        'RL': 1 * (y_height + y_gap),
        'RR': 0 * (y_height + y_gap),
    }

    for leg, color in zip(legs, colors):
        col = f'{leg}_Contact_Status'
        intervals = _intervals_from_bool_series(time, df_win[col].to_numpy())
        ax.broken_barh(intervals, (y_positions[leg], y_height),
                       facecolors=color, edgecolors='k', linewidth=0.5, alpha=0.9, label=leg)

    fl_rr_intervals = _intervals_from_two_bool_series_overlap(time, df_win['FL_Contact_Status'].to_numpy(), df_win['RR_Contact_Status'].to_numpy())
    fr_rl_intervals = _intervals_from_two_bool_series_overlap(time, df_win['FR_Contact_Status'].to_numpy(), df_win['RL_Contact_Status'].to_numpy())

    for start, dur in fl_rr_intervals:
        ax.axvspan(start, start + dur, ymin=0.0, ymax=1.0, facecolor='#bbbbff', alpha=0.12)
    for start, dur in fr_rl_intervals:
        ax.axvspan(start, start + dur, ymin=0.0, ymax=1.0, facecolor='#ffdddd', alpha=0.12)

    ax.set_ylim(-0.2, 4 * (y_height + y_gap))
    ax.set_yticks([y_positions[l] + y_height / 2 for l in legs])
    ax.set_yticklabels(legs)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_xlabel('Time (s)',fontsize=16)
    # ax.set_title(f'Gait Contact Timeline — {start_time:.2f}s to {window_end:.2f}s')
    ax.grid(axis='x', linestyle='--', alpha=0.4)

    from matplotlib.patches import Patch
    patches = [Patch(facecolor=colors[i], edgecolor='k', label=legs[i]) for i in range(len(legs))]
    ax.legend(handles=patches, bbox_to_anchor=(1.02, 1.0),fontsize=20, loc='upper left')
    plt.savefig(save_path, dpi=100,bbox_inches='tight')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    gait="walk"
    csv_to_plot = f"/home/song/cpg_jump/deploy/csv/gait_contact_forces_10s_{gait}.csv"
    save_path = f"/home/song/cpg_jump/deploy/figures/contact_{gait}.png"
    # choose window start and duration (seconds)
    start_time = 6
    duration = 2
    plot_gait_bars(csv_to_plot, start_time=start_time, duration=duration, contact_threshold=CONTACT_THRESHOLD)
