"""Regenerate grid_bce.png, grid_qrs.png, grid_qt.png from runs dynamically."""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS_DIR = 'grids/tv_scale_shift_sweep_20260421_225521/runs/'
OUT_DIR  = 'grids/tv_scale_shift_sweep_20260421_225521/results'


# -----------------------------
# Parse parameters from run.log
# -----------------------------
def parse_run_log(log_path):
    with open(log_path, 'r') as f:
        text = f.read()

    tv_scale = None
    tv_shift = None

    m_scale = re.search(r'--lambda_tv_scale\s+([0-9.eE+-]+)', text)
    m_shift = re.search(r'--lambda_tv_shift\s+([0-9.eE+-]+)', text)

    if m_scale:
        tv_scale = float(m_scale.group(1))
    if m_shift:
        tv_shift = float(m_shift.group(1))

    return tv_scale, tv_shift


# -----------------------------
# Load all runs dynamically
# -----------------------------
def load_metrics():
    data = {}
    run_dirs = glob.glob(os.path.join(RUNS_DIR, '*'))

    for run_dir in run_dirs:
        log_path = os.path.join(run_dir, 'run.log')
        csv_path = os.path.join(run_dir, 'metrics.csv')

        if not os.path.exists(log_path) or not os.path.exists(csv_path):
            continue

        tv, dur = parse_run_log(log_path)

        if tv is None or dur is None:
            print(f'skipping {run_dir} (missing params)')
            continue

        key = (round(tv, 8), round(dur, 8))

        print(f'loading {run_dir} (tv={tv}, dur={dur})')
        df = pd.read_csv(csv_path)

        data[key] = df

    return data


# -----------------------------
# Plot grid
# -----------------------------
def plot_grid(data, col, out_path, title_prefix, y_min=1e-3):
    if not data:
        print("No data to plot.")
        return

    TV_VALS  = sorted(set(k[0] for k in data.keys()))
    DUR_VALS = sorted(set(k[1] for k in data.keys()))

    n_tv  = len(TV_VALS)
    n_dur = len(DUR_VALS)

    fig, axes = plt.subplots(
        n_tv, n_dur,
        figsize=(3.5 * n_dur, 2.8 * n_tv),
        sharex=True, sharey=True
    )

    # Ensure 2D axes even if 1xN or Nx1
    if n_tv == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_dur == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, tv in enumerate(TV_VALS):
        for j, dur in enumerate(DUR_VALS):
            ax = axes[i][j]
            key = (tv, dur)
            df = data.get(key)

            if df is None:
                ax.text(
                    0.5, 0.5, 'missing',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    color='red'
                )
            else:
                epochs = df['epoch']

                ax.plot(epochs, df[f'tr_{col}'],
                        label='train', linewidth=0.8, color='steelblue')
                ax.plot(epochs, df[f'va_{col}'],
                        label='val', linewidth=0.8, color='orange')
                ax.plot(epochs, df[f'ho_{col}'],
                        label='ho', linewidth=0.8, color='green')

                ax.set_yscale('log')
                ax.set_ylim(bottom=y_min)
                ax.grid(alpha=0.15)

            if i == 0:
                ax.set_title(f'dur={dur}', fontsize=8)
            if j == 0:
                ax.set_ylabel(f'tv={tv}', fontsize=8)

            ax.tick_params(labelsize=6)

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color='steelblue', linewidth=1.2, label='train'),
        plt.Line2D([0], [0], color='orange',    linewidth=1.2, label='val'),
        plt.Line2D([0], [0], color='green',     linewidth=1.2, label='holdout'),
    ]

    fig.legend(
        handles=handles,
        loc='lower center',
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01)
    )

    fig.suptitle(
        f'{title_prefix}  —  rows: TV weight, cols: shift weight',
        fontsize=11
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

    print(f'saved {out_path}')


# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_metrics()
    print(f'loaded {len(data)} runs')

    plot_grid(
        data,
        'bce',
        os.path.join(OUT_DIR, 'grid_bce.png'),
        'BCE',
        y_min=1e-2
    )

    plot_grid(
        data,
        'qrs',
        os.path.join(OUT_DIR, 'grid_qrs.png'),
        'QRS MAE (ms)',
        y_min=1
    )

    plot_grid(
        data,
        'qt',
        os.path.join(OUT_DIR, 'grid_qt.png'),
        'QT MAE (ms)'
    )