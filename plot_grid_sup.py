"""Plot supervised grid search results (single axis: lambda_tv).

Usage:
    python plot_grid_sup.py                          # latest grid_sup_* run
    python plot_grid_sup.py runs/grid_sup_176887     # specific run
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Must match grid_sup.pbs TV_VALS order
TV_VALS = [
    0.0,  0.01, 0.05,
    0.1,  0.2,  0.5,
    0.75, 1.0,  1.5,
    2.0,  3.0,  5.0,
    7.0,  10.0, 15.0,
    20.0, 30.0, 50.0,
]

OUT_DIR = 'results'


def find_latest():
    dirs = sorted(glob.glob('runs/grid_sup_*'), key=os.path.getmtime, reverse=True)
    if not dirs:
        raise FileNotFoundError('No runs/grid_sup_* directories found')
    return dirs[0]


def load_metrics(grid_dir):
    data = {}
    for task_dir in glob.glob(os.path.join(grid_dir, 's*_c*')):
        m = re.search(r's\d+_c(\d+)$', task_dir)
        if not m:
            continue
        cfg = int(m.group(1))
        if cfg >= len(TV_VALS):
            continue
        csv = os.path.join(task_dir, 'metrics.csv')
        if os.path.exists(csv):
            data[cfg] = (TV_VALS[cfg], pd.read_csv(csv))
    return data  # cfg → (tv, df)


def plot_curves(data, col, ylabel, title, out_path, log_scale=False):
    """One panel per lambda_tv value showing tr/val/holdout curves for a given metric."""
    n = len(TV_VALS)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 2.8 * nrows),
                             sharex=False, sharey=True,
                             squeeze=False)
    for cfg in range(n):
        row, col_idx = divmod(cfg, ncols)
        ax = axes[row][col_idx]
        tv = TV_VALS[cfg]
        if cfg not in data:
            ax.text(0.5, 0.5, 'missing', ha='center', va='center',
                    transform=ax.transAxes, color='red')
        else:
            _, df = data[cfg]
            ax.plot(df['epoch'], df[f'tr_{col}'], linewidth=0.8, color='steelblue', label='train')
            ax.plot(df['epoch'], df[f'va_{col}'], linewidth=0.8, color='orange',    label='val')
            ax.plot(df['epoch'], df[f'ho_{col}'], linewidth=0.8, color='green',     label='holdout')
            if log_scale:
                ax.set_yscale('log')
            ax.grid(alpha=0.15)
        ax.set_title(f'λ_tv={tv}', fontsize=8)
        ax.tick_params(labelsize=6)

    # hide unused panels
    for cfg in range(n, nrows * ncols):
        row, col_idx = divmod(cfg, ncols)
        axes[row][col_idx].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color='steelblue', linewidth=1.2, label='train'),
        plt.Line2D([0], [0], color='orange',    linewidth=1.2, label='val'),
        plt.Line2D([0], [0], color='green',     linewidth=1.2, label='holdout'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'Supervised grid — {title} vs epoch', fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


def plot_summary(data, out_path):
    """Best holdout QRS and min val BCE per lambda_tv — two-panel summary."""
    tvs, best_ho_qrs, best_va_qrs, best_va_bce, best_ho_bce = [], [], [], [], []
    for cfg in sorted(data):
        tv, df = data[cfg]
        tvs.append(tv)
        best_ho_qrs.append(df['ho_qrs'].min())
        best_va_qrs.append(df['va_qrs'].min())
        best_va_bce.append(df['va_bce'].min())
        best_ho_bce.append(df['ho_bce'].min())

    x = np.arange(len(tvs))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # ── top: QRS MAE ──────────────────────────────────────────────────────────
    ax1.plot(x, best_ho_qrs, 'o-',  color='green',  linewidth=1.5, markersize=5, label='best holdout QRS')
    ax1.plot(x, best_va_qrs, 's--', color='orange',  linewidth=1.2, markersize=4, label='best val QRS')
    best_idx = int(np.argmin(best_ho_qrs))
    ax1.annotate(f'{best_ho_qrs[best_idx]:.1f} ms\nλ={tvs[best_idx]}',
                 xy=(x[best_idx], best_ho_qrs[best_idx]),
                 xytext=(x[best_idx] + 0.5, best_ho_qrs[best_idx] + 2),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                 fontsize=8)
    ax1.set_ylabel('QRS MAE (ms)')
    ax1.set_title('Supervised grid — best holdout QRS MAE vs λ_tv')
    ax1.legend(fontsize=9); ax1.grid(alpha=0.2)

    # ── bottom: BCE ───────────────────────────────────────────────────────────
    ax2.plot(x, best_ho_bce, 'o-',  color='green',  linewidth=1.5, markersize=5, label='best holdout BCE')
    ax2.plot(x, best_va_bce, 's--', color='orange',  linewidth=1.2, markersize=4, label='best val BCE')
    best_bce_idx = int(np.argmin(best_va_bce))
    ax2.annotate(f'{best_va_bce[best_bce_idx]:.3f}\nλ={tvs[best_bce_idx]}',
                 xy=(x[best_bce_idx], best_va_bce[best_bce_idx]),
                 xytext=(x[best_bce_idx] + 0.5, best_va_bce[best_bce_idx] + 0.01),
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
                 fontsize=8)
    ax2.set_ylabel('BCE loss')
    ax2.set_title('Best val/holdout BCE vs λ_tv')
    ax2.legend(fontsize=9); ax2.grid(alpha=0.2)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in tvs], rotation=45, ha='right', fontsize=8)
    ax2.set_xlabel('λ_tv')

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')


if __name__ == '__main__':
    grid_dir = sys.argv[1] if len(sys.argv) > 1 else find_latest()
    print(f'Grid dir: {grid_dir}')
    os.makedirs(OUT_DIR, exist_ok=True)

    data = load_metrics(grid_dir)
    print(f'Loaded {len(data)}/{len(TV_VALS)} configs')

    plot_curves(data, 'qrs', 'QRS MAE (ms)', 'QRS MAE (ms)',
                os.path.join(OUT_DIR, 'grid_sup_qrs.png'))
    plot_curves(data, 'bce', 'BCE loss', 'BCE loss',
                os.path.join(OUT_DIR, 'grid_sup_bce.png'), log_scale=True)
    plot_curves(data, 'tv',  'TV loss (smoothness)', 'TV loss (smoothness)',
                os.path.join(OUT_DIR, 'grid_sup_tv.png'), log_scale=True)
    plot_summary(data, os.path.join(OUT_DIR, 'grid_sup_summary.png'))
