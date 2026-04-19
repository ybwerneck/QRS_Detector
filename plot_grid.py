"""Regenerate grid_bce.png, grid_qrs.png, grid_qt.png from runs/."""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS_DIR  = 'runs'
OUT_DIR   = 'results'
PREFIX    = '114_20260418_155140'

TV_VALS  = [0.001, 0.05, 0.2, 1, 5]
DUR_VALS = [0.001, 0.05, 0.2, 1, 5]

def load_metrics():
    data = {}
    for tv in TV_VALS:
        for dur in DUR_VALS:
            run_dir = os.path.join(RUNS_DIR, f'{PREFIX}_tv{tv}_dur{dur}')
            csv = os.path.join(run_dir, 'metrics.csv')
            if os.path.exists(csv):
                data[(tv, dur)] = pd.read_csv(csv)
    return data

def plot_grid(data, col, out_path, title_prefix):
    n_tv  = len(TV_VALS)
    n_dur = len(DUR_VALS)
    fig, axes = plt.subplots(n_tv, n_dur,
                             figsize=(3.5 * n_dur, 2.8 * n_tv),
                             sharex=True, sharey=True)
    for i, tv in enumerate(TV_VALS):
        for j, dur in enumerate(DUR_VALS):
            ax = axes[i][j]
            df = data.get((tv, dur))
            if df is None:
                ax.text(0.5, 0.5, 'missing', ha='center', va='center',
                        transform=ax.transAxes, color='red')
            else:
                epochs = df['epoch']
                ax.plot(epochs, df[f'tr_{col}'],  label='train', linewidth=0.8, color='steelblue')
                ax.plot(epochs, df[f'va_{col}'],  label='val',   linewidth=0.8, color='orange')
                ax.plot(epochs, df[f'ho_{col}'],  label='ho',    linewidth=0.8, color='green')
                ax.set_yscale('log')
                ax.set_ylim(bottom=1e-3)
                ax.grid(alpha=0.15)
            if i == 0:
                ax.set_title(f'dur={dur}', fontsize=8)
            if j == 0:
                ax.set_ylabel(f'tv={tv}', fontsize=8)
            ax.tick_params(labelsize=6)

    # shared legend
    handles = [
        plt.Line2D([0], [0], color='steelblue', linewidth=1.2, label='train'),
        plt.Line2D([0], [0], color='orange',    linewidth=1.2, label='val'),
        plt.Line2D([0], [0], color='green',     linewidth=1.2, label='holdout'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f'{title_prefix}  —  rows: TV weight, cols: dur weight', fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}')

if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_metrics()
    print(f'loaded {len(data)}/25 runs')
    plot_grid(data, 'bce', os.path.join(OUT_DIR, 'grid_bce.png'), 'BCE')
    plot_grid(data, 'qrs', os.path.join(OUT_DIR, 'grid_qrs.png'), 'QRS MAE (ms)')
    plot_grid(data, 'qt',  os.path.join(OUT_DIR, 'grid_qt.png'),  'QT MAE (ms)')
