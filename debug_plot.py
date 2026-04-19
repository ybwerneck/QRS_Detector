"""Debug plotting hook — called periodically during training.

Each call creates a per-epoch subfolder:
    out_dir/epoch_NNNN/
        predictions.png   — scatter + error histograms
        loss_curves.png   — QRS / QT / BCE evolution across datasets
        logits.png        — raw logit curves for 2 validation samples

All arguments are passed as kwargs so new ones can be added freely.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# =========================================================
# Scatter — pred vs target
# =========================================================

def _plot_pred_vs_target(ax_qrs, ax_qt, plot_data=None, **kwargs):
    if plot_data is None or 'val' not in plot_data:
        return
    preds, targets = plot_data['val']

    for ax, col, label in [
        (ax_qrs, 0, 'QRS (ms)'),
        (ax_qt,  1, 'QT (ms)'),
    ]:
        p, t = preds[:, col], targets[:, col]
        mask = ~np.isnan(t)
        p, t = p[mask], t[mask]
        lim  = [min(p.min(), t.min()) - 5, max(p.max(), t.max()) + 5]
        ax.scatter(t, p, s=18, alpha=0.6, edgecolors='none', color='steelblue')
        ax.plot(lim, lim, 'k--', linewidth=0.8)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f'Target {label}', fontsize=8)
        ax.set_ylabel(f'Pred {label}',   fontsize=8)
        mae = np.abs(p - t).mean()
        ax.set_title(f'{label}  MAE={mae:.1f} ms', fontsize=9)
        ax.grid(alpha=0.2)


# =========================================================
# Error histograms
# =========================================================

def _plot_error_histograms(plot_data=None, **kwargs):
    """One row per set (train/val/holdout), one column per metric (QRS/QT)."""
    if plot_data is None:
        return None

    BIN_STEP  = 5
    CLIP      = 50
    bin_edges = np.concatenate([
        [-CLIP - BIN_STEP],
        np.arange(-CLIP, CLIP + 1, BIN_STEP),
        [CLIP + BIN_STEP],
    ])
    tick_pos    = (bin_edges[:-1] + bin_edges[1:]) / 2
    tick_labels = [f'≤{-CLIP}'] + \
                  [str(int(v)) for v in np.arange(-CLIP, CLIP, BIN_STEP)] + \
                  [f'≥{CLIP}']

    sets = [
        ('train',   'steelblue'),
        ('val',     'darkorange'),
        ('holdout', 'seagreen'),
    ]
    present = [(name, color) for name, color in sets if name in plot_data]
    if not present:
        return None

    metrics = [(0, 'QRS (ms)'), (1, 'QT (ms)')]
    n_rows  = len(present)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows), squeeze=False)

    for row, (name, color) in enumerate(present):
        preds, targets = plot_data[name]
        errs = preds - targets
        for col_idx, (col, label) in enumerate(metrics):
            ax = axes[row, col_idx]
            e  = errs[:, col]
            e  = np.clip(e[~np.isnan(e)], bin_edges[0], bin_edges[-1])
            ax.hist(e, bins=bin_edges, color=color, alpha=0.7,
                    edgecolor='white', linewidth=0.4)
            ax.axvline(np.clip(e.mean(), bin_edges[0], bin_edges[-1]),
                       color=color, linewidth=1.2, linestyle='--',
                       label=f'μ={e.mean():.1f}  σ={e.std():.1f}')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha='right')
            ax.set_xlabel(f'Error {label}  (pred − target)', fontsize=8)
            ax.set_ylabel('Count', fontsize=8)
            ax.set_title(f'{name} — {label}', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)

    fig.suptitle('Error distributions by split', fontsize=10)
    plt.tight_layout()
    return fig


# =========================================================
# Loss evolution (NEW)
# =========================================================

def _plot_loss_evolution(history, **kwargs):
    """Three-panel figure: QRS MAE / QT MAE / BCE across train, val, holdout."""
    if not history:
        return None

    epochs = [r['epoch'] for r in history]

    panels = [
        ('QRS MAE (ms)', 'tr_qrs', 'va_qrs', 'ho_qrs'),
        ('QT MAE (ms)',  'tr_qt',  'va_qt',  'ho_qt'),
    ]
    if 'tr_bce' in history[0]:
        panels.append(('BCE', 'tr_bce', 'va_bce', 'ho_bce'))

    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]

    colors  = {'train': 'steelblue', 'val': 'darkorange', 'holdout': 'seagreen'}
    lstyles = {'train': '-',         'val': '-',           'holdout': '--'}

    for ax, (title, tr_key, va_key, ho_key) in zip(axes, panels):
        for split, key, color in [
            ('train',   tr_key, colors['train']),
            ('val',     va_key, colors['val']),
            ('holdout', ho_key, colors['holdout']),
        ]:
            vals = [r[key] for r in history]
            ax.plot(epochs, vals, color=color, linestyle=lstyles[split], label=split)

        ax.set_yscale('log')
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel(title, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, which='both')

    fig.suptitle('Loss evolution', fontsize=10)
    plt.tight_layout()
    return fig


# =========================================================
# Sample logits (NEW)
# =========================================================

def _plot_sample_logits(sample_data=None, **kwargs):
    """N×3 grid: for each sample — lead II + GT | f & g branches | final logit + mask."""
    if sample_data is None:
        return None

    logits    = sample_data['logits']      # (N, 1, 550)
    mask      = sample_data['mask']       # (N, 1, 550)
    f_sig     = sample_data['f_sig']      # (N, 1, 550)
    g_sig     = sample_data['g_sig']      # (N, 1, 550)
    y_mask    = sample_data['y_mask']     # (N, 2, 550)
    all_leads = sample_data.get('all_leads')  # (N, 12, 550) or None

    labels = sample_data.get('labels')  # list of str or None
    n = len(logits)
    t = np.arange(logits.shape[-1])

    fig, axes = plt.subplots(n, 3, figsize=(18, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        gt  = y_mask[i, 0]          # QRS GT mask
        mk  = mask[i, 0]
        lg  = logits[i, 0]

        pred_dur = mk.sum()
        gt_dur   = gt.sum()
        row_label = labels[i] if labels is not None else f'sample {i+1}'

        # ── col 0: all 12 leads superimposed + GT ────────────────
        ax = axes[i, 0]
        if all_leads is not None:
            ecg = all_leads[i]          # (13, W): leads 0-11 ECG, 12 stimulus
            offset_step = 1.5
            cmap = plt.cm.tab20(np.linspace(0, 0.9, 12))
            for li in range(12):
                sig = ecg[li]
                sig_n = (sig - sig.mean()) / (sig.std() + 1e-8)
                ax.plot(t, sig_n + li * offset_step, color=cmap[li],
                        linewidth=0.6, alpha=0.75)
            # stimulus channel (lead 12) — only if present (non-zero)
            stim = ecg[12]
            if stim.any():
                ax.plot(t, stim / (stim.max() + 1e-8) * 1.2 + 12 * offset_step,
                        color='crimson', linewidth=0.8, alpha=0.9, label='stimulus')
        else:
            ax.text(0.5, 0.5, 'leads unavailable', transform=ax.transAxes,
                    ha='center', va='center', fontsize=8, color='grey')
        ax2 = ax.twinx()
        ax2.fill_between(t, 0, gt, color='seagreen', alpha=0.25, label='GT')
        ax2.set_ylim(-0.1, 1.5)
        ax2.set_yticks([])
        ax.set_ylabel('leads (norm + offset)', fontsize=7)
        ax.set_xlabel('time (ms)', fontsize=7)
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        ax.set_title(f'{row_label} — 12-lead ECG', fontsize=8)
        ax.grid(alpha=0.15)
        handles = [Patch(color='seagreen', alpha=0.4, label='GT QRS')]
        ax.legend(handles=handles, fontsize=7, loc='upper right')

        # ── col 1: f (PT branch) and g (EMB branch) ──────────────
        ax = axes[i, 1]
        f = f_sig[i, 0]
        g = g_sig[i, 0]
        ax.fill_between(t, 0, gt, color='seagreen', alpha=0.15)
        ax.plot(t, f, color='steelblue',  linewidth=1.0, label='f  (PT branch)')
        ax.plot(t, g, color='darkorange', linewidth=1.0, label='g  (EMB branch)')
        ax.set_ylabel('z-scored prior', fontsize=7)
        ax.set_xlabel('time (ms)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_title(f'{row_label} — branch priors', fontsize=8)
        handles = [
            plt.Line2D([0], [0], color='steelblue',  label='f  (PT branch)'),
            plt.Line2D([0], [0], color='darkorange',  label='g  (EMB branch)'),
            Patch(color='seagreen', alpha=0.4,        label='GT QRS'),
        ]
        ax.legend(handles=handles, fontsize=7, loc='upper right')
        ax.grid(alpha=0.2)

        # ── col 2: final logit + predicted mask ───────────────────
        ax = axes[i, 2]
        ax2 = ax.twinx()
        ax2.fill_between(t, 0, gt, color='seagreen', alpha=0.2, zorder=1)
        ax2.plot(t, mk, color='darkorange', linewidth=1.0, zorder=2, label='mask')
        ax2.set_ylim(-0.05, 1.3)
        ax2.set_ylabel('mask / GT', fontsize=7, color='darkorange')
        ax2.tick_params(labelsize=6, colors='darkorange')
        ax.plot(t, lg, color='steelblue', linewidth=1.2, zorder=3, label='logit')
        ax.axhline(0, color='steelblue', linewidth=0.5, linestyle=':', alpha=0.5)
        ax.set_ylabel('logit', fontsize=7, color='steelblue')
        ax.set_xlabel('time (ms)', fontsize=7)
        ax.tick_params(labelsize=6, colors='steelblue')
        ax.set_title(
            f'{row_label} — fusion  '
            f'pred={pred_dur:.1f} ms  gt={gt_dur:.0f} ms  err={pred_dur-gt_dur:+.1f} ms',
            fontsize=8,
        )
        ax.grid(alpha=0.2, zorder=0)
        handles = [
            plt.Line2D([0], [0], color='steelblue',  label='logit'),
            plt.Line2D([0], [0], color='darkorange', label='pred mask'),
            Patch(color='seagreen', alpha=0.4,       label='GT'),
        ]
        ax.legend(handles=handles, fontsize=7, loc='upper right')

    fig.suptitle('Sample debug — 12-lead ECG / branch priors / fusion', fontsize=10)
    plt.tight_layout()
    return fig


# =========================================================
# Main hook
# =========================================================

def debug_plot(**kwargs):
    """Render plots into two levels of out_dir.

    out_dir/                  ← run-level: plots that accumulate over time
        loss_curves.png           overwritten on every tick

    out_dir/epoch_NNNN/       ← epoch-level: per-snapshot plots
        predictions.png           scatter + error histograms
        logits.png                sample logit curves

    Expected kwargs
    ---------------
    epoch       : int
    history     : list[dict]
    plot_data   : dict  — keys 'train'/'val'/'holdout', values (preds_np, targets_np) (N,2)
    sample_data : dict  — keys 'logits','mask','y_mask','decision', arrays (N,2,550)/(N,550)
    out_dir     : str
    """
    epoch    = kwargs.get('epoch', 0)
    out_dir  = kwargs.get('out_dir', 'debug')
    step_dir = os.path.join(out_dir, f'epoch_{epoch:04d}')
    os.makedirs(out_dir,  exist_ok=True)
    os.makedirs(step_dir, exist_ok=True)

    # ── run-level: loss evolution (overwritten each tick) ─────────
    fig = _plot_loss_evolution(**kwargs)
    if fig is not None:
        fig.savefig(os.path.join(out_dir, 'loss_curves.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── epoch-level: scatter ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    try:
        _plot_pred_vs_target(ax_qrs=axes[0], ax_qt=axes[1], **kwargs)
    except Exception as e:
        axes[0].set_title(f'_plot_pred_vs_target error: {e}', fontsize=7, color='red')
    fig.suptitle(f'epoch {epoch}', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(step_dir, 'predictions.png'), dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── epoch-level: error histograms (one row per split) ─────────
    fig = _plot_error_histograms(**kwargs)
    if fig is not None:
        fig.savefig(os.path.join(step_dir, 'histograms.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    # ── epoch-level: sample logits ────────────────────────────────
    fig = _plot_sample_logits(**kwargs)
    if fig is not None:
        fig.savefig(os.path.join(step_dir, 'logits.png'), dpi=120, bbox_inches='tight')
        plt.close(fig)

    return step_dir
