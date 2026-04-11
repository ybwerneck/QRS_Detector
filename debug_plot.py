"""Debug plotting hook — called periodically during training.

Called as:
    debug_plot(epoch=..., history=..., plot_data=..., out_dir=...)

plot_data : dict with keys 'train', 'val', 'holdout'
            Each value is a (preds_np, targets_np) tuple of shape (N, 2).
            Predictions are already computed with the correct ablation masking.

All arguments are passed as kwargs so new ones can be added freely without
changing the call site. Each sub-plot routine picks what it needs.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =========================================================
# Sub-plot routines
# =========================================================

def _plot_loss_curves(history, ax_qrs, ax_qt, **kwargs):
    """Train / val / holdout MAE curves for QRS and QT."""
    if not history:
        return
    epochs  = [r['epoch']   for r in history]
    tr_qrs  = [r['tr_qrs']  for r in history]
    va_qrs  = [r['va_qrs']  for r in history]
    ho_qrs  = [r['ho_qrs']  for r in history]
    tr_qt   = [r['tr_qt']   for r in history]
    va_qt   = [r['va_qt']   for r in history]
    ho_qt   = [r['ho_qt']   for r in history]

    for ax, tr, va, ho, label in [
        (ax_qrs, tr_qrs, va_qrs, ho_qrs, 'QRS MAE (ms)'),
        (ax_qt,  tr_qt,  va_qt,  ho_qt,  'QT MAE (ms)'),
    ]:
        ax.plot(epochs, tr, label='train', color='steelblue')
        ax.plot(epochs, va, label='val',   color='darkorange')
        ax.plot(epochs, ho, label='holdout', color='seagreen', linestyle='--')
        ax.set_yscale('log')
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(bottom=1, top=2e2)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, which='both')


def _plot_pred_vs_target(ax_qrs, ax_qt, plot_data=None, **kwargs):
    """Scatter of predictions vs targets on the validation set."""
    if plot_data is None or 'val' not in plot_data:
        return
    preds, targets = plot_data['val']   # numpy (N, 2) each

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


def _plot_error_histograms(ax_qrs, ax_qt, plot_data=None, **kwargs):
    """Overlaid error histograms for train / val / holdout, per output.

    Bins: uniform 5 ms steps from -50 to +50, plus one overflow bin on each
    side for errors outside that range.  Y-axis is linear.
    """
    if plot_data is None:
        return

    BIN_STEP = 5
    CLIP     = 50
    bin_edges = np.concatenate([
        [-CLIP - BIN_STEP],
        np.arange(-CLIP, CLIP + 1, BIN_STEP),
        [CLIP + BIN_STEP],
    ])

    sets = [
        ('train',   'steelblue'),
        ('val',     'darkorange'),
        ('holdout', 'seagreen'),
    ]

    all_errors = {}
    for name, _ in sets:
        if name not in plot_data:
            continue
        preds, targets = plot_data[name]
        errs = preds - targets   # (N, 2)
        for col in (0, 1):
            e = errs[:, col]
            e = e[~np.isnan(e)]
            all_errors[(name, col)] = np.clip(e, bin_edges[0], bin_edges[-1])

    y_max = 0
    for e in all_errors.values():
        counts, _ = np.histogram(e, bins=bin_edges)
        y_max = max(y_max, counts.max())
    y_max = y_max * 1.15

    for ax, col, label in [
        (ax_qrs, 0, 'QRS (ms)'),
        (ax_qt,  1, 'QT (ms)'),
    ]:
        for name, color in sets:
            if (name, col) not in all_errors:
                continue
            e = all_errors[(name, col)]
            ax.hist(e, bins=bin_edges, color=color, alpha=0.5,
                    edgecolor='white', linewidth=0.4,
                    label=f'{name}  μ={e.mean():.1f} σ={e.std():.1f}')
            ax.axvline(np.clip(e.mean(), bin_edges[0], bin_edges[-1]),
                       color=color, linewidth=1.0, linestyle='--')

        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
        ax.set_ylim(0, y_max)

        tick_pos    = (bin_edges[:-1] + bin_edges[1:]) / 2
        tick_labels = [f'≤{-CLIP}'] + \
                      [str(int(v)) for v in np.arange(-CLIP, CLIP, BIN_STEP)] + \
                      [f'≥{CLIP}']
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, fontsize=6, rotation=45, ha='right')

        ax.set_xlabel(f'Error {label}  (pred − target)', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.set_title(f'Error distribution — {label}', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)


# =========================================================
# Registry — add new routines here
# =========================================================

# Each entry: (n_rows, n_cols, callable)
# The callable receives all axes for its panel + all kwargs.
_PANELS = [
    # (rows, cols, fn)
    (1, 2, _plot_loss_curves),       # axes: ax_qrs, ax_qt
    (1, 2, _plot_pred_vs_target),    # axes: ax_qrs, ax_qt
    (1, 2, _plot_error_histograms),  # axes: ax_qrs, ax_qt
]


# =========================================================
# Main hook
# =========================================================


def debug_plot(**kwargs):
    """Call from the training loop.  Saves to out_dir/debug_<epoch>.png.

    Expected kwargs
    ---------------
    epoch     : int
    history   : list[dict]  — each dict has keys epoch/tr_qrs/va_qrs/ho_qrs/tr_qt/va_qt/ho_qt
    plot_data : dict        — keys 'train'/'val'/'holdout', values (preds_np, targets_np) (N,2)
    out_dir   : str         (default 'debug')
    """
    epoch   = kwargs.get('epoch', 0)
    out_dir = kwargs.get('out_dir', 'debug')
    os.makedirs(out_dir, exist_ok=True)

    n_panels = len(_PANELS)
    fig, axes = plt.subplots(n_panels, 2, figsize=(12, n_panels * 4))
    if n_panels == 1:
        axes = axes[np.newaxis, :]

    for row, (_, _, fn) in enumerate(_PANELS):
        ax_left, ax_right = axes[row, 0], axes[row, 1]
        try:
            fn(ax_qrs=ax_left, ax_qt=ax_right, **kwargs)
        except Exception as e:
            ax_left.set_title(f'{fn.__name__} error: {e}', fontsize=7, color='red')

    zero_emb = kwargs.get('zero_emb', False)
    zero_dec = kwargs.get('zero_dec', False)
    variant  = os.path.basename(kwargs.get('out_dir', 'debug').rstrip('/'))
    ablation = ('full' if not zero_emb and not zero_dec
                else 'emb_only' if not zero_emb and zero_dec
                else 'dec_only' if zero_emb and not zero_dec
                else 'none')
    fig.suptitle(f'{variant}  —  epoch {epoch}  |  ablation={ablation}  '
                 f'(zero_emb={zero_emb}, zero_dec={zero_dec})', fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, f'debug_{epoch:04d}.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return path
