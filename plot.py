"""Plotting utilities for beat inspection."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from beat import WINDOW_PRE, WINDOW_POST


def plot_signal_windows(signal_1d, beats, t_start=0, t_end=5000,
                        decision=None, threshold=None):
    """Lead II signal (top) with optional Pan-Tompkins decision variable mirrored below.

    Grey region  = saved window around each beat.
    Red line     = detected spike.
    Orange line  = detection threshold (shown on decision panel).

    Parameters
    ----------
    decision  : np.ndarray  — output of pan_tompkins_signal(), same length as signal
    threshold : float       — the detection threshold to draw on the decision panel
    """
    t    = np.arange(len(signal_1d))
    mask = (t >= t_start) & (t < t_end)

    n_rows = 2 if decision is not None else 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows),
                             sharex=True, gridspec_kw={'hspace': 0})
    ax_sig = axes[0] if n_rows == 2 else axes

    ax_sig.plot(t[mask], signal_1d[mask], color='steelblue', linewidth=0.7)
    ax_sig.set_ylabel('Amplitude')
    ax_sig.set_title(f'Lead II  —  spike + window markers  [{t_start}–{t_end} ms]')

    sorted_beats = sorted(beats, key=lambda b: b.spike_idx)
    prev_win_hi  = 0
    for b in sorted_beats:
        win_lo = max(b.spike_idx - b.window_pre, prev_win_hi)  # borrowed region, but no overlap with prev
        win_hi = b.spike_idx + b.window_post
        prev_win_hi = win_hi
        if win_lo >= t_end or win_hi <= t_start:
            continue
        lo = max(win_lo, t_start)
        hi = min(win_hi, t_end)
        ax_sig.axvspan(lo, hi, color='grey', alpha=0.15)
        ax_sig.axvline(b.spike_idx, color='red', linewidth=0.8, alpha=0.7)

    if decision is not None:
        ax_dec = axes[1]
        dec = decision[mask]
        # Mirror: invert y-axis so the decision variable grows downward
        ax_dec.plot(t[mask], dec, color='darkorange', linewidth=0.7)
        if threshold is not None:
            ax_dec.axhline(threshold, color='black', linewidth=0.8,
                           linestyle='--', label=f'thr={threshold:.1f}')
            ax_dec.legend(fontsize=7, loc='upper right')
        for b in beats:
            if t_start <= b.spike_idx < t_end:
                ax_dec.axvline(b.spike_idx, color='red', linewidth=0.8, alpha=0.7)
       # ax_dec.invert_yaxis()
        ax_dec.set_ylabel('Decision var.')
        ax_dec.set_xlabel('Time (ms)')
    else:
        ax_sig.set_xlabel('Time (ms)')

    ax_sig.set_xlim(t_start, t_end)
    plt.tight_layout()
    return fig


def plot_beat(beat, lead_names=None, ax=None):
    """All leads of one beat, each with a distinct colour, offset vertically.

    X-axis is ms relative to the spike (centre = 0).
    Y-axis is amplitude + per-channel offset (no scale meaning).
    """
    window = beat.window                    # (n_leads, window_size)
    n_leads, n_samples = window.shape
    x = np.arange(n_samples) - beat.window_pre   # ms from spike

    scale  = np.percentile(np.abs(window), 95) or 1.0
    colors = plt.cm.tab20(np.linspace(0, 1, n_leads))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.get_figure()

    for i, (ch, color) in enumerate(zip(window, colors)):
        offset = i * scale * 2.5
        label  = lead_names[i] if lead_names else f'ch{i}'
        ax.plot(x, ch + offset, color=color, linewidth=0.7, label=label)
        ax.text(x[-1] + 5, ch[-1] + offset, label,
                fontsize=6, color=color, va='center')

    ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('ms from spike')
    ax.set_yticks([])
    title = f'spike={beat.spike_idx}'
    if beat.qrs_duration:
        title += f'  QRS={beat.qrs_duration:.0f}ms'
    if beat.qt_interval:
        title += f'  QT={beat.qt_interval:.0f}ms'
    ax.set_title(title, fontsize=8)

    if own_fig:
        plt.tight_layout()
    return fig


def _beat_annotation_spans(beat):
    """Return (qrs_rel_start, qrs_rel_end, qt_rel_start, qt_rel_end) in ms from spike.
    Any value is None when the annotation is absent."""
    qrs_rel_start = qrs_rel_end = qt_rel_start = qt_rel_end = None
    if beat.qrs_start is not None and beat.qrs_duration is not None:
        qrs_rel_start = beat.qrs_start - beat.spike_idx
        qrs_rel_end   = qrs_rel_start + beat.qrs_duration
    if beat.qt_start is not None and beat.qt_interval is not None:
        qt_rel_start = beat.qt_start - beat.spike_idx
        qt_rel_end   = qt_rel_start + beat.qt_interval
    return qrs_rel_start, qrs_rel_end, qt_rel_start, qt_rel_end


def _draw_spans(ax, qrs_rel_start, qrs_rel_end, qt_rel_start, qt_rel_end):
    """Paint QRS (blue) and ST+T (orange) background spans on an Axes."""
    if qrs_rel_start is not None:
        ax.axvspan(qrs_rel_start, qrs_rel_end, color='steelblue', alpha=0.18, zorder=0)
    if qt_rel_start is not None:
        st_start = qrs_rel_end if qrs_rel_end is not None else qt_rel_start
        ax.axvspan(st_start, qt_rel_end, color='darkorange', alpha=0.15, zorder=0)
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.5, zorder=1)


def plot_annotated_beat(beat, lead_names=None, fig=None):
    """All leads of one beat, each in its own subplot row with independent y-scale.

    Regions (drawn on every row)
    ----------------------------
    Blue   = QRS complex   [qrs_start → qrs_start + qrs_duration]
    Orange = ST + T wave   [qrs_end   → qt_start  + qt_interval ]
    Dashed = spike (R peak, x = 0)

    Parameters
    ----------
    beat       : Beat
    lead_names : list[str] | None
    fig        : Figure | None   pass a subfigure to embed inside plot_annotated_beats
    """
    window   = beat.window
    n_leads, n_samples = window.shape
    x        = np.arange(n_samples) - beat.window_pre
    colors   = plt.cm.tab20(np.linspace(0, 1, n_leads))

    qrs_rel_start, qrs_rel_end, qt_rel_start, qt_rel_end = _beat_annotation_spans(beat)

    if fig is None:
        fig = plt.figure(figsize=(10, n_leads * 1.3))

    axes = fig.subplots(n_leads, 1, sharex=True,
                        gridspec_kw={'hspace': 0})

    for i, (ax, ch, color) in enumerate(zip(axes, window, colors)):
        ax.plot(x, ch, color=color, linewidth=0.8, zorder=2)
        _draw_spans(ax, qrs_rel_start, qrs_rel_end, qt_rel_start, qt_rel_end)

        # lead name as text inside the plot, no y-axis at all
        name = lead_names[i] if lead_names else f'ch{i}'
        ax.text(0.01, 0.5, name, transform=ax.transAxes,
                fontsize=6, va='center', ha='left', color=color,
                fontweight='bold')

        ax.set_yticks([])
        ax.tick_params(axis='x', labelbottom=(i == n_leads - 1), length=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # draw a subtle separator line at the top of each row
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_linewidth(0.3)
        ax.spines['top'].set_color('#cccccc')

    axes[-1].set_xlabel('ms from spike', fontsize=8)
    axes[-1].spines['bottom'].set_visible(True)
    axes[-1].spines['bottom'].set_linewidth(0.5)

    # legend on first row only
    legend_handles = []
    if qrs_rel_start is not None:
        legend_handles.append(
            Patch(color='steelblue', alpha=0.5, label=f'QRS {beat.qrs_duration:.0f} ms'))
    if qt_rel_start is not None:
        legend_handles.append(
            Patch(color='darkorange', alpha=0.5, label=f'QT {beat.qt_interval:.0f} ms'))
    if legend_handles:
        axes[0].legend(handles=legend_handles, fontsize=6,
                       loc='upper right', framealpha=0.7)

    LABEL_TAG = {0: 'normal', 1: 'arrhythmia', 2: 'extrasystole'}
    label_str = LABEL_TAG.get(beat.label, '?') if beat.label is not None else 'unlabelled'
    noisy_tag = '  [NOISY]' if beat.noisy else ''
    axes[0].set_title(
        f'spike={beat.spike_idx} ms  |  {label_str}{noisy_tag}',
        fontsize=8,
    )

    return fig


def plot_annotated_beats(beats, lead_names=None, n=9, cols=3):
    """Grid of annotated-beat plots using subfigures (one full multi-lead beat per cell)."""
    n    = min(n, len(beats))
    rows = int(np.ceil(n / cols))

    n_leads = beats[0].window.shape[0] if beats else 13
    fig = plt.figure(figsize=(cols * 8, rows * n_leads * 1.1))
    subfigs = fig.subfigures(rows, cols, wspace=0.04, hspace=0.08)
    subfigs_flat = np.asarray(subfigs).flatten()

    for subfig, beat in zip(subfigs_flat[:n], beats[:n]):
        plot_annotated_beat(beat, lead_names, fig=subfig)

    for subfig in subfigs_flat[n:]:
        subfig.set_visible(False)

    return fig


def plot_beats(beats, lead_names=None, n=9, cols=3):
    """Grid of single-beat plots (one subplot per beat)."""
    n    = min(n, len(beats))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 5, rows * 4),
                              squeeze=False)
    axes_flat = axes.flatten()

    for ax, beat in zip(axes_flat, beats[:n]):
        plot_beat(beat, lead_names, ax=ax)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig
