"""Post-processing: detect beats with overlapping windows and correlate with error.

Uses the same window-overlap logic as mark_noisy_beats in beat.py:
  window i spans [spike_idx[i] - window_pre[i], spike_idx[i] + window_post[i])
  two beats overlap iff  lo_i < hi_j  AND  hi_i > lo_j  (same source file)

Beat-cache metadata (spike_idx, window_pre, source) is aligned to the emb cache
by sequential decision-window matching, correctly skipping the small number of
beats that BeatDataset drops (qrs mask quality check).

Per-beat CSV: <out_dir>/peaks.csv
Plots:        <out_dir>/peaks_summary.png
              <out_dir>/peaks_examples.png

Usage:
  python analyze_peaks.py
  python analyze_peaks.py --ckpt ckpts/head_best_val.pt   # + per-beat errors
"""

import os
import glob
import json
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from beat import (load_ecg, load_patient_beats, FS,
                  WINDOW_PRE, WINDOW_POST, WINDOW_SIZE, _pan_tompkins_detect)


# ── alignment helpers ─────────────────────────────────────────────────────────

def align_beats_to_emb(ann_decision: np.ndarray, emb_decision: np.ndarray) -> np.ndarray:
    """Return the ann-cache indices that correspond to emb-cache entries.

    BeatDataset silently drops a handful of beats (clipped QRS mask).
    Sequential matching on the decision window array recovers the correspondence.
    Both arrays have shape (N, 12, 550).
    """
    kept = []
    emb_idx = 0
    for ann_idx in range(len(ann_decision)):
        if emb_idx >= len(emb_decision):
            break
        if np.allclose(ann_decision[ann_idx], emb_decision[emb_idx], atol=1e-6):
            kept.append(ann_idx)
            emb_idx += 1
    assert len(kept) == len(emb_decision), (
        f'Alignment failed: matched {len(kept)} but expected {len(emb_decision)}'
    )
    return np.array(kept, dtype=np.int32)



# ── overlap detection ─────────────────────────────────────────────────────────
def compute_overlaps(spike_idx: np.ndarray,
                     window_pre: np.ndarray,
                     source: np.ndarray,
                     all_spikes_by_source: dict) -> dict:
    """For each beat, count how many other beat windows overlap with its window.

    Two windows overlap iff they share any sample:
      beat i : [spike_i - window_pre_i, spike_i + window_post_i)
      beat j : [spike_j - WINDOW_PRE,   spike_j + WINDOW_POST)

    Returns dict of arrays, all length N:
      n_beats_in_window  int   — number of other windows that overlap
      first_overlap_pos  float — window-relative position of nearest overlapping spike; NaN if none
    """
    N          = len(spike_idx)
    window_post = WINDOW_SIZE - window_pre  # (N,)
    n_beats    = np.zeros(N, dtype=np.int32)
    first_pos  = np.full(N, np.nan)

    for i in range(N):
        src    = source[i]
        all_sp = all_spikes_by_source.get(src, np.array([], dtype=np.int64))
        lo_i   = spike_idx[i] - window_pre[i]
        hi_i   = spike_idx[i] + window_post[i]
        # other beat j's window [spike_j - WINDOW_PRE, spike_j + WINDOW_POST) overlaps
        # beat i's window [lo_i, hi_i) iff they share any sample
        overlapping = all_sp[
            (all_sp != spike_idx[i]) &
            (all_sp - WINDOW_PRE < hi_i) &
            (all_sp + WINDOW_POST > lo_i)
        ]
        n_beats[i] = len(overlapping)
        if len(overlapping):
            rel = overlapping - lo_i   # position relative to window start
            nearest = rel[np.argmin(np.abs(rel - window_pre[i]))]
            first_pos[i] = float(nearest)

    return dict(
        n_beats_in_window=n_beats,
        overlaps_any=n_beats > 0,
        first_overlap_pos=first_pos,
    )


# ── inference for per-beat errors ─────────────────────────────────────────────

@torch.no_grad()
def infer_errors(head, emb_path: str, dec_path: str, ys: np.ndarray,
                 device, batch_size: int = 64) -> np.ndarray:
    embs = torch.from_numpy(np.load(emb_path, mmap_mode='r'))
    decs = torch.from_numpy(np.load(dec_path))
    errs = []
    head.eval()
    for i in range(0, len(embs), batch_size):
        e  = embs[i:i+batch_size].to(device)
        d  = decs[i:i+batch_size].to(device)
        ym = torch.from_numpy(ys[i:i+batch_size]).to(device)
        _, mask, _ = head(e, d)
        hard = (mask > 0.5).float().sum(dim=-1)[:, 0]
        gt   = ym[:, 0].float().sum(dim=-1)
        errs.append((hard - gt).abs().cpu().numpy())
    return np.concatenate(errs)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_summary(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    splits = df['split'].unique()
    colors = {'annotated': '#4c72b0', 'unannotated': '#c44e52'}
    has_err = df['err_ms'].notna().any()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Overlapping beat analysis', fontsize=13)

    # 1. n_beats_in_window distribution (normalised per split)
    ax = axes[0, 0]
    for sp in splits:
        sub = df[df['split'] == sp]
        n_total = len(sub)
        bins = range(0, sub['n_beats_in_window'].max() + 2)
        counts, edges = np.histogram(sub['n_beats_in_window'], bins=bins)
        ax.bar(edges[:-1], counts / n_total, width=0.4,
               alpha=0.6, label=f'{sp} (n={n_total})', color=colors.get(sp, 'grey'),
               align='edge')
    ax.set_xlabel('# other spikes in window');  ax.set_ylabel('fraction of beats')
    ax.set_title('Other detected spikes per window (normalised)');  ax.legend(fontsize=8)

    # 2. overlaps_any rate per split
    ax = axes[0, 1]
    for sp in splits:
        sub = df[df['split'] == sp]
        rate = sub['overlaps_any'].mean()
        n_total = len(sub)
        ax.bar(sp, rate, color=colors.get(sp, 'grey'), alpha=0.8)
        ax.text(sp, rate + 0.01, f'{rate:.1%}\n(n={n_total})', ha='center', fontsize=9)
    ax.set_ylabel('Fraction with ≥1 overlapping beat')
    ax.set_title('Overlap rate per split')
    ax.set_ylim(0, 1.1)

    # 3. n_beats_in_window distribution (normalised)
    ax = axes[0, 2]
    for sp in splits:
        sub = df[df['split'] == sp]
        n_total = len(sub)
        counts = sub['n_beats_in_window'].value_counts().sort_index()
        offset = list(splits).index(sp) * 0.25
        ax.bar(counts.index + offset, counts.values / n_total, width=0.25,
               label=f'{sp} (n={n_total})', color=colors.get(sp, 'grey'), alpha=0.8)
    ax.set_xlabel('# additional beats in window');  ax.set_ylabel('fraction of beats')
    ax.set_title('Additional beats per window (normalised)');  ax.legend(fontsize=8)

    # 4. distance from anchor to nearest other spike — fine bins around 100ms
    ax = axes[1, 0]
    # non-uniform bins: 5ms steps up to 300ms, then 25ms steps to window end
    bins = np.concatenate([np.arange(0, 305, 5), np.arange(325, WINDOW_SIZE + 26, 25)])
    for sp in splits:
        sub = df[(df['split'] == sp) & df['first_overlap_pos'].notna()].copy()
        n_total = len(df[df['split'] == sp])
        sub['dist_ms'] = (sub['first_overlap_pos'] - sub['window_pre']).abs()
        counts, edges = np.histogram(sub['dist_ms'], bins=bins)
        widths = np.diff(edges)
        ax.bar(edges[:-1], counts / n_total, width=widths,
               alpha=0.5, label=f'{sp} (n={n_total})', color=colors.get(sp, 'grey'),
               align='edge')
    ax.axvline(100, color='red',    ls='--', lw=1, label='100ms (min_distance)')
    ax.axvline(300, color='orange', ls=':',  lw=1, label='300ms (physio min RR)')
    ax.set_xlabel('Distance from anchor spike (ms)')
    ax.set_ylabel('fraction of beats')
    ax.set_title('Nearest other spike — distance from anchor (normalised)')
    ax.legend(fontsize=8)

    # 5. err_ms vs first_overlap_pos
    ax = axes[1, 1]
    if has_err:
        for sp in splits:
            sub = df[(df['split'] == sp) & df['first_overlap_pos'].notna()].dropna(subset=['err_ms']).copy()
            if len(sub):
                sub['pos_bin'] = pd.cut(sub['first_overlap_pos'], bins=8)
                med = sub.groupby('pos_bin', observed=False)['err_ms'].median()
                mids = [iv.mid for iv in med.index]
                ax.plot(mids, med.values, marker='o', label=sp, color=colors.get(sp, 'grey'))
        ax.set_xlabel('First other spike position (sample)')
        ax.set_ylabel('Median QRS error (ms)')
        ax.set_title('Error vs overlap position\n(earlier = more contamination)')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No per-beat errors.\nRun with --ckpt to populate.',
                ha='center', va='center', transform=ax.transAxes, fontsize=9, color='grey')
        ax.axis('off')

    # 6. err_ms vs n_beats_in_window
    ax = axes[1, 2]
    if has_err:
        for sp in splits:
            sub = df[(df['split'] == sp)].dropna(subset=['err_ms'])
            ns   = sorted(sub['n_beats_in_window'].unique())
            meds = [sub[sub['n_beats_in_window'] == n]['err_ms'].median() for n in ns]
            counts = [len(sub[sub['n_beats_in_window'] == n]) for n in ns]
            ax.plot(ns, meds, marker='o', label=sp, color=colors.get(sp, 'grey'))
            for n_val, med, cnt in zip(ns, meds, counts):
                ax.text(n_val, med + 0.3, f'n={cnt}', ha='center', fontsize=6)
        ax.set_xlabel('# additional beats in window')
        ax.set_ylabel('Median QRS error (ms)')
        ax.set_title('Error vs # beats in window')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'No per-beat errors.\nRun with --ckpt to populate.',
                ha='center', va='center', transform=ax.transAxes, fontsize=9, color='grey')
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(out_dir, 'peaks_summary.png')
    plt.savefig(path, dpi=150);  plt.close()
    print(f'  saved {path}')


def plot_examples(df: pd.DataFrame, all_leads_dict: dict, ys_dict: dict, out_dir: str, n: int = 16):
    """Plot 12-lead ECG (superimposed, z-scored + offset) for the most overlapping beats."""
    sub = (df[df['overlaps_any'] & df['subsplit'].isin(all_leads_dict)]
           .sort_values(['n_beats_in_window', 'first_overlap_pos'], ascending=[False, True])
           .head(n))
    if len(sub) == 0:
        return

    cmap        = plt.cm.tab20(np.linspace(0, 0.9, 12))
    offset_step = 1.5
    t           = np.arange(WINDOW_SIZE)

    ncols = 4
    nrows = max(1, int(np.ceil(len(sub) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2), squeeze=False)
    axes = axes.flatten()
    fig.suptitle('Most-overlapping beats — 12-lead ECG superimposed', fontsize=11)

    for ax_i, (_, row) in enumerate(sub.iterrows()):
        ax   = axes[ax_i]
        sp   = row['subsplit']
        bidx = int(row['beat_idx'])
        ecg  = all_leads_dict[sp][bidx]   # (13, 550)
        ys_arr = ys_dict.get(sp)
        ys_b   = ys_arr[bidx, 0] if ys_arr is not None else None

        # 12 ECG leads, z-scored + offset
        for li in range(12):
            sig = ecg[li]
            sig_n = (sig - sig.mean()) / (sig.std() + 1e-8)
            ax.plot(t, sig_n + li * offset_step, color=cmap[li], lw=0.6, alpha=0.75)

        # stimulus channel if present
        stim = ecg[12]
        if stim.any():
            ax.plot(t, stim / (stim.max() + 1e-8) * 1.2 + 12 * offset_step,
                    color='crimson', lw=0.8, alpha=0.9)

        # GT QRS mask as background shading (twinx to not disturb lead scale)
        ax2 = ax.twinx()
        if ys_b is not None:
            ax2.fill_between(t, 0, ys_b, color='seagreen', alpha=0.2)
        ax2.set_ylim(-0.1, 1.5);  ax2.set_yticks([])

        # anchor and next-beat marker
        ax.axvline(int(row['window_pre']), color='red',    lw=0.8, ls='--', alpha=0.6)
        fop = row['first_overlap_pos']
        if not np.isnan(fop):
            ax.axvline(fop, color='orange', lw=0.8, ls=':', alpha=0.8)

        err_str = f'  err={row["err_ms"]:.0f}ms' if not np.isnan(row['err_ms']) else ''
        ax.set_title(
            f'{row["source"]} @{row["spike_idx"]}  '
            f'n_in_win={int(row["n_beats_in_window"])}{err_str}',
            fontsize=7)
        ax.set_xticks([]);  ax.set_yticks([])
        ax.grid(alpha=0.1)

    for ax in axes[len(sub):]:
        ax.axis('off')
    plt.tight_layout()
    path = os.path.join(out_dir, 'peaks_examples.png')
    plt.savefig(path, dpi=150);  plt.close()
    print(f'  saved {path}')


# ── PT pipeline debug plot ───────────────────────────────────────────────────

def plot_pt_debug(ecg_path: str, out_dir: str,
                  T0: int = 16800, T1: int = 18000,
                  min_distance_ms: int = 150):
    leads, _ = load_ecg(ecg_path)
    combined, delay, filt_ref, peaks_raw, thr2, refined = _pan_tompkins_detect(
        leads, min_distance_ms
    )
    refined = np.array(refined)

    # spikes visible in the window, each gets a distinct colour
    visible_pt      = [p - delay for p in peaks_raw if T0 <= p - delay <= T1]
    visible_refined = [r         for r in refined   if T0 <= r         <= T1]
    cmap             = plt.cm.tab10(np.linspace(0, 0.9, max(len(visible_refined), 1)))
    spike_colors     = {r: cmap[i] for i, r in enumerate(visible_refined)}

    def vlines(ax, positions, ls='--'):
        for pos in positions:
            ax.axvline(pos, color=spike_colors.get(pos, 'grey'), lw=1, ls=ls, alpha=0.8)

    t = np.arange(T0, T1)
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    rec = os.path.basename(os.path.dirname(ecg_path))
    fig.suptitle(f'{rec} — combined PT pipeline (min_dist={min_distance_ms}ms)', fontsize=11)

    axes[0].plot(t, leads['II'][T0:T1], color='steelblue', lw=0.7)
    axes[0].set_ylabel('Raw Lead II')
    vlines(axes[0], visible_refined)

    axes[1].plot(t, filt_ref[T0:T1], color='purple', lw=0.7)
    axes[1].set_ylabel('Bandpass filtered\nLead II')
    vlines(axes[1], visible_refined)

    t_shifted = t - delay
    axes[2].plot(t_shifted, combined[T0:T1], color='darkorange', lw=0.8,
                 label='combined PT (12-lead)')
    axes[2].axhline(thr2, color='grey', lw=0.9, ls='--', label=f'thr2={thr2:.3f}')
    for pc in visible_pt:
        axes[2].axvline(pc, color='steelblue', lw=1, alpha=0.6, ls=':')
    axes[2].set_ylabel('Combined PT\n(delay-corrected)')
    axes[2].legend(fontsize=7, loc='upper left')
    vlines(axes[2], visible_refined)

    axes[3].plot(t, filt_ref[T0:T1], color='dimgray', lw=0.7)
    for r in visible_refined:
        axes[3].axvline(r, color=spike_colors[r], lw=1.2, alpha=0.9,
                        label=f'spike @{r}')
    axes[3].set_ylabel('Refined spikes\n(Lead II filtered)')
    axes[3].set_xlabel('time (ms)')
    axes[3].legend(fontsize=7, loc='upper left')

    plt.tight_layout()
    path = os.path.join(out_dir, f'pt_debug_{rec}_{T0}.png')
    plt.savefig(path, dpi=150);  plt.close()
    print(f'  saved {path}')


# ── main ─────────────────────────────────────────────────────────────────────

def _beats_to_meta(ann_beats, unann_beats):
    """Build meta dict from Beat objects (no cache)."""
    def _arr(beats, attr, default=None):
        vals = [getattr(b, attr) for b in beats]
        if default is not None:
            vals = [v if v is not None else default for v in vals]
        return np.array(vals)

    all_spikes_by_source: dict = {}
    for b in ann_beats + unann_beats:
        all_spikes_by_source.setdefault(b.source, []).append(b.spike_idx)
    all_spikes_by_source = {k: np.array(sorted(v)) for k, v in all_spikes_by_source.items()}

    return dict(
        spike_idx            = _arr(ann_beats,  'spike_idx').astype(np.int32),
        window_pre           = _arr(ann_beats,  'window_pre').astype(np.int32),
        source               = _arr(ann_beats,  'source'),
        qrs_duration         = _arr(ann_beats,  'qrs_duration', np.nan).astype(np.float32),
        unann_spike_idx      = _arr(unann_beats, 'spike_idx').astype(np.int32),
        unann_window_pre     = _arr(unann_beats, 'window_pre').astype(np.int32),
        unann_source         = _arr(unann_beats, 'source'),
        all_spikes_by_source = all_spikes_by_source,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     default='data')
    parser.add_argument('--cache_dir',    default='cache',
                        help='only used for --inf emb cache lookup')
    parser.add_argument('--out_dir',      default='results/peak_analysis')
    parser.add_argument('--holdout_pat',  default='p9',
                        help='folder name prefix for holdout patient(s)')
    parser.add_argument('--inf',          default=None,
                        help='path to head checkpoint for per-beat inference (uses emb cache)')
    parser.add_argument('--width',        type=int, default=256)
    parser.add_argument('--batch_size',   type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── discover ECG folders ──────────────────────────────────────────────────
    all_folders = sorted(
        f for f in glob.glob(os.path.join(args.data_dir, '*'))
        if os.path.isfile(os.path.join(f, 'ecg_data.txt'))
    )
    ho_folders = [f for f in all_folders if os.path.basename(f).startswith(args.holdout_pat)]
    tr_folders = [f for f in all_folders if f not in ho_folders]
    print(f'Train folders  : {[os.path.basename(f) for f in tr_folders]}')
    print(f'Holdout folders: {[os.path.basename(f) for f in ho_folders]}')

    # ── process from scratch ──────────────────────────────────────────────────
    print('Processing train...')
    tr_ann, tr_unann, _ = load_patient_beats(tr_folders)
    print('Processing holdout...')
    ho_ann, ho_unann, _ = load_patient_beats(ho_folders)

    tr_n    = len(tr_ann)
    ho_n    = len(ho_ann)
    un_n    = len(tr_unann)
    tr_meta = _beats_to_meta(tr_ann, tr_unann)
    ho_meta = _beats_to_meta(ho_ann, ho_unann)
    tr_al   = np.stack([b.window for b in tr_ann]) if tr_ann else None
    ho_al   = np.stack([b.window for b in ho_ann]) if ho_ann else None

    print(f'  annotated train   : {tr_n} beats')
    print(f'  annotated holdout : {ho_n} beats')
    print(f'  unannotated train : {un_n} beats')

    un_meta = dict(
        spike_idx            = tr_meta['unann_spike_idx'],
        window_pre           = tr_meta['unann_window_pre'],
        source               = tr_meta['unann_source'],
        all_spikes_by_source = tr_meta['all_spikes_by_source'],
    ) if un_n else None

    # ── overlap detection ─────────────────────────────────────────────────────
    print('Computing window overlaps...')
    tr_ov = compute_overlaps(tr_meta['spike_idx'], tr_meta['window_pre'],
                             tr_meta['source'],    tr_meta['all_spikes_by_source'])
    ho_ov = compute_overlaps(ho_meta['spike_idx'], ho_meta['window_pre'],
                             ho_meta['source'],    ho_meta['all_spikes_by_source'])
    un_ov = (compute_overlaps(un_meta['spike_idx'], un_meta['window_pre'],
                              un_meta['source'],    un_meta['all_spikes_by_source'])
             if un_meta else None)

    # ── optional inference (emb cache required) ───────────────────────────────
    tr_errs = ho_errs = None
    if args.inf:
        from model import MaskHead

        def find_emb_cache(pat):
            paths = sorted(glob.glob(os.path.join(args.cache_dir, f'emb_{pat}_*_decisions.npy')))
            non_aug = [p for p in paths if 'aug' not in os.path.basename(p)]
            return (non_aug[0] if non_aug else paths[0]) if paths else None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        head = MaskHead(embed_dim=768, window_size=WINDOW_SIZE, width=args.width).to(device)
        head.load_state_dict(torch.load(args.inf, map_location=device))
        print(f'Loaded head from {args.inf}')

        tr_dec_path = find_emb_cache(args.holdout_pat + '_inv')  # train key = not holdout
        ho_dec_path = find_emb_cache(args.holdout_pat)
        if tr_dec_path is None or ho_dec_path is None:
            raise FileNotFoundError('Emb cache not found; run embed first.')
        tr_emb_path = tr_dec_path.replace('_decisions.npy', '_embs.npy')
        ho_emb_path = ho_dec_path.replace('_decisions.npy', '_embs.npy')

        tr_emb_dec = np.load(tr_dec_path)
        ho_emb_dec = np.load(ho_dec_path)
        tr_dec_wins = np.stack([b.decision_window for b in tr_ann if b.decision_window is not None])
        ho_dec_wins = np.stack([b.decision_window for b in ho_ann if b.decision_window is not None])
        tr_kept = align_beats_to_emb(tr_dec_wins, tr_emb_dec)
        ho_kept = align_beats_to_emb(ho_dec_wins, ho_emb_dec)

        dummy_ys = lambda n: np.zeros((n, 1, WINDOW_SIZE))
        tr_errs_all = infer_errors(head, tr_emb_path, tr_dec_path, dummy_ys(len(tr_emb_dec)), device, args.batch_size)
        ho_errs_all = infer_errors(head, ho_emb_path, ho_dec_path, dummy_ys(len(ho_emb_dec)), device, args.batch_size)
        tr_errs = np.full(tr_n, np.nan); tr_errs[tr_kept] = tr_errs_all
        ho_errs = np.full(ho_n, np.nan); ho_errs[ho_kept] = ho_errs_all
        print(f'  train mean err={np.nanmean(tr_errs):.1f}ms  holdout mean err={np.nanmean(ho_errs):.1f}ms')

    # ── build dataframe ───────────────────────────────────────────────────────
    def make_df(split, n, meta, ov, errs, subsplit=None):
        src = meta.get('source', np.full(n, ''))
        rows = dict(
            split=split,
            subsplit=subsplit or split,
            beat_idx=np.arange(n),
            source=[os.path.basename(os.path.dirname(s)) for s in src],
            spike_idx=meta.get('spike_idx', np.full(n, -1)).astype(int),
            qrs_gt_ms=meta.get('qrs_duration', np.full(n, np.nan)).astype(float),
            window_pre=meta.get('window_pre', np.full(n, 150)).astype(int),
            overlaps_any=ov['overlaps_any'],
            n_beats_in_window=ov['n_beats_in_window'],
            first_overlap_pos=ov['first_overlap_pos'],
            err_ms=errs if errs is not None else np.full(n, np.nan),
        )
        return pd.DataFrame(rows)

    frames = [
        make_df('annotated', tr_n, tr_meta, tr_ov, tr_errs, subsplit='train'),
        make_df('annotated', ho_n, ho_meta, ho_ov, ho_errs, subsplit='holdout'),
    ]
    if un_meta is not None:
        frames.append(make_df('unannotated', un_n, un_meta, un_ov, None, subsplit='unann'))
    df = pd.concat(frames, ignore_index=True)

    csv_path = os.path.join(args.out_dir, 'peaks.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved {csv_path}  ({len(df)} beats)')

    # ── summary ───────────────────────────────────────────────────────────────
    print('\n── Overlap rate ─────────────────────────────────────────────────')
    print(df.groupby('split').agg(
        overlap_rate=('overlaps_any','mean'),
        mean_n_beats=('n_beats_in_window','mean'),
    ).round(3).to_string())

    print('\n── n_beats_in_window distribution ──────────────────────────────')
    print(df.groupby(['split','n_beats_in_window']).size().rename('count').to_string())

    if df['err_ms'].notna().any():
        print('\n── Error by overlap presence ─────────────────────────────────')
        print(df.groupby(['split','overlaps_any'])['err_ms'].agg(['median','mean','count']).round(1).to_string())

    # ── plots ─────────────────────────────────────────────────────────────────
    print('\nPlotting...')
    plot_summary(df, args.out_dir)
    plot_pt_debug('data/p19_1/ecg_data.txt', args.out_dir, T0=16800,  T1=18000)
    plot_pt_debug('data/p19_1/ecg_data.txt', args.out_dir, T0=890462, T1=891662)
    plot_pt_debug('data/p19_1/ecg_data.txt', args.out_dir, T0=570619, T1=571819)
    # p7 dense-cluster regions (annotated beats @ 711815 and @ 718790, n_in_win=4)
    plot_pt_debug('data/p7/ecg_data.txt', args.out_dir, T0=710000, T1=714000)
    plot_pt_debug('data/p7/ecg_data.txt', args.out_dir, T0=717800, T1=720200)
    plot_examples(df,
                  {'train': tr_al, 'holdout': ho_al},
                  {'train': None, 'holdout': None},
                  args.out_dir)

    msg = '\nDone.' if args.inf else '\nDone. Add --inf ckpts/head_best_val.pt to populate err_ms.'
    print(msg)


if __name__ == '__main__':
    main()
