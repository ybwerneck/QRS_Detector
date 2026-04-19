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


WINDOW_SIZE = 550   # WINDOW_PRE + WINDOW_POST


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


def load_beats_meta(beats_dir: str) -> dict:
    """Load annotated + unannotated beat metadata + all spike positions per source."""
    fields = ['qrs_duration', 'window_pre', 'spike_idx']
    out = {}
    for f in fields:
        p = os.path.join(beats_dir, f'ann_{f}.npy')
        if os.path.exists(p):
            out[f] = np.load(p)
    src_path = os.path.join(beats_dir, 'ann_source.json')
    if os.path.exists(src_path):
        out['source'] = np.array(json.load(open(src_path)))

    # unannotated beat arrays (for overlap analysis on the unann split)
    for f in ['window_pre', 'spike_idx']:
        p = os.path.join(beats_dir, f'unann_{f}.npy')
        if os.path.exists(p):
            out[f'unann_{f}'] = np.load(p)
    usrc_path = os.path.join(beats_dir, 'unann_source.json')
    if os.path.exists(usrc_path):
        out['unann_source'] = np.array(json.load(open(usrc_path)))

    # all spikes in the recording (annotated + unannotated) keyed by source
    all_spikes: dict[str, np.ndarray] = {}
    for prefix in ('ann', 'unann'):
        sp_path  = os.path.join(beats_dir, f'{prefix}_spike_idx.npy')
        src_path = os.path.join(beats_dir, f'{prefix}_source.json')
        if os.path.exists(sp_path) and os.path.exists(src_path):
            spikes  = np.load(sp_path)
            sources = json.load(open(src_path))
            for sp, src in zip(spikes, sources):
                all_spikes.setdefault(src, []).append(int(sp))
    out['all_spikes_by_source'] = {k: np.array(sorted(v)) for k, v in all_spikes.items()}
    return out


# ── overlap detection ─────────────────────────────────────────────────────────

def compute_overlaps(spike_idx: np.ndarray,
                     window_pre: np.ndarray,
                     source: np.ndarray,
                     all_spikes_by_source: dict) -> dict:
    """For each annotated beat, count how many OTHER detected spikes fall inside its window.

    Uses the full set of detected spikes (ann + unann) from the beats cache,
    grouped by source recording.  No reliance on the period annotation.

    Returns dict of arrays, all length N:
      n_beats_in_window  int   — spikes inside (spike - window_pre, spike + window_post)
                                 excluding the beat's own spike
      first_overlap_pos  float — window-relative position of nearest such spike; NaN if none
    """
    N          = len(spike_idx)
    window_post = WINDOW_SIZE - window_pre  # (N,)
    n_beats    = np.zeros(N, dtype=np.int32)
    first_pos  = np.full(N, np.nan)

    for i in range(N):
        src   = source[i]
        all_sp = all_spikes_by_source.get(src, np.array([], dtype=np.int64))
        lo    = spike_idx[i] - window_pre[i]
        hi    = spike_idx[i] + window_post[i]
        # spikes strictly inside the window, excluding self
        inside = all_sp[(all_sp > lo) & (all_sp < hi) & (all_sp != spike_idx[i])]
        # sanity: after cache rebuild with min_distance_ms=100, none should be < 100ms from anchor
        too_close = np.sum(np.abs(inside - spike_idx[i]) < 100)
        if too_close:
            print(f'  [warn] beat {i} (src={source[i]}, spike={spike_idx[i]}): '
                  f'{too_close} spike(s) within 100ms of anchor — stale cache?')
        n_beats[i] = len(inside)
        if len(inside):
            # nearest to the anchor
            rel = inside - lo   # position relative to window start
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
        ys_b = ys_dict[sp][bidx, 0]       # (550,) QRS mask

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


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',    default='cache')
    parser.add_argument('--out_dir',      default='results/peak_analysis')
    parser.add_argument('--holdout_key',  default='p9')
    parser.add_argument('--train_key',    default='p19')
    parser.add_argument('--ckpt',         default=None,
                        help='path to head_best_val.pt for per-beat errors')
    parser.add_argument('--width',        type=int,   default=256)
    parser.add_argument('--batch_size',   type=int,   default=64)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── discover caches ───────────────────────────────────────────────────────
    def find_emb_cache(prefix_key):
        paths = sorted(glob.glob(os.path.join(args.cache_dir, f'emb_{prefix_key}_*_decisions.npy')))
        non_aug = [p for p in paths if 'aug' not in os.path.basename(p)]
        return (non_aug[0] if non_aug else paths[0]) if paths else None

    def find_beats_dir(key):
        candidates = sorted(glob.glob(os.path.join(args.cache_dir, f'beats_*{key}*')))
        return candidates[0] if candidates else None

    tr_dec_path   = find_emb_cache('train')
    ho_dec_path   = find_emb_cache('holdout')
    un_dec_path   = find_emb_cache('unann')
    tr_beats_dir  = find_beats_dir(args.train_key)
    ho_beats_dir  = find_beats_dir(args.holdout_key)

    for p in [tr_dec_path, ho_dec_path, tr_beats_dir, ho_beats_dir]:
        if not p:
            raise FileNotFoundError(f'Missing cache: {p}')

    # ── load & align ──────────────────────────────────────────────────────────
    print('Loading and aligning caches...')

    def load_split(dec_path, beats_dir):
        base     = dec_path.replace('_decisions.npy', '')
        emb_dec  = np.load(dec_path)
        ys       = np.load(f'{base}_ys.npy')
        al_path  = f'{base}_all_leads.npy'
        all_leads = np.load(al_path) if os.path.exists(al_path) else None

        meta     = load_beats_meta(beats_dir)
        ann_dec  = np.load(os.path.join(beats_dir, 'ann_decision_window.npy'))
        kept_idx = align_beats_to_emb(ann_dec, emb_dec)

        # slice ann arrays to kept indices; leave unann_* and dict-valued entries intact
        aligned = {k: (v[kept_idx] if isinstance(v, np.ndarray) and not k.startswith('unann_') else v)
                   for k, v in meta.items()}
        return emb_dec, ys, all_leads, aligned, base

    tr_dec, tr_ys, tr_al, tr_meta, tr_base = load_split(tr_dec_path, tr_beats_dir)
    ho_dec, ho_ys, ho_al, ho_meta, ho_base = load_split(ho_dec_path, ho_beats_dir)
    print(f'  annotated train   : {tr_dec.shape[0]} beats')
    print(f'  annotated holdout : {ho_dec.shape[0]} beats')

    if tr_al is None or ho_al is None:
        raise FileNotFoundError('all_leads cache missing')

    # unannotated: align emb_unann decisions to beats-cache unann_decision_window
    un_dec, un_meta = None, None
    if un_dec_path:
        emb_un_dec = np.load(un_dec_path)
        un_beats_dec = np.load(os.path.join(tr_beats_dir, 'unann_decision_window.npy'))
        kept_un = align_beats_to_emb(un_beats_dec, emb_un_dec)
        un_meta = {
            'spike_idx':           tr_meta['unann_spike_idx'][kept_un],
            'window_pre':          tr_meta['unann_window_pre'][kept_un],
            'source':              tr_meta['unann_source'][kept_un],
            'all_spikes_by_source': tr_meta['all_spikes_by_source'],
        }
        un_dec = emb_un_dec
        print(f'  unannotated       : {len(un_dec)} beats')

    # ── overlap detection ─────────────────────────────────────────────────────
    print('Computing window overlaps...')
    tr_ov = compute_overlaps(tr_meta['spike_idx'], tr_meta['window_pre'],
                             tr_meta['source'],    tr_meta['all_spikes_by_source'])
    ho_ov = compute_overlaps(ho_meta['spike_idx'], ho_meta['window_pre'],
                             ho_meta['source'],    ho_meta['all_spikes_by_source'])
    un_ov = (compute_overlaps(un_meta['spike_idx'], un_meta['window_pre'],
                              un_meta['source'],    un_meta['all_spikes_by_source'])
             if un_meta else None)

    # ── optional inference ────────────────────────────────────────────────────
    tr_errs = ho_errs = None
    if args.ckpt:
        from model import MaskHead
        from beat import WINDOW_PRE as WP, WINDOW_POST as WPO
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        head = MaskHead(embed_dim=768, window_size=WP + WPO, width=args.width).to(device)
        head.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f'Loaded head from {args.ckpt}')
        tr_emb = tr_base + '_embs.npy'
        ho_emb = ho_base + '_embs.npy'
        tr_errs = infer_errors(head, tr_emb, tr_dec_path, tr_ys, device, args.batch_size)
        ho_errs = infer_errors(head, ho_emb, ho_dec_path, ho_ys, device, args.batch_size)
        print(f'  train mean err={tr_errs.mean():.1f}ms  holdout mean err={ho_errs.mean():.1f}ms')

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
        make_df('annotated', len(tr_dec), tr_meta, tr_ov, tr_errs, subsplit='train'),
        make_df('annotated', len(ho_dec), ho_meta, ho_ov, ho_errs, subsplit='holdout'),
    ]
    if un_meta is not None:
        frames.append(make_df('unannotated', len(un_dec), un_meta, un_ov, None, subsplit='unann'))
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
    plot_examples(df,
                  {'train': tr_al, 'holdout': ho_al},
                  {'train': tr_ys, 'holdout': ho_ys},
                  args.out_dir)

    msg = '\nDone.' if args.ckpt else '\nDone. Add --ckpt ckpts/head_best_val.pt to populate err_ms.'
    print(msg)


if __name__ == '__main__':
    main()
