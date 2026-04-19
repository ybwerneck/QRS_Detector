"""Shared utilities for all training scripts.

Covers:
  - Non-blocking debug plot dispatch
  - Embedding cache path helpers
  - Model / head loading (_ModelWithHead, load_or_build_model)
  - Embedding precomputation and caching (annotated + unannotated)
  - PT-only decision precomputation (no encoder)
  - Eval helpers (collect_predictions, pick_best_worst, collect_sample_logits)
  - Loss helpers (tv)
"""

import os
import sys

def _np_save_atomic(path, arr):
    """Write numpy array atomically (temp file + rename) to avoid partial writes."""
    dirpath = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(suffix='.npy', dir=dirpath)
    os.close(fd)
    np.save(tmp, arr)
    os.replace(tmp, path)


def _load_all_leads(path, expected_n):
    """Load all_leads cache, discarding the file if missing, corrupt, or wrong size."""
    if not os.path.exists(path):
        return None
    try:
        al = torch.from_numpy(np.load(path))
    except Exception as e:
        print(f'  [cache] all_leads unreadable ({e}) — discarding')
        os.remove(path)
        return None
    if al.shape[0] != expected_n:
        print(f'  [cache] all_leads size mismatch ({al.shape[0]} vs {expected_n}) — discarding')
        os.remove(path)
        return None
    return al


def _pad_to_13(window):
    """Pad (n_leads, W) to (13, W) so all beats stack uniformly.
    Leads 0-11: ECG; lead 12: stimulus (zeros if absent).
    """
    w = window.astype(np.float32)
    if w.shape[0] < 13:
        w = np.concatenate([w, np.zeros((13 - w.shape[0], w.shape[1]), dtype=np.float32)], axis=0)
    return w[:13, :]
import pickle
import subprocess
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


# =========================================================
# Non-blocking plot dispatch
# =========================================================

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_plot_worker.py')
_plot_procs: list = []


def _prune_procs():
    global _plot_procs
    _plot_procs = [p for p in _plot_procs if p.poll() is None]


def dispatch_debug_plot(**kwargs):
    """Pickle kwargs and spawn a fresh subprocess to render plots.

    Each invocation starts a clean interpreter so edits to debug_plot.py
    are picked up on the next tick without restarting training.

    Also writes metrics.csv and summary.txt to out_dir for quick cross-run search.
    """
    _prune_procs()

    history  = kwargs.get('history', [])
    out_dir  = kwargs.get('out_dir', 'debug')
    epoch    = kwargs.get('epoch', 0)
    os.makedirs(out_dir, exist_ok=True)

    # ── metrics.csv: one row per epoch, overwritten each tick ─────
    if history:
        csv_path = os.path.join(out_dir, '..', 'metrics.csv')
        csv_path = os.path.normpath(csv_path)
        cols = list(history[0].keys())
        with open(csv_path, 'w') as f:
            f.write(','.join(cols) + '\n')
            for row in history:
                f.write(','.join(str(row.get(c, '')) for c in cols) + '\n')

        # ── summary.txt: best + latest ────────────────────────────
        best_va = min(history, key=lambda r: r.get('va_qrs', float('inf')) + r.get('va_qt', float('inf')))
        best_ho = min(history, key=lambda r: r.get('ho_qrs', float('inf')) + r.get('ho_qt', float('inf')))
        last    = history[-1]
        lines = [
            f"run_dir : {os.path.normpath(os.path.join(out_dir, '..'))}",
            f"epoch   : {epoch}",
            f"latest  : tr_qrs={last.get('tr_qrs',''):.1f}  tr_qt={last.get('tr_qt',''):.1f}"
            f"  va_qrs={last.get('va_qrs',''):.1f}  va_qt={last.get('va_qt',''):.1f}"
            f"  ho_qrs={last.get('ho_qrs',''):.1f}  ho_qt={last.get('ho_qt',''):.1f}",
            f"best_va : epoch={best_va['epoch']}  va_qrs={best_va.get('va_qrs',''):.1f}  va_qt={best_va.get('va_qt',''):.1f}"
            f"  (ho_qrs={best_va.get('ho_qrs',''):.1f}  ho_qt={best_va.get('ho_qt',''):.1f})",
            f"best_ho : epoch={best_ho['epoch']}  ho_qrs={best_ho.get('ho_qrs',''):.1f}  ho_qt={best_ho.get('ho_qt',''):.1f}"
            f"  (va_qrs={best_ho.get('va_qrs',''):.1f}  va_qt={best_ho.get('va_qt',''):.1f})",
        ]
        with open(os.path.join(out_dir, '..', 'summary.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')

    fd, tmp_path = tempfile.mkstemp(suffix='.pkl')
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(kwargs, f)
    proc = subprocess.Popen([sys.executable, _WORKER, tmp_path])
    _plot_procs.append(proc)
    tqdm.write(f'  [plot] dispatched epoch {epoch} → {out_dir}')


# =========================================================
# Cache path helpers
# =========================================================

def emb_cache_path(cache_dir, folders, tag):
    """Canonical cache path for a set of patient folders + tag (train/holdout/unann)."""
    key = '_'.join(sorted(os.path.basename(f) for f in folders))
    return os.path.join(cache_dir, f'emb_{tag}_{key}')


def _embs_exist(base):
    return os.path.exists(f'{base}_embs.npy')


def _emb_cache_exists(base):
    """True if the full annotated embedding cache exists."""
    return (_embs_exist(base)
            and os.path.exists(f'{base}_decisions.npy')
            and os.path.exists(f'{base}_ys.npy'))


def _unann_cache_exists(base):
    """True if the unannotated embedding cache exists."""
    return _embs_exist(base) and os.path.exists(f'{base}_decisions.npy')


def _load_embs(base):
    arr = np.load(f'{base}_embs.npy')   # mmap: avoids 9 GB upfront spike
    return torch.from_numpy(arr)


def _save_embs(base, tensor):
    np.save(f'{base}_embs.npy', tensor.numpy())


def _ys_valid(base):
    """True if the ys cache exists and has the mask shape (N, 2, W)."""
    path = f'{base}_ys.npy'
    if not os.path.exists(path):
        return False
    return np.load(path).ndim == 3


def _decisions_valid(base):
    """True if the decisions cache exists and has the 12-lead shape (N, 12, W)."""
    from beat import PT_N_LEADS
    path = f'{base}_decisions.npy'
    if not os.path.exists(path):
        return False
    arr = np.load(path)
    return arr.ndim == 3 and arr.shape[1] == PT_N_LEADS


# =========================================================
# Cache loaders  (run precompute.py first)
# =========================================================

def load_cache(cache_path):
    """Load precomputed annotated cache.  Exits with a clear message if missing."""
    for suffix in ('_embs.npy', '_decisions.npy', '_ys.npy'):
        if not os.path.exists(f'{cache_path}{suffix}'):
            sys.exit(f'Cache missing: {cache_path}{suffix}\nRun: python precompute.py')
    embs      = _load_embs(cache_path)
    decisions = torch.from_numpy(np.load(f'{cache_path}_decisions.npy'))
    ys        = torch.from_numpy(np.load(f'{cache_path}_ys.npy'))
    all_leads = _load_all_leads(f'{cache_path}_all_leads.npy', embs.shape[0])
    print(f'  [cache] {cache_path}_* ({len(embs)} beats)')
    return embs, decisions, ys, all_leads


def load_cache_unann(cache_path):
    """Load precomputed unannotated cache (embs + decisions only)."""
    for suffix in ('_embs.npy', '_decisions.npy'):
        if not os.path.exists(f'{cache_path}{suffix}'):
            sys.exit(f'Cache missing: {cache_path}{suffix}\nRun: python precompute.py')
    embs      = _load_embs(cache_path)
    decisions = torch.from_numpy(np.load(f'{cache_path}_decisions.npy'))
    print(f'  [cache] {cache_path}_embs/decisions ({len(embs)} beats)')
    return embs, decisions


def load_beat_types(cache_path):
    """Load per-beat type labels (0=original, 1=scale, 2=shift) if available, else None."""
    path = f'{cache_path}_beat_types.npy'
    if not os.path.exists(path):
        return None
    return torch.from_numpy(np.load(path).astype(np.int64))


def load_cache_pt(cache_path):
    """Load decisions + ys only (PT baseline — no embeddings needed)."""
    for suffix in ('_decisions.npy', '_ys.npy'):
        if not os.path.exists(f'{cache_path}{suffix}'):
            sys.exit(f'Cache missing: {cache_path}{suffix}\nRun: python precompute.py')
    decisions = torch.from_numpy(np.load(f'{cache_path}_decisions.npy'))
    ys        = torch.from_numpy(np.load(f'{cache_path}_ys.npy'))
    all_leads = _load_all_leads(f'{cache_path}_all_leads.npy', decisions.shape[0])
    print(f'  [cache] {cache_path}_decisions/ys ({len(decisions)} beats)')
    return decisions, ys, all_leads


# =========================================================
# Eval helpers
# =========================================================

@torch.no_grad()
def pick_best_worst(head, emb, decision, y_mask, device):
    """Return (best_idx, worst_idx) by absolute QRS duration error."""
    head.eval()
    _, mask, _ = head(emb.to(device), decision.to(device))
    hard_dur = (mask > 0.5).float().sum(dim=-1)
    gt_dur   = y_mask[:, 0, :].float().sum(dim=-1)
    err      = (hard_dur[:, 0] - gt_dur.to(device)).abs()
    return err.argmin().item(), err.argmax().item()


@torch.no_grad()
def collect_sample_logits(head, emb, decision, y_mask, all_leads, device):
    """Run forward_debug on pre-selected samples; returns dict of numpy arrays."""
    head.eval()
    emb      = emb.to(device)
    decision = decision.to(device)
    logits, mask, _, f_sig, g_sig = head.forward_debug(emb, decision)
    return {
        'logits':     logits.cpu().numpy(),
        'mask':       mask.cpu().numpy(),
        'f_sig':      f_sig.cpu().numpy(),
        'g_sig':      g_sig.cpu().numpy(),
        'decision':   decision.cpu().numpy(),
        'y_mask':     y_mask.numpy(),
        'all_leads':  all_leads.numpy() if all_leads is not None else None,
    }


# =========================================================
# Loss helpers
# =========================================================

def tv_loss(logits):
    """Total variation of logits along the time axis — penalises jagged masks."""
    return (logits[:, :, 1:] - logits[:, :, :-1]).abs().mean()


def dur_prior_loss(dur, mu, delta):
    """Dead-zone duration prior: zero penalty within ±delta ms of mu, quadratic beyond.

    Parameters
    ----------
    dur   : (N,) predicted soft durations in ms
    mu    : scalar — mean QRS duration from annotated ground truth
    delta : scalar — tolerance half-width in ms (default 100 ms)
    """
    excess = (dur - mu).abs() - delta
    return F.relu(excess).pow(2).mean()
