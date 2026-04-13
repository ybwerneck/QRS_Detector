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
    """
    _prune_procs()
    fd, tmp_path = tempfile.mkstemp(suffix='.pkl')
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(kwargs, f)
    proc = subprocess.Popen([sys.executable, _WORKER, tmp_path])
    _plot_procs.append(proc)
    tqdm.write(f'  [plot] dispatched epoch {kwargs.get("epoch", 0)} → {kwargs.get("out_dir", "debug")}')


# =========================================================
# Cache path helpers
# =========================================================

def emb_cache_path(cache_dir, folders, tag):
    """Canonical cache path for a set of patient folders + tag (train/holdout/unann)."""
    key = '_'.join(sorted(os.path.basename(f) for f in folders))
    return os.path.join(cache_dir, f'emb_{tag}_{key}')


def _emb_cache_exists(base):
    """True if the full annotated embedding cache (embs + decisions + ys) exists."""
    return all(os.path.exists(f'{base}_{s}.npy') for s in ('embs', 'decisions', 'ys'))


def _unann_cache_exists(base):
    """True if the unannotated embedding cache (embs + decisions, no ys) exists."""
    return all(os.path.exists(f'{base}_{s}.npy') for s in ('embs', 'decisions'))


def _ys_valid(base):
    """True if the ys cache exists and has the mask shape (N, 2, W)."""
    path = f'{base}_ys.npy'
    if not os.path.exists(path):
        return False
    return np.load(path, mmap_mode='r').ndim == 3


# =========================================================
# Model / head loading
# =========================================================

class _ModelWithHead:
    """Thin wrapper so head-only code looks the same as a full HuBERTECGRegressor."""
    def __init__(self, head):
        self.head = head


def load_or_build_model(cache_dir, force, device, width,
                        train_folders, holdout_folders,
                        extra_caches=()):
    """Load HuBERT + build MaskHead, or skip HuBERT if all caches are present.

    Parameters
    ----------
    extra_caches : iterable of cache base paths
        Additional caches that must exist before HuBERT loading is skipped.
        Pass the unannotated cache path here when using train_semi.py.
    """
    from model import MaskHead, build_model
    from beat import WINDOW_PRE, WINDOW_POST

    train_cp   = emb_cache_path(cache_dir, train_folders,   'train')
    holdout_cp = emb_cache_path(cache_dir, holdout_folders, 'holdout')
    cfg_path   = os.path.join(cache_dir, 'model_config.pt')

    all_cached = (
        not force
        and _emb_cache_exists(train_cp)
        and _emb_cache_exists(holdout_cp)
        and os.path.exists(cfg_path)
        and all(_unann_cache_exists(p) for p in extra_caches)
    )

    if all_cached:
        cfg = torch.load(cfg_path, map_location='cpu', weights_only=True)
        print(f'  [cache] skipping HuBERT load — building head from {cfg_path}')
        head = MaskHead(
            embed_dim=cfg['embed_dim'],
            window_size=WINDOW_PRE + WINDOW_POST,
            width=width,
        ).to(device)
        return _ModelWithHead(head)

    model, _ = build_model(device=device, width=width)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(
        dict(embed_dim=model.encoder.config.hidden_size,
             embed_L=model.L,
             embed_t=model.t),
        cfg_path,
    )
    print(f'  [cache] saved model config to {cfg_path}')
    return model


# =========================================================
# Annotated embedding precomputation
# =========================================================

@torch.no_grad()
def precompute_embeddings(model, dataset, batch_size, device, desc=''):
    """Run the frozen encoder once; returns (embs, decisions, ys, lead2) on CPU."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    embs, decisions, ys = [], [], []
    model.eval()
    for x, d, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        embs.append(model.encode(x).cpu())
        decisions.append(d)
        ys.append(y)
    lead2 = torch.from_numpy(
        np.stack([b.window[2, :].astype(np.float32) for b in dataset.beats])
    )
    return torch.cat(embs), torch.cat(decisions), torch.cat(ys), lead2


def load_or_precompute(model, dataset, batch_size, device,
                       cache_path, force=False, desc=''):
    """Load annotated embeddings from cache, or precompute and save."""
    lead2_path = f'{cache_path}_lead2.npy'

    if not force and _emb_cache_exists(cache_path):
        ys_raw = np.load(f'{cache_path}_ys.npy')
        if ys_raw.ndim == 3:
            print(f'  [cache] loading {cache_path}_*.npy')
            embs      = torch.from_numpy(np.load(f'{cache_path}_embs.npy'))
            decisions = torch.from_numpy(np.load(f'{cache_path}_decisions.npy'))
            lead2     = (torch.from_numpy(np.load(lead2_path))
                         if os.path.exists(lead2_path) else None)
            return embs, decisions, torch.from_numpy(ys_raw), lead2

        # embs+decisions are fine; only ys are stale — rebuild without re-encoding
        print(f'  [cache] stale ys {ys_raw.shape} — rebuilding (no re-encoding)')
        embs      = torch.from_numpy(np.load(f'{cache_path}_embs.npy'))
        decisions = torch.from_numpy(np.load(f'{cache_path}_decisions.npy'))
        ys = torch.stack([dataset[i][2] for i in range(len(dataset))])
        np.save(f'{cache_path}_ys.npy', ys.numpy())
        lead2 = (torch.from_numpy(np.load(lead2_path))
                 if os.path.exists(lead2_path) else None)
        return embs, decisions, ys, lead2

    embs, decisions, ys, lead2 = precompute_embeddings(
        model, dataset, batch_size, device, desc=desc)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    np.save(f'{cache_path}_embs.npy',      embs.numpy())
    np.save(f'{cache_path}_decisions.npy', decisions.numpy())
    np.save(f'{cache_path}_ys.npy',        ys.numpy())
    np.save(lead2_path,                    lead2.numpy())
    print(f'  [cache] saved {cache_path}_*.npy')
    return embs, decisions, ys, lead2


# =========================================================
# Unannotated embedding precomputation
# =========================================================

@torch.no_grad()
def precompute_embeddings_unann(model, dataset, batch_size, device, desc=''):
    """Like precompute_embeddings but skips ys (unannotated beats have no labels)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    embs, decisions = [], []
    model.eval()
    for x, d, _y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        embs.append(model.encode(x).cpu())
        decisions.append(d)
    return torch.cat(embs), torch.cat(decisions)


def load_or_precompute_unann(model, dataset, batch_size, device,
                             cache_path, force=False, desc=''):
    """Load unannotated embeddings from cache, or precompute and save."""
    if not force and _unann_cache_exists(cache_path):
        print(f'  [cache] loading {cache_path}_embs/decisions.npy')
        embs      = torch.from_numpy(np.load(f'{cache_path}_embs.npy'))
        decisions = torch.from_numpy(np.load(f'{cache_path}_decisions.npy'))
        return embs, decisions

    embs, decisions = precompute_embeddings_unann(
        model, dataset, batch_size, device, desc=desc)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    np.save(f'{cache_path}_embs.npy',      embs.numpy())
    np.save(f'{cache_path}_decisions.npy', decisions.numpy())
    print(f'  [cache] saved {cache_path}_embs/decisions.npy')
    return embs, decisions


# =========================================================
# PT-only decision precomputation  (no encoder)
# =========================================================

def load_decisions(dataset, batch_size, desc=''):
    """Collect decision windows and targets without running any encoder."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    decisions, ys = [], []
    for _, d, y in tqdm(loader, desc=desc, leave=False):
        decisions.append(d)
        ys.append(y)
    lead2 = torch.from_numpy(
        np.stack([b.window[2, :].astype(np.float32) for b in dataset.beats])
    )
    return torch.cat(decisions), torch.cat(ys), lead2


def load_or_precompute_pt(dataset, batch_size, cache_path, force=False, desc=''):
    """Load decisions+ys from cache (shared format), or compute from dataset.

    Used by train_pt.py — no encoder, so embs are not stored/needed.
    """
    lead2_path = f'{cache_path}_lead2.npy'

    if not force and _emb_cache_exists(cache_path) and _ys_valid(cache_path):
        print(f'  [cache] loading {cache_path}_decisions/ys.npy')
        decisions = torch.from_numpy(np.load(f'{cache_path}_decisions.npy'))
        ys        = torch.from_numpy(np.load(f'{cache_path}_ys.npy'))
        lead2     = (torch.from_numpy(np.load(lead2_path))
                     if os.path.exists(lead2_path) else None)
        return decisions, ys, lead2

    decisions, ys, lead2 = load_decisions(dataset, batch_size, desc=desc)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    np.save(f'{cache_path}_decisions.npy', decisions.numpy())
    np.save(f'{cache_path}_ys.npy',        ys.numpy())
    np.save(lead2_path,                    lead2.numpy())
    print(f'  [cache] saved {cache_path}_decisions/ys/lead2.npy')
    return decisions, ys, lead2


# =========================================================
# Eval helpers
# =========================================================

@torch.no_grad()
def collect_predictions(head, loader, device):
    """Return (preds_ms, targets_ms) as numpy arrays of shape (N, 2).
    QT column is zeroed while the head only predicts QRS.
    """
    head.eval()
    preds, targets = [], []
    for emb, d, y_mask in loader:
        emb    = emb.to(device, non_blocking=True)
        d      = d.to(device, non_blocking=True)
        y_mask = y_mask.to(device, non_blocking=True)
        _, mask, _ = head(emb, d)
        hard_dur = (mask > 0.5).float().sum(dim=-1)
        pad = torch.zeros_like(hard_dur)
        preds.append(torch.cat([hard_dur, pad], dim=-1).cpu())
        targets.append(y_mask.sum(dim=-1).cpu())
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


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
def collect_sample_logits(head, emb, decision, y_mask, lead2, device):
    """Run forward_debug on pre-selected samples; returns dict of numpy arrays."""
    head.eval()
    emb      = emb.to(device)
    decision = decision.to(device)
    logits, mask, _, f_sig, g_sig = head.forward_debug(emb, decision)
    return {
        'logits':   logits.cpu().numpy(),
        'mask':     mask.cpu().numpy(),
        'f_sig':    f_sig.cpu().numpy(),
        'g_sig':    g_sig.cpu().numpy(),
        'decision': decision.cpu().numpy(),
        'y_mask':   y_mask.numpy(),
        'lead2':    lead2.numpy() if lead2 is not None else None,
    }


def build_sample_data(head, splits, device):
    """Collect best+worst sample logits across train/val/holdout splits.

    Parameters
    ----------
    splits : list of (name, embs, decisions, ys, lead2)

    Returns
    -------
    sample_data : dict of concatenated numpy arrays
    labels      : list of strings  e.g. ['train (best)', 'train (worst)', ...]
    """
    parts, labels = [], []
    for name, embs, decisions, ys, lead2 in splits:
        bi, wi = pick_best_worst(head, embs, decisions, ys, device)
        for idx, tag in ((bi, 'best'), (wi, 'worst')):
            l2_sel = lead2[[idx]] if lead2 is not None else None
            parts.append(collect_sample_logits(
                head, embs[[idx]], decisions[[idx]], ys[[idx]], l2_sel, device))
            labels.append(f'{name} ({tag})')

    keys = ['logits', 'mask', 'f_sig', 'g_sig', 'decision', 'y_mask']
    sample_data = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    l2_parts = [p['lead2'] for p in parts]
    sample_data['lead2'] = (
        np.concatenate(l2_parts, axis=0)
        if all(x is not None for x in l2_parts) else None
    )
    return sample_data, labels


# =========================================================
# Loss helpers
# =========================================================

def tv_loss(logits):
    """Total variation of logits along the time axis — penalises jagged masks."""
    return (logits[:, :, 1:] - logits[:, :, :-1]).abs().mean()
