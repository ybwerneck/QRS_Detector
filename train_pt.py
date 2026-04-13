"""Training script — Pan-Tompkins baseline head (no HuBERT encoder).

PTHead operates purely on the z-scored PT decision signal. No embeddings
are computed or cached. Decisions and targets are loaded from the existing
embedding cache if present (shared format with train.py), or recomputed
from beat pickles.
"""

import os
import sys
import glob
import pickle
import subprocess
import tempfile
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from beat import load_or_process_beats
from dataset import BeatDataset, preprocess_hubert
from model import PTHead


# =========================================================
# Non-blocking plot dispatch
# =========================================================

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_plot_worker.py')
_plot_procs: list = []


def _prune_procs():
    global _plot_procs
    _plot_procs = [p for p in _plot_procs if p.poll() is None]


def dispatch_debug_plot(**kwargs):
    _prune_procs()
    fd, tmp_path = tempfile.mkstemp(suffix='.pkl')
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(kwargs, f)
    proc = subprocess.Popen([sys.executable, _WORKER, tmp_path])
    _plot_procs.append(proc)
    tqdm.write(f'  [plot] dispatched epoch {kwargs.get("epoch", 0)} → {kwargs.get("out_dir", "debug")}')


# =========================================================
# Data helpers
# =========================================================

def _cache_path(cache_dir, folders, tag):
    key = '_'.join(sorted(os.path.basename(f) for f in folders))
    return os.path.join(cache_dir, f'emb_{tag}_{key}')


def _cache_exists(base):
    return all(os.path.exists(f'{base}_{s}.npy') for s in ('decisions', 'ys'))


def _ys_valid(base):
    path = f'{base}_ys.npy'
    if not os.path.exists(path):
        return False
    return np.load(path, mmap_mode='r').ndim == 3


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


def load_or_precompute(dataset, batch_size, cache_path, force=False, desc=''):
    lead2_path = f'{cache_path}_lead2.npy'

    if not force and _cache_exists(cache_path) and _ys_valid(cache_path):
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
# Train / eval
# =========================================================

def run_epoch(head, loader, optimizer, device, train=True):
    head.train(train)
    total_bce = total_qrs = n = 0

    with torch.set_grad_enabled(train):
        for emb, d, y_mask in loader:
            # emb is a dummy zero tensor — PTHead ignores it
            emb    = emb.to(device, non_blocking=True)
            d      = d.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            logits, mask, durations = head(emb, d)
            tv   = (logits[:, :, 1:] - logits[:, :, :-1]).abs().mean()
            loss = F.binary_cross_entropy_with_logits(logits, y_mask[:, 0:1, :]) + tv

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()

            with torch.no_grad():
                true_dur = y_mask.sum(dim=-1)
                hard_dur = (mask > 0.5).float().sum(dim=-1)
                qrs_mae  = (hard_dur[:, 0] - true_dur[:, 0]).abs().mean().item()

            B = d.size(0)
            total_bce += loss.item() * B
            total_qrs += qrs_mae     * B
            n         += B

    return total_bce / n, total_qrs / n, 0.0


@torch.no_grad()
def collect_predictions(head, loader, device):
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
    head.eval()
    _, mask, _ = head(emb.to(device), decision.to(device))
    hard_dur = (mask > 0.5).float().sum(dim=-1)
    gt_dur   = y_mask[:, 0, :].float().sum(dim=-1)
    err      = (hard_dur[:, 0] - gt_dur.to(device)).abs()
    return err.argmin().item(), err.argmax().item()


@torch.no_grad()
def collect_sample_logits(head, emb, decision, y_mask, lead2, device):
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


# =========================================================
# Main
# =========================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    all_folders     = sorted(glob.glob(os.path.join(args.data_dir, 'p*')))
    holdout_pat     = args.holdout_patient
    train_folders   = [f for f in all_folders
                       if not os.path.basename(f).startswith(holdout_pat)]
    holdout_folders = [f for f in all_folders
                       if os.path.basename(f).startswith(holdout_pat)]

    print(f'Train folders   : {[os.path.basename(f) for f in train_folders]}')
    print(f'Holdout folders : {[os.path.basename(f) for f in holdout_folders]}')

    train_cp   = _cache_path(args.cache_dir, train_folders,   'train')
    holdout_cp = _cache_path(args.cache_dir, holdout_folders, 'holdout')

    need_beats = (args.force
                  or not (_cache_exists(train_cp) and _cache_exists(holdout_cp))
                  or not (_ys_valid(train_cp) and _ys_valid(holdout_cp)))

    if need_beats:
        print('Loading / processing beats...')
        train_ann, _, _   = load_or_process_beats(train_folders,   args.cache_dir, args.force)
        holdout_ann, _, _ = load_or_process_beats(holdout_folders, args.cache_dir, args.force)
        train_ds_full = BeatDataset(train_ann,   transform=preprocess_hubert)
        holdout_ds    = BeatDataset(holdout_ann, transform=preprocess_hubert)
    else:
        print('  [cache] decisions found — skipping beat load')
        train_ds_full = holdout_ds = None

    decisions,    ys,    lead2    = load_or_precompute(
        train_ds_full, args.batch_size, train_cp,   args.force, '  train  ')
    ho_decisions, ho_ys, ho_lead2 = load_or_precompute(
        holdout_ds,    args.batch_size, holdout_cp, args.force, '  holdout')

    # PTHead ignores embeddings — pass dummy zeros
    dummy    = torch.zeros(len(decisions),    1, 1, 1)
    ho_dummy = torch.zeros(len(ho_decisions), 1, 1, 1)

    full_ds = TensorDataset(dummy, decisions, ys)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    holdout_full_ds = TensorDataset(ho_dummy, ho_decisions, ho_ys)
    print(f'Train / Val / Holdout ({holdout_pat}) : {n_train} / {n_val} / {len(holdout_full_ds)}')

    train_dl   = DataLoader(train_ds,        batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    val_dl     = DataLoader(val_ds,          batch_size=args.batch_size, shuffle=False, pin_memory=True)
    holdout_dl = DataLoader(holdout_full_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    head = PTHead().to(device)

    optimizer    = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
    warmup_epochs = max(1, args.epochs // 10)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr * 1e-2),
        ],
        milestones=[warmup_epochs],
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)
    debug_dir = args.plots_dir if args.plots_dir else os.path.join(args.ckpt_dir, 'debug')
    if os.path.isdir(debug_dir):
        import shutil
        try:
            shutil.rmtree(debug_dir)
        except OSError:
            pass

    best_val = float('inf')
    best_ho  = float('inf')
    history  = []

    bar = tqdm(range(1, args.epochs + 1), desc='training', unit='ep')
    for epoch in bar:
        tr_bce, tr_qrs, tr_qt = run_epoch(head, train_dl,   optimizer, device, train=True)
        va_bce, va_qrs, va_qt = run_epoch(head, val_dl,     optimizer, device, train=False)
        ho_bce, ho_qrs, ho_qt = run_epoch(head, holdout_dl, optimizer, device, train=False)
        scheduler.step()

        best_val = min(best_val, va_qrs + va_qt)
        best_ho  = min(best_ho,  ho_qrs + ho_qt)
        history.append(dict(
            epoch=epoch,
            tr_bce=tr_bce, va_bce=va_bce, ho_bce=ho_bce,
            tr_qrs=tr_qrs, va_qrs=va_qrs, ho_qrs=ho_qrs,
            tr_qt=tr_qt,   va_qt=va_qt,   ho_qt=ho_qt,
        ))

        bar.set_postfix(
            tr=f'{tr_qrs:.1f}/{tr_qt:.1f}',
            va=f'{va_qrs:.1f}/{va_qt:.1f}',
            ho=f'{ho_qrs:.1f}/{ho_qt:.1f}',
            lr=f'{scheduler.get_last_lr()[0]:.2e}',
        )

        if args.plot_every > 0 and (epoch % args.plot_every == 0 or epoch == args.epochs):
            plot_data = {
                'train':   collect_predictions(head, train_dl,   device),
                'val':     collect_predictions(head, val_dl,     device),
                'holdout': collect_predictions(head, holdout_dl, device),
            }
            tr_idx = torch.tensor(train_ds.indices)
            va_idx = torch.tensor(val_ds.indices)
            _splits = [
                ('train',   dummy[tr_idx], decisions[tr_idx], ys[tr_idx],
                 lead2[tr_idx] if lead2 is not None else None),
                ('val',     dummy[va_idx], decisions[va_idx], ys[va_idx],
                 lead2[va_idx] if lead2 is not None else None),
                ('holdout', ho_dummy, ho_decisions, ho_ys, ho_lead2),
            ]
            _parts, _labels = [], []
            for _name, _e, _d, _y, _l2 in _splits:
                _bi, _wi = pick_best_worst(head, _e, _d, _y, device)
                for _i, _tag in ((_bi, 'best'), (_wi, 'worst')):
                    _l2_sel = _l2[[_i]] if _l2 is not None else None
                    _parts.append(collect_sample_logits(
                        head, _e[[_i]], _d[[_i]], _y[[_i]], _l2_sel, device))
                    _labels.append(f'{_name} ({_tag})')
            _keys = ['logits', 'mask', 'f_sig', 'g_sig', 'decision', 'y_mask']
            sample_data = {k: np.concatenate([p[k] for p in _parts], axis=0) for k in _keys}
            _l2_parts = [p['lead2'] for p in _parts]
            sample_data['lead2'] = (
                np.concatenate(_l2_parts, axis=0)
                if all(x is not None for x in _l2_parts) else None
            )
            sample_data['labels'] = _labels
            dispatch_debug_plot(
                epoch=epoch,
                history=history,
                plot_data=plot_data,
                sample_data=sample_data,
                out_dir=debug_dir,
            )

        if epoch % args.log_every == 0 or epoch == args.epochs:
            tqdm.write(
                f'[{epoch:04d}]  '
                f'train bce={tr_bce:.4f}  qrs={tr_qrs:.1f}  qt={tr_qt:.1f}  '
                f'val qrs={va_qrs:.1f}  qt={va_qt:.1f}  '
                f'holdout({holdout_pat}) qrs={ho_qrs:.1f}  qt={ho_qt:.1f}  '
                f'lr={scheduler.get_last_lr()[0]:.2e}'
            )

    print(f'\nbest val MAE(qrs+qt)={best_val:.2f} ms   '
          f'best holdout({holdout_pat})={best_ho:.2f} ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',        default='data')
    parser.add_argument('--ckpt_dir',        default='ckpts_pt')
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--batch_size',      type=int,   default=16)
    parser.add_argument('--lr',              type=float, default=1e-4)
    parser.add_argument('--wd',              type=float, default=1e-4)
    parser.add_argument('--val_split',       type=float, default=0.2)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--holdout_patient', type=str,   default='p9')
    parser.add_argument('--log_every',       type=int,   default=10)
    parser.add_argument('--plot_every',      type=int,   default=10)
    parser.add_argument('--plots_dir',       default=None,
                        help='output folder for debug plots (default: <ckpt_dir>/debug)')
    parser.add_argument('--cache_dir',       default='cache')
    parser.add_argument('--force',           action='store_true',
                        help='recompute and overwrite all caches')
    args = parser.parse_args()
    main(args)
