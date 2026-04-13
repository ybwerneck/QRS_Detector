"""Training script — HuBERT-ECG + MaskHead for QRS/QT interval prediction.

Since the encoder is frozen, embeddings are precomputed once and stored on CPU.
Each training batch is moved to GPU just-in-time.

Output masks: (N, 2, W) float in [0, 1]  where W = WINDOW_PRE + WINDOW_POST.
  mask[:, 0, :].sum(-1) ≈ QRS duration in ms
  mask[:, 1, :].sum(-1) ≈ QT  interval in ms

NOTE: embedding caches (cache/*_ys.npy) from the previous scalar-head run
are incompatible — run with --force once to rebuild them.
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

from beat import load_patient_beats, load_or_process_beats
from dataset import BeatDataset, preprocess_hubert
from model import build_model


# =========================================================
# Non-blocking plot dispatch
# =========================================================

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_plot_worker.py')
_plot_procs: list = []


def _prune_procs():
    global _plot_procs
    _plot_procs = [p for p in _plot_procs if p.poll() is None]


def dispatch_debug_plot(**kwargs):
    """Pickle kwargs and spawn a fresh subprocess to render plots."""
    _prune_procs()
    fd, tmp_path = tempfile.mkstemp(suffix='.pkl')
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(kwargs, f)
    proc = subprocess.Popen([sys.executable, _WORKER, tmp_path])
    _plot_procs.append(proc)
    tqdm.write(f'  [plot] dispatched epoch {kwargs.get("epoch", 0)} → {kwargs.get("out_dir", "debug")}')


# =========================================================
# Helpers
# =========================================================

@torch.no_grad()
def precompute_embeddings(model, dataset, batch_size, device, desc=''):
    """Run the frozen encoder once; keep embeddings + decision windows on CPU."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    embs, decisions, ys = [], [], []
    model.eval()

    for x, d, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        embs.append(model.encode(x).cpu())
        decisions.append(d)
        ys.append(y)

    # lead II in the 550ms decision window — index 2 in the canonical lead order
    # (VD d=0, I=1, II=2); window shape (n_leads, 550)
    lead2 = torch.from_numpy(
        np.stack([b.window[2, :].astype(np.float32) for b in dataset.beats])
    )  # (N, 550)

    return torch.cat(embs), torch.cat(decisions), torch.cat(ys), lead2


@torch.no_grad()
def precompute_decisions(dataset, batch_size, desc=''):
    """Collect decision windows + ys without running the encoder (PT baseline)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    decisions, ys = [], []
    for _, d, y in tqdm(loader, desc=desc, leave=False):
        decisions.append(d)
        ys.append(y)
    lead2 = torch.from_numpy(
        np.stack([b.window[2, :].astype(np.float32) for b in dataset.beats])
    )
    dec = torch.cat(decisions)
    dummy_embs = torch.zeros(len(dec), 1, 1, 1)   # PTHead ignores embs
    return dummy_embs, dec, torch.cat(ys), lead2


class _ModelWithHead:
    """Head-only model used when the encoder output is fully cached."""
    def __init__(self, head):
        self.head = head


def load_or_build_model(cache_dir, force, device, width,
                        train_folders, holdout_folders):
    train_cp   = emb_cache_path(cache_dir, train_folders,   'train')
    holdout_cp = emb_cache_path(cache_dir, holdout_folders, 'holdout')
    cfg_path   = os.path.join(cache_dir, 'model_config.pt')

    all_cached = (
        not force
        and _emb_cache_exists(train_cp)
        and _emb_cache_exists(holdout_cp)
        and os.path.exists(cfg_path)
    )

    if all_cached:
        cfg = torch.load(cfg_path, map_location='cpu', weights_only=True)
        print(f'  [cache] skipping HuBERT load — building head from {cfg_path}')
        from beat import WINDOW_PRE, WINDOW_POST
        from model import MaskHead
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


def emb_cache_path(cache_dir, folders, tag):
    key = '_'.join(sorted(os.path.basename(f) for f in folders))
    return os.path.join(cache_dir, f'emb_{tag}_{key}')


def _emb_cache_exists(base):
    return all(os.path.exists(f'{base}_{s}.npy') for s in ('embs', 'decisions', 'ys'))


def _ys_valid(base):
    """True if the ys cache exists and has the mask shape (N, 2, W)."""
    path = f'{base}_ys.npy'
    if not os.path.exists(path):
        return False
    return np.load(path, mmap_mode='r').ndim == 3


def load_or_precompute(model, dataset, batch_size, device,
                       cache_path, force=False, desc=''):
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
        print(f'  [cache] stale ys {ys_raw.shape} — rebuilding from dataset (no re-encoding)')
        embs      = torch.from_numpy(np.load(f'{cache_path}_embs.npy'))
        decisions = torch.from_numpy(np.load(f'{cache_path}_decisions.npy'))
        ys = torch.stack([dataset[i][2] for i in range(len(dataset))])
        np.save(f'{cache_path}_ys.npy', ys.numpy())
        print(f'  [cache] saved new ys {tuple(ys.shape)}')
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
    print(f'  [cache] saved  {cache_path}_*.npy')
    return embs, decisions, ys, lead2


# =========================================================
# Train / eval
# =========================================================

def run_epoch(head, loader, optimizer, device, train=True, scaler=None, lambda_tv=1):
    head.train(train)

    total_bce = total_qrs = n = 0

    with torch.set_grad_enabled(train):
        for emb, d, y_mask in loader:
            emb    = emb.to(device, non_blocking=True)
            d      = d.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)   # (N, 2, 550)

            if train and scaler is not None:
                with torch.autocast(device_type=device.type):
                    logits, mask, durations = head(emb, d)
                    tv   = (logits[:, :, 1:] - logits[:, :, :-1]).abs().mean()
                    loss = F.binary_cross_entropy_with_logits(logits, y_mask[:, 0:1, :]) + 1 * tv
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, mask, durations = head(emb, d)
           
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    y_mask[:, 0:1, :],
                    reduction='sum'
                )
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()

            with torch.no_grad():
                true_dur  = y_mask.sum(dim=-1)                        # (N, 2)
                hard_dur  = (mask > 0.5).float().sum(dim=-1)          # (N, 1)
                qrs_mae   = (hard_dur[:, 0] - true_dur[:, 0]).abs().mean().item()
                qt_mae    = 0.0

            B = emb.size(0)
            total_bce += loss.item() * B
            total_qrs += qrs_mae     * B
            n         += B

    return total_bce / n, total_qrs / n, 0.0


@torch.no_grad()
def collect_predictions(head, loader, device):
    """Return (preds_ms, targets_ms) as numpy arrays of shape (N, 2).
    QT column is zeroed while the head only predicts QRS."""
    head.eval()
    preds, targets = [], []
    for emb, d, y_mask in loader:
        emb    = emb.to(device, non_blocking=True)
        d      = d.to(device, non_blocking=True)
        y_mask = y_mask.to(device, non_blocking=True)
        _, mask, _ = head(emb, d)
        hard_dur = (mask > 0.5).float().sum(dim=-1)               # (N, 1)
        pad = torch.zeros_like(hard_dur)
        preds.append(torch.cat([hard_dur, pad], dim=-1).cpu())    # (N, 2)
        targets.append(y_mask.sum(dim=-1).cpu())                  # (N, 2)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy()


@torch.no_grad()
def pick_best_worst(head, emb, decision, y_mask, device):
    """Return (best_idx, worst_idx) by absolute QRS duration error."""
    head.eval()
    _, mask, _ = head(emb.to(device), decision.to(device))
    hard_dur = (mask > 0.5).float().sum(dim=-1)                   # (N, 1)
    gt_dur   = y_mask[:, 0, :].float().sum(dim=-1)                # (N,)
    err      = (hard_dur[:, 0] - gt_dur.to(device)).abs()
    return err.argmin().item(), err.argmax().item()


@torch.no_grad()
def collect_sample_logits(head, emb, decision, y_mask, lead2, device):
    """Run forward_debug on pre-selected samples (e.g. holdout[:2]).

    Returns a dict of numpy arrays for the debug logit plot.
    lead2 may be None if unavailable.
    """
    head.eval()
    emb      = emb.to(device)
    decision = decision.to(device)
    logits, mask, _, f_sig, g_sig = head.forward_debug(emb, decision)
    return {
        'logits':   logits.cpu().numpy(),          # (N, 1, 550)
        'mask':     mask.cpu().numpy(),            # (N, 1, 550)
        'f_sig':    f_sig.cpu().numpy(),           # (N, 1, 550)
        'g_sig':    g_sig.cpu().numpy(),           # (N, 1, 550)
        'decision': decision.cpu().numpy(),        # (N, 550)
        'y_mask':   y_mask.numpy(),                # (N, 2, 550)
        'lead2':    lead2.numpy() if lead2 is not None else None,  # (N, 550) or None
    }


# =========================================================
# Main
# =========================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ----------------------------------------------------------
    # Data — split folders by patient
    # ----------------------------------------------------------
    all_folders     = sorted(glob.glob(os.path.join(args.data_dir, 'p*')))
    holdout_pat     = args.holdout_patient
    train_folders   = [f for f in all_folders
                       if not os.path.basename(f).startswith(holdout_pat)]
    holdout_folders = [f for f in all_folders
                       if os.path.basename(f).startswith(holdout_pat)]

    print(f'Train folders   : {[os.path.basename(f) for f in train_folders]}')
    print(f'Holdout folders : {[os.path.basename(f) for f in holdout_folders]}')

    train_cp   = emb_cache_path(args.cache_dir, train_folders,   'train')
    holdout_cp = emb_cache_path(args.cache_dir, holdout_folders, 'holdout')
    need_beats = (args.force
                  or not (_emb_cache_exists(train_cp) and _emb_cache_exists(holdout_cp))
                  or not (_ys_valid(train_cp) and _ys_valid(holdout_cp)))

    if need_beats:
        print('Loading / processing beats...')
        train_ann, _, _   = load_or_process_beats(train_folders,   args.cache_dir, args.force)
        holdout_ann, _, _ = load_or_process_beats(holdout_folders, args.cache_dir, args.force)
        print(f'  train annotated={len(train_ann)}  holdout annotated={len(holdout_ann)}')
        train_ds_full = BeatDataset(train_ann,   transform=preprocess_hubert)
        holdout_ds    = BeatDataset(holdout_ann, transform=preprocess_hubert)
    else:
        print('  [cache] embeddings found — skipping beat load')
        train_ds_full = holdout_ds = None

    # ----------------------------------------------------------
    # Model + precompute embeddings
    # ----------------------------------------------------------
    if args.model == 'pt':
        print('Building PT baseline head...')
        from model import PTHead
        model = _ModelWithHead(PTHead())
        model.head.to(device)

        print('Collecting decision windows (no encoder needed)...')
        if need_beats:
            embs,    decisions, ys,     _lead2   = precompute_decisions(train_ds_full, args.batch_size, '  train  ')
            ho_embs, ho_decisions, ho_ys, ho_lead2 = precompute_decisions(holdout_ds,   args.batch_size, '  holdout')
        else:
            # load decisions+ys from cache; embs unused so use dummies
            decisions  = torch.from_numpy(np.load(f'{train_cp}_decisions.npy'))
            ys         = torch.from_numpy(np.load(f'{train_cp}_ys.npy'))
            embs       = torch.zeros(len(decisions), 1, 1, 1)
            lead2_path = f'{train_cp}_lead2.npy'
            _lead2     = torch.from_numpy(np.load(lead2_path)) if os.path.exists(lead2_path) else None
            ho_decisions = torch.from_numpy(np.load(f'{holdout_cp}_decisions.npy'))
            ho_ys        = torch.from_numpy(np.load(f'{holdout_cp}_ys.npy'))
            ho_embs      = torch.zeros(len(ho_decisions), 1, 1, 1)
            ho_lead2_path = f'{holdout_cp}_lead2.npy'
            ho_lead2      = torch.from_numpy(np.load(ho_lead2_path)) if os.path.exists(ho_lead2_path) else None
    else:
        print('Building model (or loading head from cache)...')
        model = load_or_build_model(
            args.cache_dir, args.force, device, args.width,
            train_folders, holdout_folders,
        )

        print('Precomputing embeddings (or loading from cache)...')
        embs, decisions, ys, _lead2 = load_or_precompute(
            model, train_ds_full, args.batch_size, device,
            cache_path=train_cp, force=args.force, desc='  train  ',
        )
        ho_embs, ho_decisions, ho_ys, ho_lead2 = load_or_precompute(
            model, holdout_ds, args.batch_size, device,
            cache_path=holdout_cp, force=args.force, desc='  holdout',
        )
    print(f'  train={tuple(embs.shape)}  holdout={tuple(ho_embs.shape)}')
    print(f'  mask shape: {tuple(ys.shape)}')

    # lead2 self-heal: if cache predates lead2 support, rebuild from beats pickle
    if ho_lead2 is None:
        print('  [cache] lead2 missing — rebuilding from beats cache...')
        ho_ann_raw, _, _ = load_or_process_beats(holdout_folders, args.cache_dir, force=False)
        filtered = [b for b in ho_ann_raw
                    if b.qrs_duration is not None and b.qt_interval is not None]
        ho_lead2 = torch.from_numpy(
            np.stack([b.window[2, :].astype(np.float32) for b in filtered])
        )
        np.save(f'{holdout_cp}_lead2.npy', ho_lead2.numpy())
        print(f'  [cache] saved {holdout_cp}_lead2.npy  shape={tuple(ho_lead2.shape)}')

    # ----------------------------------------------------------
    # Train / val split
    # ----------------------------------------------------------
    full_ds = TensorDataset(embs, decisions, ys)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    holdout_full_ds = TensorDataset(ho_embs, ho_decisions, ho_ys)
    print(f'Train / Val / Holdout ({holdout_pat}) : {n_train} / {n_val} / {len(holdout_full_ds)}')

    train_dl   = DataLoader(train_ds,        batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    val_dl     = DataLoader(val_ds,          batch_size=args.batch_size, shuffle=False, pin_memory=True)
    holdout_dl = DataLoader(holdout_full_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # ----------------------------------------------------------
    # Optimiser + scheduler
    # ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.wd,
    )
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
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

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
        tr_bce, tr_qrs, tr_qt = run_epoch(model.head, train_dl,   optimizer, device, train=True,  scaler=None)
        va_bce, va_qrs, va_qt = run_epoch(model.head, val_dl,     optimizer, device, train=False)
        ho_bce, ho_qrs, ho_qt = run_epoch(model.head, holdout_dl, optimizer, device, train=False)
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
                'train':   collect_predictions(model.head, train_dl,   device),
                'val':     collect_predictions(model.head, val_dl,     device),
                'holdout': collect_predictions(model.head, holdout_dl, device),
            }
            # best + worst from train, val, holdout (2 each = 6 total)
            tr_idx  = torch.tensor(train_ds.indices)
            va_idx  = torch.tensor(val_ds.indices)
            _splits = [
                ('train',   embs[tr_idx], decisions[tr_idx], ys[tr_idx],
                 _lead2[tr_idx] if _lead2 is not None else None),
                ('val',     embs[va_idx], decisions[va_idx], ys[va_idx],
                 _lead2[va_idx] if _lead2 is not None else None),
                ('holdout', ho_embs, ho_decisions, ho_ys, ho_lead2),
            ]
            _parts  = []
            _labels = []
            for _name, _e, _d, _y, _l2 in _splits:
                _bi, _wi = pick_best_worst(model.head, _e, _d, _y, device)
                for _i, _tag in ((_bi, 'best'), (_wi, 'worst')):
                    _l2_sel = _l2[[_i]] if _l2 is not None else None
                    _parts.append(collect_sample_logits(
                        model.head, _e[[_i]], _d[[_i]], _y[[_i]], _l2_sel, device))
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
    parser.add_argument('--data_dir',         default='data')
    parser.add_argument('--ckpt_dir',         default='ckpts')
    parser.add_argument('--epochs',           type=int,   default=50)
    parser.add_argument('--batch_size',       type=int,   default=16)
    parser.add_argument('--lr',               type=float, default=1e-4)
    parser.add_argument('--wd',               type=float, default=1e-4)
    parser.add_argument('--width',            type=int,   default=256)
    parser.add_argument('--val_split',        type=float, default=0.2)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--holdout_patient',  type=str,   default='p9')
    parser.add_argument('--log_every',        type=int,   default=10)
    parser.add_argument('--plot_every',       type=int,   default=10)
    parser.add_argument('--plots_dir',        default=None,
                        help='output folder for debug plots (default: <ckpt_dir>/debug)')
    parser.add_argument('--cache_dir',        default='cache')
    parser.add_argument('--force',            action='store_true',
                        help='recompute and overwrite all caches')
    parser.add_argument('--model',            default='nn', choices=['nn', 'pt'],
                        help='nn = full HuBERT+MaskHead  |  pt = PT-signal baseline')
    args = parser.parse_args()
    main(args)
