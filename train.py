"""Training script — HuBERT-ECG + MLP head for QRS/QT regression.

Since the encoder is frozen, embeddings are precomputed once and stored on CPU.
Each training batch is moved to GPU just-in-time, keeping memory usage low.
"""

import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from beat import load_patient_beats, load_or_process_beats
from dataset import BeatDataset, preprocess_hubert
from model import build_model


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

    return torch.cat(embs), torch.cat(decisions), torch.cat(ys)


class _ModelWithHead:
    """Head-only model used when the encoder output is fully cached.
    Defined at module level so it survives multiprocessing pickling.
    """
    def __init__(self, head):
        self.head = head


def load_or_build_model(cache_dir, force, device, hidden, dropout,
                        train_folders, holdout_folders):
    """Return a model object (with .head) — skips loading HuBERT when all
    embedding caches already exist, saving significant startup time.

    A tiny model_config.pt is written alongside the embedding caches so the
    head can be reconstructed without the encoder.
    """
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
        from model import ConvHead
        head = ConvHead(
            embed_dim=cfg['embed_dim'],
            embed_L=cfg['embed_L'],
            embed_t=cfg['embed_t'],
            decision_size=WINDOW_PRE + WINDOW_POST,
            hidden=hidden,
            dropout=dropout,
        ).to(device)
        return _ModelWithHead(head)

    model, _ = build_model(device=device, hidden=hidden, dropout=dropout)
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
    """Return a base path for the three embedding .npy files.

    Key = sorted folder basenames joined, so different splits produce
    different files and adding/removing patients invalidates the cache.
    Use --force to bypass.

    Actual files: <base>_embs.npy, <base>_decisions.npy, <base>_ys.npy
    """
    key = '_'.join(sorted(os.path.basename(f) for f in folders))
    return os.path.join(cache_dir, f'emb_{tag}_{key}')


def _emb_cache_exists(base):
    return all(os.path.exists(f'{base}_{s}.npy') for s in ('embs', 'decisions', 'ys'))


def load_or_precompute(model, dataset, batch_size, device,
                       cache_path, force=False, desc=''):
    """Return (embs, decisions, ys) as CPU tensors, loading from cache when available.

    Cache format: three raw .npy files loaded with mmap_mode='r' so load is
    near-instant and no extra RAM is allocated until the data is actually read.
    """
    if not force and _emb_cache_exists(cache_path):
        print(f'  [cache] loading {cache_path}_*.npy')
        embs      = torch.from_numpy(np.load(f'{cache_path}_embs.npy'))
        decisions = torch.from_numpy(np.load(f'{cache_path}_decisions.npy'))
        ys        = torch.from_numpy(np.load(f'{cache_path}_ys.npy'))
        return embs, decisions, ys

    embs, decisions, ys = precompute_embeddings(model, dataset, batch_size, device, desc=desc)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    np.save(f'{cache_path}_embs.npy',      embs.numpy())
    np.save(f'{cache_path}_decisions.npy', decisions.numpy())
    np.save(f'{cache_path}_ys.npy',        ys.numpy())
    print(f'  [cache] saved  {cache_path}_*.npy')
    return embs, decisions, ys


def mae(pred, target, delta=None):
    mask = ~torch.isnan(target)
    p, t = pred[mask], target[mask]
    if delta is None:
        return (p - t).abs().mean()
    return torch.nn.functional.huber_loss(p, t, delta=delta)


def run_epoch(head, loader, optimizer, device, train=True, scaler=None, delta=None):
    head.train(train)

    total_loss = total_qrs = total_qt = n = 0

    with torch.set_grad_enabled(train):
        for emb, d, y in loader:
            emb = emb.to(device, non_blocking=True)
            d   = d.to(device, non_blocking=True)
            y   = y.to(device, non_blocking=True)

            if train and scaler is not None:
                with torch.autocast(device_type=device.type):
                    pred = head(emb, d)
                    loss_qrs = mae(pred[:, 0], y[:, 0], delta=delta)
                    loss_qt  = mae(pred[:, 1], y[:, 1], delta=delta)
                    loss     = loss_qrs + loss_qt
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred     = head(emb, d)
                loss_qrs = mae(pred[:, 0], y[:, 0], delta=delta)
                loss_qt  = mae(pred[:, 1], y[:, 1], delta=delta)
                loss     = loss_qrs + loss_qt
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()

            B = emb.size(0)
            total_loss += loss.item() * B
            total_qrs  += loss_qrs.item() * B
            total_qt   += loss_qt.item()  * B
            n          += B

    return total_loss / n, total_qrs / n, total_qt / n


# =========================================================
# Main
# =========================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ----------------------------------------------------------
    # Data — split folders by patient
    # ----------------------------------------------------------
    all_folders   = sorted(glob.glob(os.path.join(args.data_dir, 'p*')))
    holdout_pat   = args.holdout_patient                          # e.g. 'p9'
    train_folders = [f for f in all_folders
                     if not os.path.basename(f).startswith(holdout_pat)]
    holdout_folders = [f for f in all_folders
                       if os.path.basename(f).startswith(holdout_pat)]

    print(f'Train folders   : {[os.path.basename(f) for f in train_folders]}')
    print(f'Holdout folders : {[os.path.basename(f) for f in holdout_folders]}')

    train_cp   = emb_cache_path(args.cache_dir, train_folders,   'train')
    holdout_cp = emb_cache_path(args.cache_dir, holdout_folders, 'holdout')
    need_beats = args.force or not (_emb_cache_exists(train_cp) and _emb_cache_exists(holdout_cp))

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
    print('Building model (or loading head from cache)...')
    model = load_or_build_model(
        args.cache_dir, args.force, device, args.hidden, args.dropout,
        train_folders, holdout_folders,
    )
    if hasattr(model, 'parameters'):
        print(f'Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    print('Precomputing embeddings (or loading from cache)...')
    embs, decisions, ys = load_or_precompute(
        model, train_ds_full, args.batch_size, device,
        cache_path=train_cp, force=args.force, desc='  train  ',
    )
    ho_embs, ho_decisions, ho_ys = load_or_precompute(
        model, holdout_ds, args.batch_size, device,
        cache_path=holdout_cp, force=args.force, desc='  holdout',
    )
    print(f'  train={tuple(embs.shape)}  holdout={tuple(ho_embs.shape)}')

    # ----------------------------------------------------------
    # Train / val split (within training patients only)
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

    train_dl   = DataLoader(train_ds,       batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    val_dl     = DataLoader(val_ds,         batch_size=args.batch_size, shuffle=False, pin_memory=True)
    holdout_dl = DataLoader(holdout_full_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # ----------------------------------------------------------
    # Train only the head
    # ----------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=args.lr, weight_decay=args.wd,
    )
    # warmup for 10% of epochs, then cosine decay
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
    best_val    = float('inf')
    best_ho     = float('inf')
    history     = []

    bar = tqdm(range(1, args.epochs + 1), desc='training', unit='ep')
    for epoch in bar:
        tr_loss, tr_qrs, tr_qt = run_epoch(model.head, train_dl,   optimizer, device, train=True,  scaler=scaler, delta=args.huber_delta)
        va_loss, va_qrs, va_qt = run_epoch(model.head, val_dl,     optimizer, device, train=False, delta=args.huber_delta)
        ho_loss, ho_qrs, ho_qt = run_epoch(model.head, holdout_dl, optimizer, device, train=False, delta=args.huber_delta)
        scheduler.step()

        best_val = min(best_val, va_loss)
        best_ho  = min(best_ho,  ho_loss)
        history.append(dict(
            epoch=epoch,
            tr_qrs=tr_qrs, va_qrs=va_qrs, ho_qrs=ho_qrs,
            tr_qt=tr_qt,   va_qt=va_qt,   ho_qt=ho_qt,
        ))

        bar.set_postfix(
            tr=f'{tr_loss:.1f}', va=f'{va_loss:.1f}',
            ho=f'{ho_loss:.1f}', lr=f'{scheduler.get_last_lr()[0]:.2e}',
        )

        if args.plot_every > 0 and (epoch % args.plot_every == 0 or epoch == args.epochs):
            from debug_plot import debug_plot
            path = debug_plot(
                epoch=epoch,
                history=history,
                model=model,
                train_dl=train_dl,
                val_dl=val_dl,
                holdout_dl=holdout_dl,
                device=device,
                out_dir=os.path.join(args.ckpt_dir, 'debug'),
            )
            tqdm.write(f'  [debug] saved {path}')

        if epoch % args.log_every == 0 or epoch == args.epochs:
            tqdm.write(
                f'[{epoch:04d}]  '
                f'train {tr_loss:.2f} (q={tr_qrs:.1f} t={tr_qt:.1f})  '
                f'val {va_loss:.2f} (q={va_qrs:.1f} t={va_qt:.1f})  '
                f'holdout({holdout_pat}) {ho_loss:.2f} (q={ho_qrs:.1f} t={ho_qt:.1f})  '
                f'lr={scheduler.get_last_lr()[0]:.2e}'
            )

        if va_loss < best_val and False:
            ckpt = os.path.join(args.ckpt_dir, 'best.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': best_val}, ckpt)
            tqdm.write(f'  saved {ckpt}')

    print(f'\nbest val={best_val:.2f} ms   best holdout({holdout_pat})={best_ho:.2f} ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   default='data')
    parser.add_argument('--ckpt_dir',   default='ckpts')
    parser.add_argument('--epochs',     type=int,   default=50)
    parser.add_argument('--batch_size', type=int,   default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--wd',         type=float, default=1e-4)
    parser.add_argument('--hidden',     type=int,   default=640)
    parser.add_argument('--dropout',    type=float, default=0.1)
    parser.add_argument('--val_split',       type=float, default=0.2)
    parser.add_argument('--seed',            type=int,   default=42)
    parser.add_argument('--holdout_patient', type=str,   default='p9')
    parser.add_argument('--log_every',   type=int, default=10)
    parser.add_argument('--plot_every',  type=int, default=10,
                        help='save debug plots every N epochs (0 = disabled)')
    parser.add_argument('--cache_dir',   default='cache',
                        help='directory for precomputed embedding caches')
    parser.add_argument('--force',       action='store_true',
                        help='recompute and overwrite all caches')
    parser.add_argument('--huber_delta', type=float, default=None,
                        help='Huber loss delta in ms (None = MAE)')
    args = parser.parse_args()
    main(args)
