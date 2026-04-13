"""Training script — HuBERT-ECG + MaskHead for QRS/QT interval prediction.

Since the encoder is frozen, embeddings are precomputed once and stored on CPU.
Each training batch is moved to GPU just-in-time.

Output masks: (N, 2, W) float in [0, 1]  where W = WINDOW_PRE + WINDOW_POST.
  mask[:, 0, :].sum(-1) ≈ QRS duration in ms
  mask[:, 1, :].sum(-1) ≈ QT  interval in ms

For the PT-signal baseline head, use train_pt.py instead.
For semi-supervised training with unannotated beats, use train_semi.py instead.

NOTE: embedding caches (cache/*_ys.npy) from the previous scalar-head run
are incompatible — run with --force once to rebuild them.
"""

import os
import glob
import shutil
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from beat import load_or_process_beats
from dataset import BeatDataset, preprocess_hubert
from train_utils import (
    dispatch_debug_plot,
    emb_cache_path, _emb_cache_exists, _ys_valid,
    load_or_build_model, load_or_precompute,
    collect_predictions, build_sample_data,
    tv_loss,
)


# =========================================================
# Train / eval
# =========================================================

def run_epoch(head, loader, optimizer, device, train=True, scaler=None, lambda_tv=1.0):
    head.train(train)
    total_bce = total_qrs = n = 0

    with torch.set_grad_enabled(train):
        for emb, d, y_mask in loader:
            emb    = emb.to(device, non_blocking=True)
            d      = d.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)

            if train and scaler is not None:
                with torch.autocast(device_type=device.type):
                    logits, mask, _ = head(emb, d)
                    loss = (F.binary_cross_entropy_with_logits(logits, y_mask[:, 0:1, :])
                            + lambda_tv * tv_loss(logits))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, mask, _ = head(emb, d)
                loss = (F.binary_cross_entropy_with_logits(logits, y_mask[:, 0:1, :])
                        + lambda_tv * tv_loss(logits))
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    optimizer.step()

            with torch.no_grad():
                true_dur = y_mask.sum(dim=-1)
                hard_dur = (mask > 0.5).float().sum(dim=-1)
                qrs_mae  = (hard_dur[:, 0] - true_dur[:, 0]).abs().mean().item()

            B = emb.size(0)
            total_bce += loss.item() * B
            total_qrs += qrs_mae     * B
            n         += B

    return total_bce / n, total_qrs / n, 0.0


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

    print('Building model (or loading head from cache)...')
    model = load_or_build_model(
        args.cache_dir, args.force, device, args.width,
        train_folders, holdout_folders,
    )

    print('Precomputing embeddings (or loading from cache)...')
    embs, decisions, ys, lead2 = load_or_precompute(
        model, train_ds_full, args.batch_size, device,
        cache_path=train_cp, force=args.force, desc='  train  ',
    )
    ho_embs, ho_decisions, ho_ys, ho_lead2 = load_or_precompute(
        model, holdout_ds, args.batch_size, device,
        cache_path=holdout_cp, force=args.force, desc='  holdout',
    )
    print(f'  train={tuple(embs.shape)}  holdout={tuple(ho_embs.shape)}')

    # lead2 self-heal: rebuild from beats pickle if cache predates lead2 support
    if ho_lead2 is None:
        print('  [cache] lead2 missing — rebuilding from beats cache...')
        ho_ann_raw, _, _ = load_or_process_beats(holdout_folders, args.cache_dir, force=False)
        filtered = [b for b in ho_ann_raw
                    if b.qrs_duration is not None and b.qt_interval is not None]
        ho_lead2 = torch.from_numpy(
            np.stack([b.window[2, :].astype(np.float32) for b in filtered])
        )
        np.save(f'{holdout_cp}_lead2.npy', ho_lead2.numpy())

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

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.wd)
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
        try:
            shutil.rmtree(debug_dir)
        except OSError:
            pass

    best_val = float('inf')
    best_ho  = float('inf')
    history  = []

    bar = tqdm(range(1, args.epochs + 1), desc='training', unit='ep')
    for epoch in bar:
        tr_bce, tr_qrs, tr_qt = run_epoch(model.head, train_dl,   optimizer, device, train=True,  scaler=scaler)
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
            tr_idx = torch.tensor(train_ds.indices)
            va_idx = torch.tensor(val_ds.indices)
            plot_data = {
                'train':   collect_predictions(model.head, train_dl,   device),
                'val':     collect_predictions(model.head, val_dl,     device),
                'holdout': collect_predictions(model.head, holdout_dl, device),
            }
            sample_data, labels = build_sample_data(model.head, [
                ('train',   embs[tr_idx], decisions[tr_idx], ys[tr_idx],
                 lead2[tr_idx] if lead2 is not None else None),
                ('val',     embs[va_idx], decisions[va_idx], ys[va_idx],
                 lead2[va_idx] if lead2 is not None else None),
                ('holdout', ho_embs, ho_decisions, ho_ys, ho_lead2),
            ], device)
            sample_data['labels'] = labels
            dispatch_debug_plot(
                epoch=epoch, history=history,
                plot_data=plot_data, sample_data=sample_data,
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
    parser.add_argument('--ckpt_dir',        default='ckpts')
    parser.add_argument('--epochs',          type=int,   default=50)
    parser.add_argument('--batch_size',      type=int,   default=32)
    parser.add_argument('--lr',              type=float, default=1e-3)
    parser.add_argument('--wd',              type=float, default=1e-3)
    parser.add_argument('--width',           type=int,   default=256)
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
