"""Semi-supervised training — HuBERT-ECG + MaskHead with continuity constraint.

Annotated beats  → BCE + TV loss  (supervised)
Unannotated beats → TV loss only  (unsupervised, high lambda)

The total variation penalty on the logits forces the mask to be smooth and
connected even on signals the model has never seen labelled ground truth for.

Usage:
    python train_semi.py [--lambda_tv_unann 10.0] [--data_dir data] ...
"""

import os
import glob
import shutil
import itertools
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from train_utils import (
    dispatch_debug_plot,
    emb_cache_path, load_cache, load_cache_unann, load_beat_types,
    collect_sample_logits,
    tv_loss, dur_prior_loss,
)


# =========================================================
# Train / eval
# =========================================================

BEAT_TYPE_ORIG  = 0   # original annotated
BEAT_TYPE_SCALE = 1   # scale-augmented
BEAT_TYPE_SHIFT = 2   # shift-augmented


def run_epoch(head, ann_loader, unann_iter, optimizer, device,
              train=True, scaler=None,
              lambda_tv_orig=0.0, lambda_tv_scale=0.0, lambda_tv_shift=0.0,
              lambda_tv_unann=0.0,
              lambda_dur=0.0, ann_dur_mu=None, dur_delta=100.0, collect=False):
    head.train(train)
    total_bce = total_qrs = n = 0

    if collect:
        all_preds, all_targets = [], []
        best_err, worst_err = float('inf'), float('-inf')
        best_sample = worst_sample = None

    with torch.set_grad_enabled(train):
        for emb, d, y_mask, l2, bt in ann_loader:
            emb    = emb.to(device, non_blocking=True)
            d      = d.to(device, non_blocking=True)
            y_mask = y_mask.to(device, non_blocking=True)
            bt     = bt.to(device, non_blocking=True)

            logits, mask, _ = head(emb, d)

            # BCE + TV on original beats (always)
            mask_orig = (bt == BEAT_TYPE_ORIG)
            loss = F.binary_cross_entropy_with_logits(
                logits[mask_orig], y_mask[mask_orig, 0:1, :])
            if mask_orig.any():
                loss = loss + lambda_tv_orig * tv_loss(logits[mask_orig])

            # Augmented beats: lambda scales both BCE and TV together
            for cls, lam in ((BEAT_TYPE_SCALE, lambda_tv_scale),
                             (BEAT_TYPE_SHIFT, lambda_tv_shift)):
                if lam > 0.0:
                    mask_cls = (bt == cls)
                    if mask_cls.any():
                        loss = loss + lam * (
                            F.binary_cross_entropy_with_logits(
                                logits[mask_cls], y_mask[mask_cls, 0:1, :])
                            + tv_loss(logits[mask_cls])
                        )

            if train:
                if lambda_tv_unann > 0.0 or lambda_dur > 0.0:
                    u_emb, u_d = next(unann_iter)
                    u_logits, u_mask, u_dur = head(u_emb.to(device, non_blocking=True),
                                                   u_d.to(device,  non_blocking=True))
                    if lambda_tv_unann > 0.0:
                        loss = loss + lambda_tv_unann * tv_loss(u_logits)
                    if lambda_dur > 0.0 and ann_dur_mu is not None:
                        loss = loss + lambda_dur * dur_prior_loss(
                            u_dur[:, 0], ann_dur_mu, dur_delta)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()

            with torch.no_grad():
                true_dur = y_mask.sum(dim=-1)
                hard_dur = (mask > 0.5).float().sum(dim=-1)
                qrs_mae  = (hard_dur[:, 0] - true_dur[:, 0]).abs().mean().item()

            if collect:
                pad = torch.zeros_like(hard_dur)
                all_preds.append(torch.cat([hard_dur, pad], dim=-1).cpu().numpy())
                all_targets.append(true_dur.cpu().numpy())
                errs = (hard_dur[:, 0] - true_dur[:, 0]).abs()
                bi = errs.argmin().item()
                wi = errs.argmax().item()
                if errs[bi].item() < best_err:
                    best_err = errs[bi].item()
                    best_sample = (emb[[bi]].cpu(), d[[bi]].cpu(), y_mask[[bi]].cpu(), l2[[bi]], bt[[bi]])
                if errs[wi].item() > worst_err:
                    worst_err = errs[wi].item()
                    worst_sample = (emb[[wi]].cpu(), d[[wi]].cpu(), y_mask[[wi]].cpu(), l2[[wi]], bt[[wi]])

            B = emb.size(0)
            total_bce += loss.item() * B
            total_qrs += qrs_mae     * B
            n         += B

    metrics = total_bce / n, total_qrs / n, 0.0
    if not collect:
        return metrics
    return metrics + ({'preds': np.concatenate(all_preds), 'targets': np.concatenate(all_targets),
                       'best': best_sample, 'worst': worst_sample},)


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

    train_tag  = f'train_aug{args.augment_seed}' if args.augment else 'train'
    train_cp   = emb_cache_path(args.cache_dir, train_folders,   train_tag)
    holdout_cp = emb_cache_path(args.cache_dir, holdout_folders, 'holdout')
    unann_cp   = emb_cache_path(args.cache_dir, train_folders,   'unann')

    print('Loading cache...')
    embs,       decisions,       ys,    all_leads    = load_cache(train_cp)
    ho_embs,    ho_decisions,    ho_ys, ho_all_leads = load_cache(holdout_cp)
    unann_embs, unann_decisions                      = load_cache_unann(unann_cp)
    print(f'  train={tuple(embs.shape)}  unann={tuple(unann_embs.shape)}  holdout={tuple(ho_embs.shape)}')

    from model import MaskHead
    from beat import WINDOW_PRE, WINDOW_POST
    head = MaskHead(embed_dim=embs.shape[-1], window_size=WINDOW_PRE + WINDOW_POST,
                    width=args.width).to(device)
    print(head)

    _al_tr = all_leads    if all_leads    is not None else torch.zeros(len(embs),    13, ys.shape[-1])
    _al_ho = ho_all_leads if ho_all_leads is not None else torch.zeros(len(ho_embs), 13, ho_ys.shape[-1])
    _bt_tr = load_beat_types(train_cp)
    if _bt_tr is None:
        _bt_tr = torch.zeros(len(embs), dtype=torch.long)
    full_ds = TensorDataset(embs, decisions, ys, _al_tr, _bt_tr)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    _bt_ho = torch.zeros(len(ho_embs), dtype=torch.long)
    holdout_full_ds = TensorDataset(ho_embs, ho_decisions, ho_ys, _al_ho, _bt_ho)
    unann_full_ds   = TensorDataset(unann_embs, unann_decisions)

    print(f'Train / Val / Holdout ({holdout_pat}) / Unannotated : '
          f'{n_train} / {n_val} / {len(holdout_full_ds)} / {len(unann_full_ds)}')

    # Annotated QRS duration distribution (ground truth mask sums, ms)
    ann_qrs_dur = ys[:, 0, :].sum(dim=-1).float()
    ann_dur_mu  = ann_qrs_dur.mean().item()
    print(f'Annotated QRS dur  mean={ann_dur_mu:.1f} ms  '
          f'std={ann_qrs_dur.std().item():.1f} ms  '
          f'dead-zone ±{args.dur_delta:.0f} ms')

    train_dl   = DataLoader(train_ds,        batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    val_dl     = DataLoader(val_ds,          batch_size=args.batch_size, shuffle=False, pin_memory=True)
    holdout_dl = DataLoader(holdout_full_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    unann_iter = itertools.cycle(
        DataLoader(unann_full_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    )

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)
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
        do_collect = args.plot_every > 0 and (epoch % args.plot_every == 0 or epoch == args.epochs)
        tr_bce, tr_qrs, tr_qt, *tr_c = run_epoch(
            head, train_dl, unann_iter, optimizer, device,
            train=True, scaler=scaler,
            lambda_tv_orig=args.lambda_tv_orig,
            lambda_tv_scale=args.lambda_tv_scale,
            lambda_tv_shift=args.lambda_tv_shift,
            lambda_tv_unann=args.lambda_tv_unann,
            lambda_dur=args.lambda_dur,
            ann_dur_mu=ann_dur_mu,
            dur_delta=args.dur_delta,
            collect=do_collect,
        )
        va_bce, va_qrs, va_qt, *va_c = run_epoch(
            head, val_dl, unann_iter, optimizer, device, train=False, collect=do_collect)
        ho_bce, ho_qrs, ho_qt, *ho_c = run_epoch(
            head, holdout_dl, unann_iter, optimizer, device, train=False, collect=do_collect)
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

        if do_collect:
            tr_col, va_col, ho_col = tr_c[0], va_c[0], ho_c[0]
            plot_data = {
                'train':   (tr_col['preds'], tr_col['targets']),
                'val':     (va_col['preds'], va_col['targets']),
                'holdout': (ho_col['preds'], ho_col['targets']),
            }
            _beat_type_names = {0: 'original', 1: 'scale', 2: 'shift'}
            parts, labels = [], []
            for name, col in [('train', tr_col), ('val', va_col), ('holdout', ho_col)]:
                for tag, samp in [('best', col['best']), ('worst', col['worst'])]:
                    e, d_s, y, l2, bt = samp
                    parts.append(collect_sample_logits(head, e, d_s, y, l2, device))
                    type_str = _beat_type_names.get(int(bt.item()), '?')
                    labels.append(f'{name} ({tag}, {type_str})')
            keys = ['logits', 'mask', 'f_sig', 'g_sig', 'decision', 'y_mask']
            sample_data = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
            al_parts = [p['all_leads'] for p in parts]
            sample_data['all_leads'] = (
                np.concatenate(al_parts, axis=0)
                if all(x is not None for x in al_parts) else None
            )
            if all_leads is None and ho_all_leads is None:
                sample_data['all_leads'] = None
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
    parser.add_argument('--data_dir',         default='data')
    parser.add_argument('--ckpt_dir',         default='ckpts_semi')
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
    parser.add_argument('--augment',          action='store_true',
                        help='expand annotated training set 10x with scale+shift augmentation')
    parser.add_argument('--augment_seed',     type=int, default=42,
                        help='RNG seed for augmentation (also used as cache key)')
    parser.add_argument('--lambda_tv_orig',   type=float, default=0.2,
                        help='TV weight on original annotated beats (beat_type=0)')
    parser.add_argument('--lambda_tv_scale',  type=float, default=0.0,
                        help='TV weight on scale-augmented beats (beat_type=1)')
    parser.add_argument('--lambda_tv_shift',  type=float, default=0.0,
                        help='TV weight on shift-augmented beats (beat_type=2)')
    parser.add_argument('--lambda_tv_unann',  type=float, default=0.0,
                        help='TV weight on natural unannotated beats')
    parser.add_argument('--lambda_dur',       type=float, default=0.01,
                        help='weight for duration prior loss on unannotated beats')
    parser.add_argument('--dur_delta',        type=float, default=70.0,
                        help='dead-zone half-width in ms (no penalty within ±dur_delta of annotated mean)')
    args = parser.parse_args()
    main(args)
