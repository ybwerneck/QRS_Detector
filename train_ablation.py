"""Ablation training script — trains three head variants simultaneously.

Variants
--------
  full      : embedding + decision  (no ablation)
  emb_only  : embedding only        (decision zeroed)
  dec_only  : decision only         (embedding zeroed)

All variants share the frozen encoder and the same precomputed embeddings.
Parallelism: torch.vmap runs all three heads simultaneously on the same batch.
Each head has independent weights; a single stacked optimizer updates all three.
"""

import os
import sys
import copy
import glob
import shutil
import argparse
import pickle
import subprocess
import tempfile
import torch
from torch.func import stack_module_state, functional_call
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from beat import load_patient_beats, load_or_process_beats
from dataset import BeatDataset, preprocess_hubert
from model import build_model
from train import (precompute_embeddings, load_or_precompute, emb_cache_path,
                   _emb_cache_exists, load_or_build_model, mae)


# =========================================================
# Parallel plotting helpers
# =========================================================

_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_plot_worker.py')
_plot_procs: list = []


def _prune_procs():
    global _plot_procs
    _plot_procs = [p for p in _plot_procs if p.poll() is None]


def dispatch_debug_plot(**kwargs):
    """Spawn a fresh Python process to render debug plots.

    kwargs must include 'plot_data': dict with keys 'train'/'val'/'holdout',
    each a (preds_np, targets_np) tuple of shape (N, 2).

    Each invocation starts a clean interpreter, so 'import debug_plot' always
    reads the .py file from disk — edits made while training are picked up on
    the next plot tick without restarting.
    """
    _prune_procs()

    fd, tmp_path = tempfile.mkstemp(suffix='.pkl')
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(kwargs, f)

    out_dir = kwargs.get('out_dir', 'debug')
    epoch   = kwargs.get('epoch', 0)
    proc = subprocess.Popen([sys.executable, _WORKER, tmp_path])
    _plot_procs.append(proc)
    tqdm.write(f'  [plot] dispatched epoch {epoch} → {out_dir}')


# =========================================================
# Variant definitions  (masking is applied externally)
# =========================================================

VARIANTS = [
    dict(name='full',     zero_emb=False, zero_dec=False),
    dict(name='emb_only', zero_emb=False, zero_dec=True),
    dict(name='dec_only', zero_emb=True,  zero_dec=False),
]
N_VARIANTS = len(VARIANTS)


# =========================================================
# Vmapped forward
# =========================================================

def make_vmapped_fwd(base_head):
    def fwd_single(params, buffers, emb, dec):
        return functional_call(base_head, (params, buffers), (emb, dec))
    return torch.vmap(fwd_single, in_dims=(0, 0, 0, 0), randomness='different')


def mask_inputs(emb, d, variant):
    e   = torch.zeros_like(emb) if variant['zero_emb'] else emb
    dec = torch.zeros_like(d)   if variant['zero_dec'] else d
    return e, dec


# =========================================================
# Epoch runner
# =========================================================

def run_epoch_vmapped(vmapped_fwd, params, buffers, loader,
                      optimizer, device, train=True, scaler=None, collect=False, delta=None):

    totals    = {v['name']: [0., 0., 0., 0] for v in VARIANTS}
    collected = {v['name']: ([], []) for v in VARIANTS} if collect else None

    with torch.set_grad_enabled(train):
        for emb, d, y in loader:
            emb = emb.to(device, non_blocking=True)
            d   = d.to(device,   non_blocking=True)
            y   = y.to(device,   non_blocking=True)

            embs_v = torch.stack([mask_inputs(emb, d, v)[0] for v in VARIANTS])
            decs_v = torch.stack([mask_inputs(emb, d, v)[1] for v in VARIANTS])

            def step():
                preds = vmapped_fwd(params, buffers, embs_v, decs_v)  # (V, B, 2)
                loss  = torch.tensor(0., device=device)
                for i, v in enumerate(VARIANTS):
                    l_qrs = mae(preds[i, :, 0], y[:, 0], delta=delta)
                    l_qt  = mae(preds[i, :, 1], y[:, 1], delta=delta)
                    loss  = loss + l_qrs + l_qt
                    B = emb.size(0)
                    totals[v['name']][0] += (l_qrs + l_qt).item() * B
                    totals[v['name']][1] += l_qrs.item() * B
                    totals[v['name']][2] += l_qt.item()  * B
                    totals[v['name']][3] += B
                    if collected is not None:
                        collected[v['name']][0].append(preds[i].detach().cpu())
                        collected[v['name']][1].append(y.detach().cpu())
                return loss

            if train:
                if scaler is not None:
                    with torch.autocast(device_type=device.type):
                        loss = step()
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = step()
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
                    optimizer.step()
            else:
                step()

    metrics = {
        v['name']: (
            totals[v['name']][0] / max(totals[v['name']][3], 1),
            totals[v['name']][1] / max(totals[v['name']][3], 1),
            totals[v['name']][2] / max(totals[v['name']][3], 1),
        )
        for v in VARIANTS
    }

    if collect:
        import numpy as np
        data = {
            v['name']: (
                torch.cat(collected[v['name']][0]).numpy(),
                torch.cat(collected[v['name']][1]).numpy(),
            )
            for v in VARIANTS
        }
        return metrics, data

    return metrics


# =========================================================
# Main
# =========================================================

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # ----------------------------------------------------------
    # Data
    # ----------------------------------------------------------
    all_folders     = sorted(glob.glob(os.path.join(args.data_dir, 'p*')))
    train_folders   = [f for f in all_folders
                       if not os.path.basename(f).startswith(args.holdout_patient)]
    holdout_folders = [f for f in all_folders
                       if os.path.basename(f).startswith(args.holdout_patient)]

    print(f'Train   : {[os.path.basename(f) for f in train_folders]}')
    print(f'Holdout : {[os.path.basename(f) for f in holdout_folders]}')

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
    # Shared encoder — precompute embeddings once
    # ----------------------------------------------------------
    print('Building model (or loading head from cache)...')
    model = load_or_build_model(
        args.cache_dir, args.force, device, args.hidden, args.dropout,
        train_folders, holdout_folders,
    )
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

    full_ds = TensorDataset(embs, decisions, ys)
    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    holdout_full_ds = TensorDataset(ho_embs, ho_decisions, ho_ys)
    print(f'Train / Val / Holdout ({args.holdout_patient}) : '
          f'{n_train} / {n_val} / {len(holdout_full_ds)}')

    train_dl   = DataLoader(train_ds,        batch_size=args.batch_size, shuffle=True,  pin_memory=True)
    val_dl     = DataLoader(val_ds,          batch_size=args.batch_size, shuffle=False, pin_memory=True)
    holdout_dl = DataLoader(holdout_full_ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # ----------------------------------------------------------
    # Stack N_VARIANTS independent heads
    # ----------------------------------------------------------
    heads = []
    for v in VARIANTS:
        h = copy.deepcopy(model.head).to(device)
        h.set_ablation(embedding=False, decision=False)
        heads.append(h)
        print(f'  variant={v["name"]:10s}  zero_emb={v["zero_emb"]}  zero_dec={v["zero_dec"]}')

    params, buffers = stack_module_state(heads)
    base_head = copy.deepcopy(heads[0])
    base_head.train()
    vmapped_fwd = make_vmapped_fwd(base_head)

    warmup_epochs = max(1, args.epochs // 10)
    optimizer = torch.optim.AdamW(list(params.values()), lr=args.lr, weight_decay=args.wd)
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

    # Wipe debug folders so each run starts clean
    for v in VARIANTS:
        debug_dir = os.path.join(args.ckpt_dir, 'debug', v['name'])
        if os.path.isdir(debug_dir):
            shutil.rmtree(debug_dir)

    histories = {v['name']: [] for v in VARIANTS}
    best_val  = {v['name']: float('inf') for v in VARIANTS}
    best_ho   = {v['name']: float('inf') for v in VARIANTS}

    # ----------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------
    bar = tqdm(range(1, args.epochs + 1), desc='training', unit='ep')
    for epoch in bar:
        base_head.train()
        tr = run_epoch_vmapped(vmapped_fwd, params, buffers, train_dl,
                               optimizer, device, train=True, scaler=scaler, delta=args.huber_delta)
        base_head.eval()
        va = run_epoch_vmapped(vmapped_fwd, params, buffers, val_dl,
                               optimizer, device, train=False, delta=args.huber_delta)
        ho = run_epoch_vmapped(vmapped_fwd, params, buffers, holdout_dl,
                               optimizer, device, train=False, delta=args.huber_delta)
        scheduler.step()

        for v in VARIANTS:
            n = v['name']
            best_val[n] = min(best_val[n], va[n][0])
            best_ho[n]  = min(best_ho[n],  ho[n][0])
            histories[n].append(dict(
                epoch=epoch,
                tr_qrs=tr[n][1], va_qrs=va[n][1], ho_qrs=ho[n][1],
                tr_qt =tr[n][2], va_qt =va[n][2], ho_qt =ho[n][2],
            ))

        bar.set_postfix({v['name'][:3]: f'{va[v["name"]][0]:.1f}' for v in VARIANTS})

        if epoch % args.log_every == 0 or epoch == args.epochs:
            lines = [f'[{epoch:04d}]']
            for v in VARIANTS:
                n = v['name']
                lines.append(
                    f'  {n:10s}  tr={tr[n][0]:.2f}  va={va[n][0]:.2f}  ho={ho[n][0]:.2f}')
            tqdm.write('\n'.join(lines))

        if args.plot_every > 0 and (epoch % args.plot_every == 0 or epoch == args.epochs):
            base_head.eval()
            _, tr_pt = run_epoch_vmapped(vmapped_fwd, params, buffers, train_dl,
                                         None, device, train=False, collect=True, delta=args.huber_delta)
            _, va_pt = run_epoch_vmapped(vmapped_fwd, params, buffers, val_dl,
                                         None, device, train=False, collect=True, delta=args.huber_delta)
            _, ho_pt = run_epoch_vmapped(vmapped_fwd, params, buffers, holdout_dl,
                                         None, device, train=False, collect=True, delta=args.huber_delta)
            for v in VARIANTS:
                n = v['name']
                dispatch_debug_plot(
                    epoch=epoch,
                    history=histories[n],
                    plot_data={
                        'train':   tr_pt[n],
                        'val':     va_pt[n],
                        'holdout': ho_pt[n],
                    },
                    out_dir=os.path.join(args.ckpt_dir, 'debug', n),
                    zero_emb=v['zero_emb'],
                    zero_dec=v['zero_dec'],
                )
                tqdm.write(f'  [debug:{n}] dispatched epoch {epoch}')

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print('\n=== Final results ===')
    for v in VARIANTS:
        n = v['name']
        print(f'  {n:10s}  best_val={best_val[n]:.2f} ms  best_holdout={best_ho[n]:.2f} ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',         default='data')
    parser.add_argument('--ckpt_dir',         default='ckpts_ablation')
    parser.add_argument('--epochs',           type=int,   default=200)
    parser.add_argument('--batch_size',       type=int,   default=64)
    parser.add_argument('--lr',               type=float, default=1e-3)
    parser.add_argument('--wd',               type=float, default=1e-4)
    parser.add_argument('--hidden',           type=int,   default=640)
    parser.add_argument('--dropout',          type=float, default=0.1)
    parser.add_argument('--val_split',        type=float, default=0.2)
    parser.add_argument('--seed',             type=int,   default=42)
    parser.add_argument('--holdout_patient',  type=str,   default='p9')
    parser.add_argument('--log_every',   type=int, default=10)
    parser.add_argument('--plot_every',  type=int, default=50)
    parser.add_argument('--cache_dir',   default='cache',
                        help='directory for precomputed embedding caches')
    parser.add_argument('--force',       action='store_true',
                        help='recompute and overwrite all caches')
    parser.add_argument('--huber_delta', type=float, default=None,
                        help='Huber loss delta in ms (None = MAE)')
    args = parser.parse_args()
    main(args)
