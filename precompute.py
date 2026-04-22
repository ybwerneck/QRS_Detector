"""Precompute and cache all embeddings, decisions, masks, and leads.

Run once before any training script:
    python precompute.py [--augment_seed 42] [--force]

Produces under cache_dir:
    emb_train_{key}_*           annotated train (no augment)
    emb_train_aug{seed}_{key}_* annotated train + scale/shift augmentation
    emb_holdout_{key}_*         holdout beats
    emb_unann_{key}_*           unannotated train (embs + decisions, no ys)

Files per split: *_embs.npy  *_decisions.npy  *_ys.npy  *_all_leads.npy
Unannotated:     *_embs.npy  *_decisions.npy
"""

import os
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from beat import load_or_process_beats
from dataset import BeatDataset, preprocess_hubert, generate_expansion_scale, generate_expansion_shift
from train_utils import (
    emb_cache_path,
    _emb_cache_exists, _unann_cache_exists, _ys_valid,
    _save_embs, _np_save_atomic, _pad_to_13,
)


@torch.no_grad()
def _run_encoder(model, dataset, batch_size, device, desc=''):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    embs, decisions, ys = [], [], []
    model.eval()
    for x, d, y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        embs.append(model.encode(x).cpu())
        decisions.append(d)
        ys.append(y)
    all_leads = torch.from_numpy(
        np.stack([_pad_to_13(b.window) for b in dataset.beats])
    )
    return torch.cat(embs), torch.cat(decisions), torch.cat(ys), all_leads


@torch.no_grad()
def _run_encoder_unann(model, dataset, batch_size, device, desc=''):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    embs, decisions = [], []
    model.eval()
    for x, d, _y in tqdm(loader, desc=desc, leave=False):
        x = x.to(device, non_blocking=True)
        embs.append(model.encode(x).cpu())
        decisions.append(d)
    return torch.cat(embs), torch.cat(decisions)


def _save(base, embs, decisions, ys, all_leads):
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
    _save_embs(base, embs)
    np.save(f'{base}_decisions.npy', decisions.numpy())
    np.save(f'{base}_ys.npy',        ys.numpy())
    _np_save_atomic(f'{base}_all_leads.npy', all_leads.numpy())
    print(f'  saved {base}_* ({len(embs)} beats)')


def _save_unann(base, embs, decisions):
    os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
    _save_embs(base, embs)
    np.save(f'{base}_decisions.npy', decisions.numpy())
    print(f'  saved {base}_embs/decisions ({len(embs)} beats)')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    all_folders     = sorted(glob.glob(os.path.join(args.data_dir, 'p*')))
    train_folders   = [f for f in all_folders
                       if not os.path.basename(f).startswith(args.holdout_patient)]
    holdout_folders = [f for f in all_folders
                       if os.path.basename(f).startswith(args.holdout_patient)]

    print(f'Train   : {[os.path.basename(f) for f in train_folders]}')
    print(f'Holdout : {[os.path.basename(f) for f in holdout_folders]}')

    train_cp   = emb_cache_path(args.cache_dir, train_folders,   'train')
    aug_cp     = emb_cache_path(args.cache_dir, train_folders,   f'train_aug{args.augment_seed}')
    holdout_cp = emb_cache_path(args.cache_dir, holdout_folders, 'holdout')
    unann_cp   = emb_cache_path(args.cache_dir, train_folders,   'unann')

    print('Loading beats...')
    train_ann, train_unann, _ = load_or_process_beats(train_folders,   args.cache_dir, args.force)
    holdout_ann, _, _         = load_or_process_beats(holdout_folders, args.cache_dir, args.force)
    print(f'  annotated={len(train_ann)}  unannotated={len(train_unann)}  holdout={len(holdout_ann)}')

    n_aug       = int(len(train_ann) * 4.5)
    scale_beats = generate_expansion_scale(train_ann, n=n_aug, seed=args.augment_seed)
    shift_beats = generate_expansion_shift(train_ann, n=n_aug, max_shift=100,
                                           seed=args.augment_seed + 1)
    train_aug   = train_ann + scale_beats + shift_beats
    print(f'  augmented: {len(train_ann)} → {len(train_aug)} (+{n_aug} scale, +{n_aug} shift)')

    ds_train   = BeatDataset(train_ann,   transform=preprocess_hubert)
    ds_aug     = BeatDataset(train_aug,   transform=preprocess_hubert)
    ds_holdout = BeatDataset(holdout_ann, transform=preprocess_hubert)
    ds_unann   = BeatDataset(train_unann, transform=preprocess_hubert, require_both=False)

    need = {
        'train':   args.force or not (_emb_cache_exists(train_cp)   and _ys_valid(train_cp)),
        'aug':     args.force or not (_emb_cache_exists(aug_cp)     and _ys_valid(aug_cp)),
        'holdout': args.force or not (_emb_cache_exists(holdout_cp) and _ys_valid(holdout_cp)),
        'unann':   args.force or not _unann_cache_exists(unann_cp),
    }

    if not any(need.values()):
        print('All caches up to date.')
        return

    print('Loading HuBERT encoder...')
    from model import build_model
    model, _ = build_model(device=device)
    model.eval()
    os.makedirs(args.cache_dir, exist_ok=True)

    if need['train']:
        print('Encoding train (no augment)...')
        embs, dec, ys, al = _run_encoder(model, ds_train, args.batch_size, device, '  train')
        _save(train_cp, embs, dec, ys, al)
    else:
        print(f'  [skip] train cache exists')

    if need['aug']:
        print('Encoding train+aug...')
        embs, dec, ys, al = _run_encoder(model, ds_aug, args.batch_size, device, '  train+aug')
        _save(aug_cp, embs, dec, ys, al)
        beat_types = np.array([
            1 if hasattr(b, 'lead_scales') else (2 if hasattr(b, 'shift') else 0)
            for b in ds_aug.beats
        ], dtype=np.int8)
        _np_save_atomic(f'{aug_cp}_beat_types.npy', beat_types)
        print(f'  beat types: orig={( beat_types==0).sum()}  scale={(beat_types==1).sum()}  shift={(beat_types==2).sum()}')
    else:
        print(f'  [skip] aug cache exists')

    if need['holdout']:
        print('Encoding holdout...')
        embs, dec, ys, al = _run_encoder(model, ds_holdout, args.batch_size, device, '  holdout')
        _save(holdout_cp, embs, dec, ys, al)
    else:
        print(f'  [skip] holdout cache exists')

    if need['unann']:
        print('Encoding unannotated...')
        embs, dec = _run_encoder_unann(model, ds_unann, args.batch_size, device, '  unann')
        _save_unann(unann_cp, embs, dec)
    else:
        print(f'  [skip] unann cache exists')

    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',        default='data')
    parser.add_argument('--cache_dir',       default='cache')
    parser.add_argument('--holdout_patient', default='p9')
    parser.add_argument('--augment_seed',    type=int, default=42)
    parser.add_argument('--batch_size',      type=int, default=32)
    parser.add_argument('--force',           action='store_true')
    args = parser.parse_args()
    main(args)
