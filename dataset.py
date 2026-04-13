"""PyTorch Dataset for Beat objects with optional HuBERT-ECG preprocessing."""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample

from beat import FS, Beat, WINDOW_PRE, WINDOW_POST

WINDOW_SIZE = WINDOW_PRE + WINDOW_POST   # 550

# =========================================================
# HuBERT-ECG target spec
# =========================================================
HUBERT_FS        = 500    # Hz — model input sample rate
HUBERT_N_SAMPLES = 2500   # samples (5 seconds)


def preprocess_hubert(context_window):
    """
    Transform a 5-second context window into HuBERT format (MULTI-LEAD).

    Input:
        (13, CONTEXT_PRE+CONTEXT_POST) at FS (1 kHz) — centred on the spike

    Output:
        (12, 2500) at HUBERT_FS (500 Hz)
    """
    leads = []
    for lead in context_window:
        resampled = resample(lead, HUBERT_N_SAMPLES).astype(np.float32)
        mu = resampled.mean()
        std = resampled.std()
        leads.append((resampled - mu) / (std + 1e-6))

    return np.stack(leads, axis=0)[:-1, :]  # drop VD d → (12, 2500)


# =========================================================
# Mask builder
# =========================================================

def build_mask(beat):
    """Build a (2, WINDOW_SIZE) float32 binary mask from beat annotations.

    Channel 0: QRS interval  [qrs_start, qrs_start + qrs_duration)
    Channel 1: QT  interval  [qt_start,  qt_start  + qt_interval)

    All positions are in ms (= samples at 1 kHz), clipped to [0, WINDOW_SIZE).
    Summing along the last axis gives duration in ms.
    """
    mask         = np.zeros((2, WINDOW_SIZE), dtype=np.float32)
    window_start = beat.spike_idx - beat.window_pre   # absolute ms of window[0]

    if beat.qrs_start is not None and beat.qrs_duration is not None:
        lo = int(round(beat.qrs_start))    - window_start
        hi = lo + int(round(beat.qrs_duration))
        lo, hi = max(0, lo), min(WINDOW_SIZE, hi)
        if lo < hi:
            mask[0, lo:hi] = 1.0

    if beat.qt_start is not None and beat.qt_interval is not None:
        lo = int(round(beat.qt_start))     - window_start
        hi = lo + int(round(beat.qt_interval))
        lo, hi = max(0, lo), min(WINDOW_SIZE, hi)
        if lo < hi:
            mask[1, lo:hi] = 1.0

    return mask


# =========================================================
# Dataset
# =========================================================

class BeatDataset(Dataset):
    """PyTorch Dataset wrapping a list of Beat objects.

    Parameters
    ----------
    beats        : list[Beat]
    transform    : callable | None
        Applied to each context window before converting to tensor.
        Pass `preprocess_hubert` to use HuBERT-ECG formatting.
    require_both : bool
        If True (default), only beats with both qrs_duration and qt_interval
        are included.
    """

    def __init__(self, beats, transform=None, require_both=True):
        if require_both:
            beats = [b for b in beats
                     if b.qrs_duration is not None and b.qt_interval is not None]
            # Drop beats where the QRS mask sum differs from the annotation by
            # more than 10% — catches T-wave misdetections and heavily clipped beats.
            def _mask_ok(b):
                m = build_mask(b)
                qrs_sum = m[0].sum()
                return qrs_sum >= 0.9 * b.qrs_duration
            n_before = len(beats)
            beats = [b for b in beats if _mask_ok(b)]
            dropped = n_before - len(beats)
            if dropped:
                print(f'  [dataset] dropped {dropped} beat(s) with clipped/mismatched QRS mask')
        self.beats     = beats
        self.transform = transform

    def __len__(self):
        return len(self.beats)

    def __getitem__(self, idx):
        beat   = self.beats[idx]
        window = beat.context_window.astype(np.float32)

        if self.transform is not None:
            window = self.transform(window)

        x = torch.from_numpy(window)
        d = torch.from_numpy(beat.decision_window)          # (W,)
        y = torch.from_numpy(build_mask(beat))              # (2, W)

        return x, d, y


# =========================================================
# Quick test
# =========================================================

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from beat import process_study

    beats, _, _ = process_study('data/p21/ecg_data.txt')

    print('--- HuBERT dataset ---')
    ds = BeatDataset(beats, transform=preprocess_hubert)
    x, d, y = ds[0]
    print(f'  x : {tuple(x.shape)}')
    print(f'  d : {tuple(d.shape)}')
    print(f'  y : {tuple(y.shape)}  qrs={y[0].sum():.0f}ms  qt={y[1].sum():.0f}ms')

    dl = DataLoader(ds, batch_size=4, shuffle=True)
    xb, db, yb = next(iter(dl))
    print(f'  batch x:{tuple(xb.shape)}  d:{tuple(db.shape)}  y:{tuple(yb.shape)}')
