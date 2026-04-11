"""PyTorch Dataset for Beat objects with optional HuBERT-ECG preprocessing."""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import resample

from beat import FS, Beat

# =========================================================
# HuBERT-ECG target spec
# =========================================================
HUBERT_FS        = 500    # Hz — model input sample rate
HUBERT_N_SAMPLES = 2500   # samples (5 seconds)
HUBERT_LEAD_IDX  = 0      # Lead I


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
        leads.append(resampled)

    return np.stack(leads, axis=0)[:-1, :]  # drop VD d → (12, 2500)

# =========================================================
# Dataset
# =========================================================

class BeatDataset(Dataset):
    """PyTorch Dataset wrapping a list of Beat objects.

    Parameters
    ----------
    beats       : list[Beat]
    transform   : callable | None
        Applied to each window (np.ndarray) before converting to tensor.
        Pass `preprocess_hubert` to use HuBERT-ECG formatting.
        None uses the raw window as-is.
    require_both : bool
        If True (default), only beats with both qrs_duration and qt_interval
        are included.  Set False to include beats with partial / no labels
        (targets will be NaN).
    """

    def __init__(self, beats, transform=None, require_both=True):
        if require_both:
            beats = [b for b in beats
                     if b.qrs_duration is not None and b.qt_interval is not None]
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
        d = torch.from_numpy(beat.decision_window)   # (WINDOW_PRE+WINDOW_POST,)

        qrs = beat.qrs_duration if beat.qrs_duration is not None else float('nan')
        qt  = beat.qt_interval  if beat.qt_interval  is not None else float('nan')
        y   = torch.tensor([qrs, qt], dtype=torch.float32)

        return x, d, y


# =========================================================
# Quick test
# =========================================================

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from beat import process_study

    beats, _, _ = process_study('data/p21/ecg_data.txt')

    print('--- Raw dataset ---')
    ds_raw = BeatDataset(beats)
    x, y   = ds_raw[0]
    print(f'  x shape : {tuple(x.shape)}   y : {y}')

    print('--- HuBERT dataset ---')
    ds_hub = BeatDataset(beats, transform=preprocess_hubert)
    x, d, y = ds_hub[0]
    print(f'  x shape : {tuple(x.shape)}   d shape : {tuple(d.shape)}   y : {y}')

    dl = DataLoader(ds_hub, batch_size=4, shuffle=True)
    xb, db, yb = next(iter(dl))
    print(f'  batch x : {tuple(xb.shape)}   batch d : {tuple(db.shape)}   batch y : {tuple(yb.shape)}')
