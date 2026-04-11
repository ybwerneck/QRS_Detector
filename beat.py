import os
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d

FS           = 1000   # Hz  (1 sample == 1 ms)
WINDOW_PRE   = 50     # samples before spike
WINDOW_POST  = 500    # samples after spike  (QRS + ST + T-wave)

CONTEXT_PRE  = 2500   # samples before spike for encoder context (2.5 s at 1 kHz)
CONTEXT_POST = 2500   # samples after  spike for encoder context (2.5 s at 1 kHz)

_ANNOTATION_SECTIONS = {
    'Period Data', 'QRS Data', 'QT Data',
    'Extrasystole Data', 'Arrhythmia Data',
}


# =========================================================
# 1. BEAT  —  one spike + multi-channel window + annotations
# =========================================================

class Beat:
    def __init__(self, spike_idx, window):
        """
        spike_idx : int          sample index in the raw signal (== ms at 1 kHz)
        window    : np.ndarray   shape (n_leads, WINDOW_PRE+WINDOW_POST)
        """
        self.spike_idx    = spike_idx
        self.spike_time   = spike_idx          # ms
        self.window       = window             # (n_leads, window_size)
        self.window_pre   = WINDOW_PRE         # samples before spike in window
        self.window_post  = WINDOW_POST        # actual signal samples after spike (excl. zero-padding)

        self.label        = None               # 0=normal 1=arrhythmia 2=extrasystole
        self.period       = None               # RR interval ms
        self.qrs_duration = None
        self.qrs_start    = None               # absolute ms position of QRS onset
        self.qt_interval  = None
        self.qt_start     = None               # absolute ms position of QT onset
        self.noisy        = False              # True if window overlaps another beat

        self.context_window   = None          # (n_leads, CONTEXT_PRE+CONTEXT_POST) — 5 s centred on spike
        self.spike_in_context = CONTEXT_PRE   # sample index of spike inside context_window (at FS)
        self.decision_window  = None          # (WINDOW_PRE+WINDOW_POST,) Pan-Tompkins integrated signal

    def annotate(self, label=None, period=None,
                 qrs_duration=None, qrs_start=None,
                 qt_interval=None, qt_start=None):
        if label        is not None: self.label        = label
        if period       is not None: self.period       = period
        if qrs_duration is not None: self.qrs_duration = qrs_duration
        if qrs_start    is not None: self.qrs_start    = qrs_start
        if qt_interval  is not None: self.qt_interval  = qt_interval
        if qt_start     is not None: self.qt_start     = qt_start

    @property
    def is_annotated(self):
        return self.label is not None

    def __repr__(self):
        return (f"Beat(spike={self.spike_idx}ms, label={self.label}, "
                f"qrs={self.qrs_duration}, qt={self.qt_interval}, "
                f"noisy={self.noisy})")


def mark_noisy_beats(beats):
    """Mark beats whose windows overlap with any other beat's window in-place.

    A beat's window spans [spike_idx - WINDOW_PRE, spike_idx + WINDOW_POST).
    Two beats are considered overlapping (noisy) when their windows intersect.
    """
    intervals = np.array(
        [(b.spike_idx - WINDOW_PRE, b.spike_idx + WINDOW_POST) for b in beats]
    )
    for i, beat in enumerate(beats):
        lo_i, hi_i = intervals[i]
        for j in range(len(beats)):
            if i == j:
                continue
            lo_j, hi_j = intervals[j]
            if lo_i < hi_j and hi_i > lo_j:   # intervals overlap
                beat.noisy = True
                break
    return beats


def recover_noisy_beats(beats, signal_matrix):
    """Re-extract windows for noisy beats, always protecting each beat's beginning.

    The right boundary of each beat is cut at the start of the next beat's
    natural window (next_spike - WINDOW_PRE), so the next beat's pre-spike
    region is always preserved intact.  The lost tail is compensated by
    extending the current beat's start leftward.  Any remaining gap (signal
    edge or extremely short RR interval) is zero-padded so the window shape
    stays constant.

    Beats whose available range is less than half the window size are left
    noisy as they cannot carry useful information.

    Parameters
    ----------
    beats         : list[Beat]   output of mark_noisy_beats (modified in-place)
    signal_matrix : np.ndarray   shape (n_leads, n_samples)

    Returns
    -------
    beats : list[Beat]
    """
    n_leads, n_samples = signal_matrix.shape
    window_size = WINDOW_PRE + WINDOW_POST

    sorted_beats = sorted(beats, key=lambda b: b.spike_idx)
    spike_pos    = [b.spike_idx for b in sorted_beats]
    n            = len(sorted_beats)

    for i, beat in enumerate(sorted_beats):
        if not beat.noisy:
            continue

        spike = beat.spike_idx

        # Right limit: start of the next beat's natural window.
        # Left limit: midpoint to the previous beat.
        lo_hard = (spike_pos[i - 1] + spike) // 2 if i > 0     else 0
        hi_hard = spike_pos[i + 1] - WINDOW_PRE   if i < n - 1 else n_samples

        hi = min(spike + WINDOW_POST, hi_hard, n_samples)
        # Extend start leftward to recover what the clipped tail lost
        lo = max(hi - window_size, lo_hard, 0)

        if hi - lo < window_size // 2:
            continue   # too cramped to be useful

        actual = signal_matrix[:, lo:hi]
        got    = hi - lo
        if got < window_size:
            padded          = np.empty((n_leads, window_size), dtype=signal_matrix.dtype)
            padded[:, :got] = actual
            padded[:, got:] = actual[:, -1:]   # extend last real sample instead of zero
            beat.window     = padded
        else:
            beat.window     = actual.copy()
        beat.window_pre  = spike - lo      # may be > WINDOW_PRE if start extended
        beat.window_post = hi - spike
        beat.noisy       = False

    return beats


def extract_context_windows(signal_matrix, beats):
    """Attach a 5-second encoder context window to each beat, centred on its spike.

    The context spans [spike - CONTEXT_PRE, spike + CONTEXT_POST) at FS.
    Regions outside the signal are zero-padded so the shape is always constant.
    beat.spike_in_context records where the spike falls inside the context (at FS).
    """
    n_leads, n_samples = signal_matrix.shape
    context_size = CONTEXT_PRE + CONTEXT_POST

    for beat in beats:
        lo = beat.spike_idx - CONTEXT_PRE
        hi = beat.spike_idx + CONTEXT_POST

        pad_left  = max(0, -lo)
        pad_right = max(0, hi - n_samples)
        lo_clamp  = max(0, lo)
        hi_clamp  = min(n_samples, hi)

        ctx = np.zeros((n_leads, context_size), dtype=signal_matrix.dtype)
        ctx[:, pad_left: pad_left + (hi_clamp - lo_clamp)] = signal_matrix[:, lo_clamp:hi_clamp]

        beat.context_window   = ctx
        beat.spike_in_context = CONTEXT_PRE  # always centred; pad shifts signal, not spike

    return beats


# =========================================================
# 2. FILE PARSER  —  all leads + embedded annotations
# =========================================================

def load_ecg(filepath):
    """Parse a study file and return all leads and annotation sections.

    Returns
    -------
    leads : dict[str, np.ndarray]
        Lead name → signal array of shape (n_samples,).
    annotations : dict[str, list[list[float]]]
        Section name → list of rows, each row a list of floats.

    Lead order in the file: VD d, I, II, III, AVR, AVL, AVF, V1..V6
    Annotation sections: Period Data, QRS Data, QT Data,
                         Extrasystole Data, Arrhythmia Data
    """
    leads       = {}
    annotations = {}

    current_name = None
    current_mode = None        # 'lead' | 'annotation'
    current_data = []
    in_curves    = False

    with open(filepath) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if not in_curves:
                if line == 'Curves:':
                    in_curves = True
                continue

            if line.endswith(':'):
                # Save previous section
                if current_name is not None:
                    if current_mode == 'lead':
                        leads[current_name] = np.array(
                            current_data, dtype=np.float32)
                    else:
                        annotations[current_name] = current_data

                section_name = line[:-1]          # strip trailing ':'
                current_name = section_name
                current_data = []
                current_mode = (
                    'annotation' if section_name in _ANNOTATION_SECTIONS
                    else 'lead'
                )
                continue

            if current_mode == 'lead':
                try:
                    current_data.append(float(line))
                except ValueError:
                    pass
            elif current_mode == 'annotation':
                try:
                    current_data.append(
                        [float(v.strip()) for v in line.split(',')])
                except ValueError:
                    pass

    # Flush last section
    if current_name is not None:
        if current_mode == 'lead':
            leads[current_name] = np.array(current_data, dtype=np.float32)
        else:
            annotations[current_name] = current_data

    return leads, annotations


def leads_matrix(leads):
    """Stack all leads into a (n_leads, n_samples) matrix in canonical order."""
    order  = ['VD d', 'I', 'II', 'III', 'AVR', 'AVL', 'AVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    names  = [n for n in order if n in leads]
    matrix = np.stack([leads[n] for n in names], axis=0)
    return matrix, names


# =========================================================
# 3. SPIKE DETECTOR  —  Pan–Tompkins
# =========================================================

def _pan_tompkins_core(
    sig,
    fs,
    min_distance_ms=200,
    band=(5, 40),
    integration_window=150,
    percentile=80,
):
    """
    Shared Pan–Tompkins preprocessing + peak detection.

    Returns
    -------
    filtered   : bandpassed ECG
    integrated : decision variable
    peaks      : detected peaks (DELAYED)
    thr2       : final adaptive threshold
    delay      : integration delay (samples)
    """
    sig = sig.astype(np.float64)

    # 1. Bandpass filter
    b, a = butter(2, band, btype="band", fs=fs)
    filtered = filtfilt(b, a, sig)

    # 2. Derivative
    dy = np.diff(filtered, prepend=filtered[0])

    # 3–4. Squaring + moving window integration
    integrated = uniform_filter1d(dy**2, size=integration_window)
    integrated = np.clip(integrated, a_min=0, a_max=120)  # numerical stability
    delay = integration_window // 2

    # 5. Adaptive thresholding
    thr1 = np.percentile(integrated, percentile)

    peaks, props = find_peaks(
        integrated,
        height=thr1,
        distance=int(min_distance_ms),
    )

    if len(peaks) > 0:
        heights = props["peak_heights"]
        cap = np.percentile(heights, 99)
        thr2 = 0.5 * heights[heights <= cap].mean()

        peaks, _ = find_peaks(
            integrated,
            height=thr2,
            distance=200,
            prominence=0.6 * np.max(integrated)
        )
    else:
        thr2 = thr1

    return filtered, integrated, peaks, thr2, delay


def detect_spikes(leads, min_distance_ms=200):
    sig = leads["II"]

    filtered, _, peaks, _, delay = _pan_tompkins_core(
        sig,
        FS,
        min_distance_ms=min_distance_ms,
    )
    #print(peaks)
    refined = []

    search_back = 1
    search_fwd  = 1

    for p in peaks:
        p_corr = p - delay

        lo = max(0, p_corr - search_back)
        hi = min(len(filtered), p_corr + search_fwd)

        segment = filtered[lo:hi]
        idx = np.argmax(segment)

        refined.append(lo + idx)

    return np.array(refined, dtype=int)


def pan_tompkins_signal(leads):
    """Return decision variable + threshold for plotting."""
    sig = leads["II"]
    _, integrated, _, thr2, _ = _pan_tompkins_core(sig, FS)
    return integrated, thr2


# =========================================================
# 4. WINDOW EXTRACTION  —  multi-channel
# =========================================================

def extract_windows(signal_matrix, spike_indices):
    """Slice a symmetric window around each spike across all leads.

    Parameters
    ----------
    signal_matrix : np.ndarray  shape (n_leads, n_samples)
    spike_indices : np.ndarray  1-D array of spike sample positions

    Returns
    -------
    list[Beat]  — only spikes with a full window inside the signal are kept
    """
    n_samples = signal_matrix.shape[1]
    beats = []
    for idx in spike_indices:
        lo = idx - WINDOW_PRE
        hi = idx + WINDOW_POST
        if lo < 0 or hi > n_samples:
            continue
        window = signal_matrix[:, lo:hi].copy()   # (n_leads, window_size)
        beats.append(Beat(spike_idx=int(idx), window=window))
    return beats


# =========================================================
# 5. ANNOTATION  —  match spikes to marked beats
# =========================================================

def annotate_beats(beats, annotations, tol_ms=100.0):
    """Attach QRS / QT / label to each beat using embedded annotations.

    Matching strategy
    -----------------
    QRS Data row : [qrs_start_ms, qrs_end_ms, period_ms, qrs_duration_ms]
        Spike matches if it falls within [qrs_start - tol_ms, qrs_end + tol_ms].

    QT Data row  : [qt_start_ms, qt_end_ms, period_ms, qt_duration_ms]
        Same window-based matching.

    Label (0=normal / 1=arrhythmia / 2=extrasystole)
        Arrhythmia / Extrasystole rows: [period, duration, t_start, t_end]
        A beat is labelled if its spike_time falls in [t_start, t_end].
        Unmatched beats receive label 0 (normal).
    """
    qrs_data = np.array(annotations.get('QRS Data', []))
    qt_data  = np.array(annotations.get('QT Data',  []))

    arr_rows = annotations.get('Arrhythmia Data',   [])
    ext_rows = annotations.get('Extrasystole Data', [])
    arr_wins = [(r[0], r[1]) for r in arr_rows if len(r) >= 4]
    ext_wins = [(r[0], r[1]) for r in ext_rows if len(r) >= 4]

    for beat in beats:
        t = beat.spike_time

        if any(t_s <= t <= t_e for t_s, t_e in arr_wins):
            label = 1
        elif any(t_s <= t <= t_e for t_s, t_e in ext_wins):
            label = 2
        else:
            label = 0
        beat.annotate(label=label)

        if len(qrs_data):
            for row in qrs_data:
                if row[0] - tol_ms <= t <= row[1] + tol_ms:
                    beat.annotate(period=float(row[2]),
                                  qrs_duration=float(row[3]),
                                  qrs_start=float(row[0]))
                    break

        if len(qt_data):
            for row in qt_data:
                if row[0] - tol_ms <= t <= row[1] + tol_ms:
                    beat.annotate(qt_interval=float(row[3]),
                                  qt_start=float(row[0]))
                    break

    return beats


# =========================================================
# 6. FULL PIPELINE
# =========================================================

def extract_decision_windows(leads, beats):
    """Attach the Pan-Tompkins integrated signal window to each beat.

    The window spans [spike_idx - WINDOW_PRE, spike_idx + WINDOW_POST), matching
    the shape of beat.window.  Regions outside the signal are zero-padded.
    Values are in the same [0, 120] range as the detection variable.
    """
    sig = leads['II']
    _, integrated, _, _, _ = _pan_tompkins_core(sig, FS)
    n        = len(integrated)
    win_size = WINDOW_PRE + WINDOW_POST

    for beat in beats:
        lo   = beat.spike_idx - WINDOW_PRE
        hi   = beat.spike_idx + WINDOW_POST
        lo_c = max(0, lo)
        hi_c = min(n, hi)
        w    = np.zeros(win_size, dtype=np.float32)
        w[lo_c - lo : hi_c - lo] = integrated[lo_c : hi_c]
        beat.decision_window = w

    return beats


def process_study(filepath):
    """Load one study file and return annotated Beat objects + lead names.

    Returns
    -------
    beats      : list[Beat]
    lead_names : list[str]
    leads      : dict[str, np.ndarray]
    """
    leads, annotations = load_ecg(filepath)
    matrix, lead_names = leads_matrix(leads)
    spikes             = detect_spikes(leads)
    beats              = extract_windows(matrix, spikes)
    annotate_beats(beats, annotations)
    mark_noisy_beats(beats)
    recover_noisy_beats(beats, matrix)
    extract_context_windows(matrix, beats)
    extract_decision_windows(leads, beats)
    return beats, lead_names, leads


# =========================================================
# 7. DATASET UTILITIES
# =========================================================

def load_or_process_beats(folders, cache_dir='cache', force=False,
                          ecg_filename='ecg_data.txt'):
    """Return (annotated, unannotated, noisy), loading from pickle cache when available.

    Cache key: sorted folder basenames, so different patient sets produce
    different files.  Use force=True to reprocess and overwrite.
    """
    import pickle
    key        = '_'.join(sorted(os.path.basename(f) for f in folders))
    cache_path = os.path.join(cache_dir, f'beats_{key}.pkl')

    if not force and os.path.exists(cache_path):
        print(f'  [cache] loading beats from {cache_path}')
        with open(cache_path, 'rb') as fh:
            return pickle.load(fh)

    result = load_patient_beats(folders, ecg_filename)
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as fh:
        pickle.dump(result, fh, protocol=5)   # protocol 5: zero-copy for numpy buffers
    print(f'  [cache] saved beats to  {cache_path}')
    return result


def load_patient_beats(folders, ecg_filename="ecg_data.txt"):
    """Process a list of patient folders and partition beats by quality.

    Parameters
    ----------
    folders      : list[str]  paths to patient directories
    ecg_filename : str        name of the ECG file inside each folder

    Returns
    -------
    annotated   : list[Beat]  beats with qrs_duration and qt_interval filled
    unannotated : list[Beat]  beats missing qrs/qt annotation
    noisy       : list[Beat]  beats whose window overlaps another beat's window
                              (these may also appear in annotated/unannotated)
    """
    annotated   = []
    unannotated = []
    noisy       = []

    for folder in folders:
        filepath = os.path.join(folder, ecg_filename)
        if not os.path.isfile(filepath):
            continue
        try:
            beats, _, _ = process_study(filepath)
        except Exception:
            continue

        for beat in beats:
            if beat.noisy:
                noisy.append(beat)
            elif beat.qrs_duration is not None and beat.qt_interval is not None:
                annotated.append(beat)
            else:
                unannotated.append(beat)

    return annotated, unannotated, noisy


def beats_to_xy(beats):
    """Convert a list of beats into model-ready arrays.

    Parameters
    ----------
    beats : list[Beat]

    Returns
    -------
    X     : np.ndarray  shape (n_beats, n_leads, window_size)  float32
    y_qrs : np.ndarray  shape (n_beats,)  float32, NaN where annotation missing
    y_qt  : np.ndarray  shape (n_beats,)  float32, NaN where annotation missing
    """
    X     = np.stack([b.window for b in beats], axis=0).astype(np.float32)
    y_qrs = np.array(
        [b.qrs_duration if b.qrs_duration is not None else np.nan
         for b in beats],
        dtype=np.float32,
    )
    y_qt  = np.array(
        [b.qt_interval if b.qt_interval is not None else np.nan
         for b in beats],
        dtype=np.float32,
    )
    return X, y_qrs, y_qt


# =========================================================
# 8. MAIN  —  validation + feature tests
# =========================================================

def _sep(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


if __name__ == "__main__":
    import glob as _glob
    import matplotlib.pyplot as plt
    from plot import (plot_signal_windows, plot_beat, plot_beats,
                      plot_annotated_beat, plot_annotated_beats)

    DATA_DIR = "data"

    # ----------------------------------------------------------
    # C. load_patient_beats across all available folders
    # ----------------------------------------------------------
    _sep("C. load_patient_beats")

    folders = sorted(_glob.glob(os.path.join(DATA_DIR, "p*")))
    print(f"Patient folders found: {folders}")

    annotated, unannotated, noisy = load_patient_beats(folders)

    total = len(annotated) + len(unannotated) + len(noisy)
    print(f"Total beats loaded : {total}")
    print(f"  annotated        : {len(annotated)}")
    print(f"  unannotated      : {len(unannotated)}")
    print(f"  noisy            : {len(noisy)}")

    if annotated:
        print(f"\nSample annotated beat  : {annotated[0]}")
    if unannotated:
        print(f"Sample unannotated beat: {unannotated[0]}")
    if noisy:
        print(f"Sample noisy beat      : {noisy[0]}")

    # ----------------------------------------------------------
    # D. beats_to_xy
    # ----------------------------------------------------------
    _sep("D. beats_to_xy")

    if annotated:
        X, y_qrs, y_qt = beats_to_xy(annotated)
        print(f"annotated set  →  X:{X.shape}  y_qrs:{y_qrs.shape}  y_qt:{y_qt.shape}")
        print(f"  X dtype      : {X.dtype}")
        print(f"  y_qrs range  : [{np.nanmin(y_qrs):.1f}, {np.nanmax(y_qrs):.1f}] ms")
        print(f"  y_qt  range  : [{np.nanmin(y_qt):.1f},  {np.nanmax(y_qt):.1f}] ms")
        nan_qrs = int(np.isnan(y_qrs).sum())
        nan_qt  = int(np.isnan(y_qt).sum())
        print(f"  NaN y_qrs    : {nan_qrs}  |  NaN y_qt : {nan_qt}")

    if unannotated:
        X_u, y_qrs_u, y_qt_u = beats_to_xy(unannotated)
        print(f"unannotated set → X:{X_u.shape}  "
              f"NaN qrs:{np.isnan(y_qrs_u).sum()}  "
              f"NaN qt:{np.isnan(y_qt_u).sum()}")

    # ----------------------------------------------------------
    # E. Plots  (use first annotated beat's study for signal plots)
    # ----------------------------------------------------------
    _sep("E. Plots")

    plot_file = annotated[0].spike_idx if annotated else None
    _beats_plot, lead_names, leads = process_study(folders[0] + '/ecg_data.txt')

    sig_i = leads['I']
    decision, thr = pan_tompkins_signal(leads)
    plot_signal_windows(sig_i, _beats_plot, t_start=50000, t_end=70_000,
                        decision=decision, threshold=thr)
    plt.savefig("signal_windows.png", dpi=300, bbox_inches="tight")
    print("Saved signal_windows.png")

    plot_beats(_beats_plot[:9], lead_names)
    plt.savefig("beats.png", dpi=300, bbox_inches="tight")
    print("Saved beats.png")

    # ----------------------------------------------------------
    # F. Annotated beat plots
    # ----------------------------------------------------------
    _sep("F. Annotated beat plots")

    ann_beats = [b for b in _beats_plot if b.qrs_duration is not None]
    print(f"Beats with QRS annotation: {len(ann_beats)}")

    if ann_beats:
        plot_annotated_beat(ann_beats[0], lead_names)
        plt.savefig("annotated_beat.png", dpi=300, bbox_inches="tight")
        print("Saved annotated_beat.png")

      