"""EEG signal preprocessing utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.brain_brr import constants


def preprocess_recording(
    data: npt.NDArray[np.floating],
    fs_original: float,
    target_fs: int = constants.SAMPLING_RATE,
    bandpass: tuple[float, float] = (0.5, 120.0),
    notch_freq: int = 60,
) -> npt.NDArray[np.float32]:
    """Preprocess EEG recording.

    Steps:
      1) Resample to target_fs (scipy.signal.resample)
      2) Bandpass Butterworth (order=3) via lfilter
      3) Notch (iirnotch)
      4) Per-channel z-score normalization

    Returns float32 array (n_channels, n_samples_new), finite-only (NaN/Inf → 0).
    """
    from scipy.signal import (  # type: ignore[import-untyped]  # local import
        butter,
        iirnotch,
        lfilter,
        resample,
    )

    x = np.asarray(data, dtype=np.float64)  # work in float64 for filtering stability
    _n_ch, n_samp = x.shape

    # 1) Resample if required
    if float(fs_original) != float(target_fs) and n_samp > 0:
        n_new = round(n_samp * float(target_fs) / float(fs_original))
        x = resample(x, n_new, axis=1)

    fs = float(target_fs)

    # 2) Bandpass filter
    low, high = bandpass
    nyq = fs / 2.0
    b_bp, a_bp = butter(3, [low / nyq, high / nyq], btype="band")
    x = lfilter(b_bp, a_bp, x, axis=1)

    # 3) Notch filter (powerline)
    try:
        b_notch, a_notch = iirnotch(notch_freq, Q=30, fs=fs)
        x = lfilter(b_notch, a_notch, x, axis=1)
    except Exception:
        # If iirnotch with fs not available, compute normalized w0
        w0 = notch_freq / nyq
        b_notch, a_notch = iirnotch(w0, Q=30)
        x = lfilter(b_notch, a_notch, x, axis=1)

    # 4) Per-channel z-score
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True)
    x = (x - mean) / (std + 1e-8)

    # CRITICAL: Clip outliers to prevent infinities during training
    # EEG data can have extreme artifacts (>100 sigma) that cause numerical issues
    x = np.clip(x, -10.0, 10.0)  # Clip to ±10 standard deviations

    # Sanitize NaNs / Infs and cast
    x_clean: npt.NDArray[np.float32] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )
    return x_clean
