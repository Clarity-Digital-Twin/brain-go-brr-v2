"""EEG data I/O, preprocessing, windowing, and dataset utilities (Phase 1).

Design goals:
- Keep imports for heavy libs (mne, scipy) inside functions to avoid mypy
  missing-stub noise and to ease testing/mocking.
- Provide pure, deterministic functions suitable for unit tests.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from . import constants


# --- Internal helper to enable monkeypatching in tests ---
def _read_raw_edf(file_path: Path) -> Any:  # pragma: no cover - thin wrapper
    """Read EDF via MNE with preload.

    Split out for easy monkeypatching in tests (avoids depending on MNE).
    """
    import mne  # type: ignore[import-untyped]  # MNE has no stubs

    return mne.io.read_raw_edf(file_path, preload=True, verbose="WARNING")


def _repair_edf_header_inplace(file_path: Path) -> bool:
    """Repair common EDF header issues in-place (dates with wrong separators).

    Returns True if repair was needed and successful, False otherwise.
    """
    try:
        with open(file_path, "r+b") as f:
            # Check header for date format issues (byte 168-175: DD.MM.YY)
            f.seek(168)
            date_bytes = f.read(8)

            # Fix colons to periods in date field
            if b":" in date_bytes:
                fixed_date = date_bytes.replace(b":", b".")
                f.seek(168)
                f.write(fixed_date)
                return True
    except Exception:
        pass
    return False


def load_edf_file(
    file_path: Path,
    target_channels: list[str] | None = None,
    apply_montage: bool = True,
    channel_synonyms: dict[str, str] | None = None,
) -> tuple[npt.NDArray[np.float32], float]:
    """Load an EDF and return signals in canonical order and original fs.

    Args:
        file_path: Path to EDF file
        target_channels: Desired channel order; defaults to 10-20 list
        apply_montage: If True, set standard_1020 montage (best-effort)
        channel_synonyms: Mapping of alternate names to canonical names

    Returns:
        data: (n_channels, n_samples) float32 in microvolts (µV)
        fs: Original sampling frequency (Hz)

    Raises:
        ValueError: If required channels are missing after synonym remapping

    Note:
        MNE is generally permissive with malformed headers (e.g., TUSZ files
        with colon date separators). If MNE fails on header issues, we attempt
        a repair on a temp copy and retry.
    """
    if target_channels is None:
        target_channels = constants.CHANNEL_NAMES_10_20
    if channel_synonyms is None:
        channel_synonyms = constants.CHANNEL_SYNONYMS

    # First attempt with MNE (already permissive)
    try:
        raw = _read_raw_edf(file_path)
    except Exception as e:
        # If MNE fails with header/date error, try repair on temp copy
        if "startdate" in str(e) or "header" in str(e).lower():
            with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                shutil.copy2(file_path, tmp_path)

                try:
                    _repair_edf_header_inplace(tmp_path)
                    raw = _read_raw_edf(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
        else:
            raise

    # Optional montage alignment (best-effort)
    if apply_montage:
        with contextlib.suppress(Exception):
            raw.set_montage("standard_1020", on_missing="ignore")

    # Rename channels according to synonyms mapping (alt → canonical)
    try:
        rename_map = {alt: canon for alt, canon in channel_synonyms.items() if alt in raw.ch_names}
        if rename_map:
            raw.rename_channels(rename_map)
    except Exception:
        # If rename fails (non-MNE stub), ignore; tests may not require it
        pass

    # Validate required channels presence
    available = [ch for ch in target_channels if ch in raw.ch_names]
    if len(available) != len(target_channels):
        missing = set(target_channels) - set(available)
        raise ValueError(f"Missing required channels: {sorted(missing)}")

    # Reorder to canonical order
    try:
        raw.pick_channels(target_channels, ordered=True)
    except Exception:
        # Fallback: manual reindex if the object doesn't support pick_channels
        idx = [raw.ch_names.index(ch) for ch in target_channels]
        data_v = raw.get_data()[idx]
        fs_v = float(getattr(raw, "info", {}).get("sfreq", getattr(raw, "sfreq", 0.0)))
        data_uv = (data_v * 1e6).astype(np.float32)
        return data_uv, fs_v

    # Extract data and sampling rate
    data = raw.get_data()  # (n_channels, n_samples) in Volts (MNE default)
    fs = float(getattr(raw, "info", {}).get("sfreq", getattr(raw, "sfreq", 0.0)))

    # Standardize to microvolts (µV)
    data_uv = (data * 1e6).astype(np.float32)
    return data_uv, fs


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

    # Sanitize NaNs / Infs and cast
    x_clean: npt.NDArray[np.float32] = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32
    )
    return x_clean


def extract_windows(
    data: npt.NDArray[np.floating],
    window_size: int,
    stride: int,
    labels: npt.NDArray[np.floating] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None, dict[str, list[int]]]:
    """Extract sliding windows from continuous data.

    Returns:
        windows: (n_windows, n_channels, window_size)
        window_labels: (n_windows, window_size) or None
        metadata: {"start_samples": list[int]}
    """
    x = np.asarray(data)
    n_channels, n_samples = x.shape
    if n_samples < window_size:
        return (
            np.zeros((0, n_channels, window_size), dtype=np.float32),
            None if labels is None else np.zeros((0, window_size), dtype=np.float32),
            {"start_samples": []},
        )

    n_windows = (n_samples - window_size) // stride + 1
    windows = np.zeros((n_windows, n_channels, window_size), dtype=np.float32)
    starts: list[int] = []

    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        windows[i] = x[:, start:end]
        starts.append(start)

    window_labels: npt.NDArray[np.float32] | None = None
    if labels is not None:
        y = np.asarray(labels).astype(np.float32).reshape(-1)
        if y.shape[0] < n_samples:
            # pad with zeros if label sequence shorter than data
            y = np.pad(y, (0, n_samples - y.shape[0]), mode="constant")
        window_labels = np.zeros((n_windows, window_size), dtype=np.float32)
        for i in range(n_windows):
            start = i * stride
            end = start + window_size
            window_labels[i] = y[start:end]

    metadata = {"start_samples": starts}
    return windows, window_labels, metadata


class EEGWindowDataset(torch.utils.data.Dataset):
    """PyTorch dataset for windowed EEG.

    Phase 1: simple materialization in memory (OK for unit tests and small sets).
    """

    def __init__(
        self,
        edf_files: list[Path],
        label_files: list[Path] | None = None,
        cache_dir: Path | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        self.edf_files = edf_files
        self.label_files = label_files
        self.cache_dir = cache_dir
        self.transform = transform

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Materialize windows
        self._windows: list[npt.NDArray[np.float32]] = []
        self._labels: list[npt.NDArray[np.float32]] = []

        for i, edf_path in enumerate(self.edf_files):
            cache_path = None
            if self.cache_dir is not None:
                cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"

            windows_arr: npt.NDArray[np.float32]
            labels_arr: npt.NDArray[np.float32] | None

            if cache_path is not None and cache_path.exists():
                cached = np.load(cache_path)
                windows_arr = cached["windows"].astype(np.float32)
                labels_arr = cached.get("labels", None)
            else:
                windows_arr, labels_arr = self._process_file(edf_path, i)
                if cache_path is not None:
                    if labels_arr is not None:
                        np.savez_compressed(cache_path, windows=windows_arr, labels=labels_arr)
                    else:
                        np.savez_compressed(cache_path, windows=windows_arr)

            # Store
            for w_idx in range(windows_arr.shape[0]):
                self._windows.append(windows_arr[w_idx])
                if labels_arr is not None:
                    self._labels.append(labels_arr[w_idx])

    def _process_file(
        self, edf_path: Path, file_idx: int
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:
        # Load & preprocess
        data_uv, fs = load_edf_file(edf_path)
        data_proc = preprocess_recording(data_uv, fs_original=fs)

        # Labels (optional)
        labels = None
        if self.label_files is not None and file_idx < len(self.label_files):
            label_path = self.label_files[file_idx]
            labels = self._load_labels(label_path, n_samples=data_proc.shape[1])

        # Windowing
        windows, window_labels, _ = extract_windows(
            data_proc,
            window_size=constants.WINDOW_SAMPLES,
            stride=constants.STRIDE_SAMPLES,
            labels=labels,
        )
        return windows, window_labels

    def _load_labels(self, label_path: Path, n_samples: int) -> npt.NDArray[np.float32]:
        """Load labels and return binary mask at 256 Hz of length n_samples.

        This is a placeholder; format-specific loaders can be added later.
        """
        # Simple baseline: if .npy present, load; else return zeros
        if label_path.suffix == ".npy" and label_path.exists():
            arr = np.load(label_path)
            vec = np.asarray(arr).reshape(-1).astype(np.float32)
            if vec.shape[0] < n_samples:
                vec = np.pad(vec, (0, n_samples - vec.shape[0]), mode="constant")
            else:
                vec = vec[:n_samples]
            return vec
        return np.zeros((n_samples,), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return window and label as tuple. Always returns tuple for consistency.

        When no labels exist, returns zero tensor of correct shape as label.
        This ensures train_epoch always gets (window, label) tuple.
        """
        window = torch.from_numpy(self._windows[idx])
        if self.transform is not None:
            window = self.transform(window)

        if self._labels:
            label = torch.from_numpy(self._labels[idx])
        else:
            # ALWAYS return tuple with zero labels when none exist
            # Shape matches window's time dimension for per-timestep labels
            label = torch.zeros(window.shape[-1], dtype=torch.float32)

        return window, label
