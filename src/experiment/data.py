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

    # Canonicalize channel names to 10-20 using robust normalization
    try:
        # Build uppercase → canonical map (includes synonyms like T7→T3)
        upper_to_canon: dict[str, str] = {c.upper(): c for c in target_channels}
        for alt, canon in channel_synonyms.items():
            upper_to_canon[alt.upper()] = canon

        def _to_canonical(name: str) -> str | None:
            s = name.strip().upper()

            # Filter out known non-EEG channels early
            non_eeg_prefixes = (
                "EKG",
                "ECG",
                "EOG",
                "EMG",
                "RESP",
                "PHOTIC",
                "IBI",
                "BURSTS",
                "SUPPR",
                "LOC",
                "ROC",
            )
            if any(s.startswith(p) for p in non_eeg_prefixes):
                return None

            # Handle EEG prefix
            if s.startswith("EEG "):
                s = s[4:]

            # Remove reference suffixes
            for suf in ("-REF", "-LE", "-AR", "-AVG", "-DC"):
                if s.endswith(suf):
                    s = s[: -len(suf)]
                    break

            return upper_to_canon.get(s)

        rename_map_norm: dict[str, str] = {}
        for ch in list(raw.ch_names):
            canonical_name = _to_canonical(ch)
            if canonical_name is not None and canonical_name != ch:
                rename_map_norm[ch] = canonical_name
        if rename_map_norm:
            raw.rename_channels(rename_map_norm)
    except Exception:
        pass

    # Validate required channels presence
    available = [ch for ch in target_channels if ch in raw.ch_names]
    if len(available) != len(target_channels):
        missing = set(target_channels) - set(available)
        raise ValueError(f"Missing required channels: {sorted(missing)}")

    # Pick channels by name (not position) then reorder
    try:
        # First, check which required channels are available
        available_required = [ch for ch in target_channels if ch in raw.ch_names]
        if len(available_required) != len(target_channels):
            missing = set(target_channels) - set(available_required)
            raise ValueError(f"Missing required channels: {missing}")

        # Pick channels (preserves their order in the file)
        raw.pick(target_channels)
        # Now reorder to canonical order
        raw.reorder_channels(target_channels)
    except Exception as e:
        # Fallback: manual reindex if the object doesn't support pick
        if "pick" in str(e) or "reorder_channels" in str(e):
            idx = [raw.ch_names.index(ch) for ch in target_channels]
            data_v = raw.get_data()[idx]
        else:
            raise
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

    Memory-efficient: loads windows on-demand from cache or computes them.
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

        # Build index mapping: (file_idx, window_idx) for each dataset index
        self._index_map: list[tuple[int, int]] = []
        self._file_window_counts: list[int] = []
        self._has_labels = label_files is not None

        # Pre-compute or load window counts for each file
        for i, edf_path in enumerate(self.edf_files):
            cache_path = None
            if self.cache_dir is not None:
                cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"

            if cache_path is not None and cache_path.exists():
                # Just load shape info from cache
                with np.load(cache_path) as cached:
                    n_windows = cached["windows"].shape[0]
            else:
                # Process file to create cache (but don't keep in memory)
                windows_arr, labels_arr = self._process_file(edf_path, i)
                n_windows = windows_arr.shape[0]
                if cache_path is not None:
                    if labels_arr is not None:
                        np.savez_compressed(cache_path, windows=windows_arr, labels=labels_arr)
                    else:
                        np.savez_compressed(cache_path, windows=windows_arr)

            # Build index map
            self._file_window_counts.append(n_windows)
            for w_idx in range(n_windows):
                self._index_map.append((i, w_idx))

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
        # CSV_BI (Temple/TUSZ) annotations
        if label_path.suffix.lower() == ".csv" and label_path.exists():
            _duration_s, events = parse_tusz_csv(label_path)
            # Convert to binary mask aligned to requested n_samples @ 256 Hz
            return events_to_binary_mask(events, n_samples, fs=constants.SAMPLING_RATE)

        # Simple baseline: if .npy present, load; else return zeros
        if label_path.suffix.lower() == ".npy" and label_path.exists():
            arr = np.load(label_path)
            vec = np.asarray(arr).reshape(-1).astype(np.float32)
            if vec.shape[0] < n_samples:
                vec = np.pad(vec, (0, n_samples - vec.shape[0]), mode="constant")
            else:
                vec = vec[:n_samples]
            return vec

        # Fallback: no labels
        return np.zeros((n_samples,), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return window and label as tuple. Always returns tuple for consistency.

        Loads data on-demand from cache or computes if needed.
        When no labels exist, returns zero tensor of correct shape as label.
        This ensures train_epoch always gets (window, label) tuple.
        """
        file_idx, window_idx = self._index_map[idx]
        edf_path = self.edf_files[file_idx]

        # Load from cache or compute
        cache_path = None
        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{edf_path.stem}_windows.npz"

        if cache_path is not None and cache_path.exists():
            # Load specific window from cache
            with np.load(cache_path) as cached:
                window = cached["windows"][window_idx].astype(np.float32)
                if "labels" in cached and cached["labels"] is not None:
                    label = cached["labels"][window_idx].astype(np.float32)
                else:
                    label = None
        else:
            # Compute on-the-fly if no cache
            windows_arr, labels_arr = self._process_file(edf_path, file_idx)
            window = windows_arr[window_idx]
            label = labels_arr[window_idx] if labels_arr is not None else None

        # Convert to tensors
        window_tensor = torch.from_numpy(window)
        if self.transform is not None:
            window_tensor = self.transform(window_tensor)

        if label is not None:
            label_tensor = torch.from_numpy(label)
        else:
            # ALWAYS return tuple with zero labels when none exist
            # Shape matches window's time dimension for per-timestep labels
            label_tensor = torch.zeros(window_tensor.shape[-1], dtype=torch.float32)

        return window_tensor, label_tensor


def parse_tusz_csv(csv_path: Path) -> tuple[float, list[tuple[float, float, str]]]:
    """Parse a TUSZ/Temple CSV_BI annotation file.

    Returns:
        duration_s: Total recording duration in seconds (from header, or inferred)
        events: List of (start_s, stop_s, label) tuples across all channels

    Notes:
        - We treat labels "seiz" and "cpsz" as seizures; others are ignored.
        - Multi-channel rows are aggregated by union (any-channel seizure → positive).
    """
    duration: float = 0.0
    events: list[tuple[float, float, str]] = []

    if not csv_path.exists():
        return duration, events

    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Header lines like: "# duration = 301.00 secs"
                if line.lower().startswith("# duration") and "=" in line:
                    try:
                        rhs = line.split("=", 1)[1]
                        duration_str = rhs.replace("secs", "").strip()
                        duration = float(duration_str)
                    except Exception:
                        # Keep default 0.0 if parsing fails
                        pass
                continue
            # Skip column header
            if line.lower().startswith("channel,"):
                continue

            parts = [p.strip() for p in line.split(",")]
            # Expected: channel, start_time, stop_time, label, confidence
            if len(parts) < 4:
                continue
            try:
                start = float(parts[1])
                stop = float(parts[2])
            except Exception:
                continue
            label = parts[3].strip().lower()
            # Normalize label: treat any seizure-coded label as positive.
            # TUSZ commonly uses codes ending with 'sz' (e.g., cpsz, fnsz, gnsz, tcsz, tnsz, absz, mysz),
            # and sometimes the generic 'seiz'. We exclude explicit background 'bckg'.
            # Also handle 'atnz' (atonic seizure) and any label containing 'seiz'.
            if label != "bckg" and (
                label == "seiz" or label.endswith("sz") or label == "atnz" or "seiz" in label
            ):
                events.append((start, stop, label))

    # If duration not present, infer from max stop
    if duration <= 0.0 and events:
        duration = max(stop for _, stop, _ in events)
    return duration, events


def events_to_binary_mask(
    events: list[tuple[float, float, str]],
    n_samples: int,
    fs: int = constants.SAMPLING_RATE,
) -> npt.NDArray[np.float32]:
    """Convert seizure events to a binary mask of length n_samples.

    Args:
        events: List of (start_s, stop_s, label)
        n_samples: Desired length of output vector
        fs: Sampling rate (Hz), default 256

    Returns:
        mask: (n_samples,) float32 with 1.0 for seizure, 0.0 otherwise
    """
    mask = np.zeros((n_samples,), dtype=np.float32)
    if not events or n_samples <= 0:
        return mask

    for start_s, stop_s, _ in events:
        # Convert to indices and clamp to valid range
        s_idx = max(0, round(start_s * fs))
        e_idx = max(s_idx, round(stop_s * fs))
        if s_idx >= n_samples:
            continue
        e_idx = min(e_idx, n_samples)
        if e_idx > s_idx:
            mask[s_idx:e_idx] = 1.0
    return mask
