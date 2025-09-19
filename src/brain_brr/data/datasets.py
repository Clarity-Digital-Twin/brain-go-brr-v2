"""PyTorch dataset implementations for EEG data."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from src.brain_brr import constants
from src.brain_brr.data.io import events_to_binary_mask, load_edf_file, parse_tusz_csv
from src.brain_brr.data.preprocess import preprocess_recording
from src.brain_brr.data.windows import extract_windows


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
                try:
                    with np.load(cache_path) as cached:
                        n_windows = cached["windows"].shape[0]
                except Exception:
                    windows_arr, labels_arr = self._process_file(edf_path, i)
                    n_windows = windows_arr.shape[0]
                    if cache_path is not None:
                        if labels_arr is not None:
                            np.savez_compressed(cache_path, windows=windows_arr, labels=labels_arr)
                        else:
                            np.savez_compressed(cache_path, windows=windows_arr)
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