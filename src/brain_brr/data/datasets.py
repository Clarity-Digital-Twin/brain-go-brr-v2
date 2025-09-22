"""PyTorch dataset implementations for EEG data."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import Dataset

from src.brain_brr import constants
from src.brain_brr.data.cache_utils import scan_existing_cache
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
        allow_on_demand: bool = True,
    ) -> None:
        self.edf_files = edf_files
        self.label_files = label_files
        self.cache_dir = cache_dir
        self.transform = transform
        self.allow_on_demand = bool(allow_on_demand)

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Build index mapping: (file_idx, window_idx) for each dataset index
        self._index_map: list[tuple[int, int]] = []
        self._file_window_counts: list[int] = []
        self._has_labels = label_files is not None

        # Pre-compute or load window counts for each file
        print(f"[DATA] Building dataset index for {len(self.edf_files)} files...", flush=True)
        for i, edf_path in enumerate(self.edf_files):
            if i % 10 == 0:
                print(
                    f"[DATA] Processing file {i + 1}/{len(self.edf_files)}: {edf_path.name}",
                    flush=True,
                )
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
                print(f"[DATA] Building cache for {edf_path.name}...", flush=True)
                windows_arr, labels_arr = self._process_file(edf_path, i)
                n_windows = windows_arr.shape[0]
                if cache_path is not None:
                    if labels_arr is not None:
                        np.savez_compressed(cache_path, windows=windows_arr, labels=labels_arr)
                    else:
                        np.savez_compressed(cache_path, windows=windows_arr)

            self._file_window_counts.append(n_windows)
            for w_idx in range(n_windows):
                self._index_map.append((i, w_idx))

        print(f"[DATA] Dataset ready! Total windows: {len(self._index_map)}", flush=True)

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
        # Ensure float32 before caching
        windows = windows.astype(np.float32, copy=False)
        if window_labels is not None:
            window_labels = window_labels.astype(np.float32, copy=False)
        return windows, window_labels

    def _load_labels(self, label_path: Path, n_samples: int) -> npt.NDArray[np.float32]:
        """Load labels and return binary mask at 256 Hz of length n_samples.

        This is a placeholder; format-specific loaders can be added later.
        """
        # CSV_BI (Temple/TUSZ) annotations
        if label_path.suffix.lower() == ".csv" and label_path.exists():
            _duration_s, events = parse_tusz_csv(label_path)
            # Convert to binary mask aligned to requested n_samples @ 256 Hz
            # NOTE: events_to_binary_mask expects duration in SECONDS, not samples!
            duration_sec = n_samples / constants.SAMPLING_RATE
            return events_to_binary_mask(events, duration_sec, fs=constants.SAMPLING_RATE)

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
            if not self.allow_on_demand:
                raise RuntimeError(
                    f"Cache missing for {edf_path.name} at {cache_path}; on-demand disabled"
                )
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


class BalancedSeizureDataset(Dataset):
    """Dataset implementing SeizureTransformer-style balancing using a manifest.

    Uses all partial-seizure windows and adds 0.3x full-seizure and 2.5x no-seizure.
    """

    def __init__(
        self,
        cache_dir: Path,
        *,
        full_ratio: float = 0.3,
        background_ratio: float = 2.5,
        seed: int | None = 42,
        ensure_manifest: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        if ensure_manifest and not manifest_path.exists():
            _ = scan_existing_cache(self.cache_dir)

        with manifest_path.open() as f:
            manifest = json.load(f)

        partial: list[dict] = list(manifest.get("partial_seizure", []))
        full: list[dict] = list(manifest.get("full_seizure", []))
        no_seizure: list[dict] = list(manifest.get("no_seizure", []))

        # Validate we have seizures to work with
        if not partial:
            raise ValueError(
                f"No partial seizure windows found in manifest! "
                f"Full: {len(full)}, No-seizure: {len(no_seizure)}"
            )

        rng = np.random.default_rng(seed)

        indices: list[tuple[Path, int]] = []
        missing_ref_count = 0

        # Add ALL partial seizure windows (most informative)
        for item in partial:
            # Resolve relative path from manifest to absolute
            cache_file = self.cache_dir / item["cache_file"]
            if cache_file.exists():
                indices.append((cache_file, int(item["window_idx"])))
            else:
                missing_ref_count += 1

        # Add 0.3x full seizure windows
        n_full = int(full_ratio * len(partial))
        if full and n_full > 0:
            selected_indices = rng.choice(len(full), size=min(n_full, len(full)), replace=False)
            for i in selected_indices:
                item = full[i]
                cache_file = self.cache_dir / item["cache_file"]
                if cache_file.exists():
                    indices.append((cache_file, int(item["window_idx"])))
                else:
                    missing_ref_count += 1

        # Add 2.5x no-seizure windows
        n_bg = int(background_ratio * len(partial))
        if no_seizure and n_bg > 0:
            selected_indices = rng.choice(
                len(no_seizure), size=min(n_bg, len(no_seizure)), replace=False
            )
            for i in selected_indices:
                item = no_seizure[i]
                cache_file = self.cache_dir / item["cache_file"]
                if cache_file.exists():
                    indices.append((cache_file, int(item["window_idx"])))
                else:
                    missing_ref_count += 1

        # Shuffle using numpy's RNG for consistency
        indices_array = np.array(indices, dtype=object)
        rng.shuffle(indices_array)
        self._entries: list[tuple[Path, int]] = indices_array.tolist()

        # Log dataset composition
        n_partial_used = len(partial)
        n_full_used = min(n_full, len(full)) if full else 0
        n_bg_used = min(n_bg, len(no_seizure)) if no_seizure else 0

        # Store seizure statistics for fast access (avoid sampling 1000 windows!)
        self._n_seizure_windows = n_partial_used + n_full_used
        self._n_total_windows = len(self._entries)
        self._seizure_ratio = (
            self._n_seizure_windows / self._n_total_windows if self._n_total_windows > 0 else 0.0
        )

        print(
            f"[BalancedSeizureDataset] Created with {len(self._entries)} windows:\n"
            f"  - {n_partial_used} partial seizure (100% of available)\n"
            f"  - {n_full_used} full seizure ({n_full_used / n_partial_used:.1%} of partial)\n"
            f"  - {n_bg_used} no-seizure ({n_bg_used / n_partial_used:.1%} of partial)"
        )
        if missing_ref_count > 0:
            print(
                f"[WARNING] Skipped {missing_ref_count} manifest entries referencing missing cache files"
            )

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def seizure_ratio(self) -> float:
        """Return the proportion of windows containing seizures.

        This avoids needing to sample 1000 windows to calculate class weights!
        """
        return self._seizure_ratio

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_file, w_idx = self._entries[idx]
        with np.load(cache_file) as data:
            window = data["windows"][w_idx].astype(np.float32)
            if "labels" in data:
                label = data["labels"][w_idx].astype(np.float32)
            else:
                label = np.zeros((window.shape[-1],), dtype=np.float32)
        return torch.from_numpy(window), torch.from_numpy(label)
