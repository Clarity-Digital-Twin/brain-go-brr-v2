"""Disk-backed storage for validation timelines with zero RAM accumulation."""

import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

import numpy as np
import torch


class RecordingStorage:
    """Disk-backed storage for per-recording validation data.

    Guarantees:
    - Zero RAM accumulation during writes (9MB transient per recording)
    - Memory-mapped reads for minimal overhead
    - Automatic cleanup via context manager

    Memory Contract:
    - write_recording(): 9MB transient (freed immediately)
    - iter_recordings(): O(1) per iteration (memory-mapped)
    - get_all_concatenated(): 34GB (caller must free explicitly)
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize disk-backed storage.

        Args:
            cache_dir: Directory for .npy shards. If None, uses temp directory.
        """
        self._temp_dir: tempfile.TemporaryDirectory[str] | None
        if cache_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="val_")
            self.cache_dir = Path(self._temp_dir.name)
        else:
            self._temp_dir = None
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.recording_ids: list[str] = []

    def write_recording(
        self,
        file_id: str,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Write one recording to disk (9MB transient, 0GB resident).

        Memory Safety:
        - Tensor already on CPU (from validation loop .cpu() call)
        - .numpy() is zero-copy if tensor is CPU + contiguous
        - np.save() writes to disk, buffer freed after return

        Args:
            file_id: Unique identifier
            probs: Probability timeline (1D, CPU, float32)
            labels: Label timeline (1D, CPU, float32)
        """
        if not probs.is_contiguous():
            probs = probs.contiguous()
        if not labels.is_contiguous():
            labels = labels.contiguous()

        probs_np = probs.numpy()
        labels_np = labels.numpy()

        probs_path = self.cache_dir / f"{file_id}_probs.npy"
        labels_path = self.cache_dir / f"{file_id}_labels.npy"

        np.save(probs_path, probs_np)
        np.save(labels_path, labels_np)

        self.recording_ids.append(file_id)

        del probs_np, labels_np

    def iter_recordings(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Iterate over recordings with memory-mapping (O(1) memory per iteration).

        Yields:
            (probs, labels) memory-mapped from disk (lazy loading)
        """
        for file_id in self.recording_ids:
            probs = np.load(
                self.cache_dir / f"{file_id}_probs.npy",
                mmap_mode="r",
            )
            labels = np.load(
                self.cache_dir / f"{file_id}_labels.npy",
                mmap_mode="r",
            )
            yield probs, labels

    def get_all_concatenated(self) -> tuple[np.ndarray, np.ndarray]:
        """Load and concatenate all recordings (34GB allocation, single-pass).

        Uses pre-allocation + direct copy to avoid double-buffering.
        Peak memory: 34GB (not 68GB from list+concat pattern).

        WARNING: This allocates 34GB in RAM. Caller MUST free explicitly:
            probs, labels = storage.get_all_concatenated()
            # ... use data ...
            del probs, labels
            import gc; gc.collect()

        Returns:
            (all_probs, all_labels) as contiguous numpy arrays
        """
        total_samples = 0
        for probs, _ in self.iter_recordings():
            total_samples += len(probs)

        probs_all = np.empty(total_samples, dtype=np.float32)
        labels_all = np.empty(total_samples, dtype=np.float32)

        offset = 0
        for probs, labels in self.iter_recordings():
            n = len(probs)
            probs_all[offset : offset + n] = probs
            labels_all[offset : offset + n] = labels
            offset += n

        return probs_all, labels_all

    def get_all_as_torch_tensors(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Get all recordings as torch tensors (TRUE zero-copy from mmap).

        Uses copy-on-write memory mapping for safe zero-copy tensor creation.
        Peak memory: <10MB (just tensor objects, not data).

        Memory-mapping mode "c" (copy-on-write):
        - Array is memory-mapped and writeable (satisfies torch.from_numpy)
        - Any modifications create a private copy (won't happen - FA sweep is read-only)
        - Original file is never modified
        - Zero-copy as long as no writes occur

        Returns:
            (probs_list, labels_list) where each tensor shares mmap memory
        """
        probs_list = []
        labels_list = []

        for file_id in self.recording_ids:
            probs_np = np.load(
                self.cache_dir / f"{file_id}_probs.npy",
                mmap_mode="c",
            )
            labels_np = np.load(
                self.cache_dir / f"{file_id}_labels.npy",
                mmap_mode="c",
            )

            probs_tensor = torch.from_numpy(probs_np)
            labels_tensor = torch.from_numpy(labels_np)

            probs_list.append(probs_tensor)
            labels_list.append(labels_tensor)

        return probs_list, labels_list

    def cleanup(self) -> None:
        """Delete all .npy files and temp directory."""
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
        else:
            import shutil

            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def __enter__(self) -> "RecordingStorage":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.cleanup()
