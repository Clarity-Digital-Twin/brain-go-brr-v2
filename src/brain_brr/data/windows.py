"""Windowing utilities for EEG data."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


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