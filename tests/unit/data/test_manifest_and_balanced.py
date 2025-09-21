from pathlib import Path

import numpy as np
import torch

from src.brain_brr.data.cache_utils import scan_existing_cache
from src.brain_brr.data.datasets import BalancedSeizureDataset


def _make_npz(path: Path, windows: np.ndarray, labels: np.ndarray) -> None:
    np.savez_compressed(path, windows=windows.astype(np.float32), labels=labels.astype(np.float32))


def test_scan_existing_cache_and_balanced_dataset(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    windows_a = np.zeros((5, 19, 20), dtype=np.float32)
    labels_a = np.zeros((5, 20), dtype=np.float32)
    labels_a[1] = 1.0
    labels_a[2, :10] = 1.0
    labels_a[3, :1] = 1.0
    _make_npz(cache_dir / "a_windows.npz", windows_a, labels_a)

    windows_b = np.zeros((4, 19, 20), dtype=np.float32)
    labels_b = np.zeros((4, 20), dtype=np.float32)
    labels_b[0, :] = 1.0
    labels_b[1, :5] = 1.0
    _make_npz(cache_dir / "b_windows.npz", windows_b, labels_b)

    manifest = scan_existing_cache(cache_dir)
    n_partial = len(manifest["partial_seizure"])
    n_full = len(manifest["full_seizure"])
    n_none = len(manifest["no_seizure"])

    assert n_partial > 0
    assert n_full > 0
    assert n_none > 0

    ds = BalancedSeizureDataset(cache_dir, full_ratio=0.3, background_ratio=2.5, seed=0)
    expected = n_partial + min(int(0.3 * n_partial), n_full) + min(int(2.5 * n_partial), n_none)
    assert len(ds) == expected

    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] == 19
    assert x.shape[1] == 20
    assert y.shape[0] == 20
