import json
from pathlib import Path

import numpy as np

from src.brain_brr.data.cache_utils import scan_existing_cache, validate_manifest


def _save_npy_cache(
    cache_dir: Path,
    stem: str,
    windows: np.ndarray,
    labels: np.ndarray | None,
) -> None:
    """Create a pair of *_data.npy / *_labels.npy files matching production cache format."""
    np.save(cache_dir / f"{stem}_data.npy", windows.astype(np.float32))
    if labels is None:
        labels = np.zeros((0,), dtype=np.float32)
    np.save(cache_dir / f"{stem}_labels.npy", labels.astype(np.float32))


def test_validate_manifest_ok(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    w = np.zeros((3, 19, 10), dtype=np.float32)
    y = np.zeros((3, 10), dtype=np.float32)
    y[1, :5] = 1.0
    _save_npy_cache(cache_dir, "a", w, y)
    _save_npy_cache(cache_dir, "b", w, y)

    manifest = scan_existing_cache(cache_dir)
    assert validate_manifest(cache_dir, manifest) is True


def test_validate_manifest_empty_false(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # No cache files → scan creates empty manifest
    manifest = scan_existing_cache(cache_dir)
    assert validate_manifest(cache_dir, manifest) is False


def test_validate_manifest_missing_refs_false(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    w = np.zeros((2, 19, 10), dtype=np.float32)
    y = np.zeros((2, 10), dtype=np.float32)
    y[0, :] = 1.0
    _save_npy_cache(cache_dir, "keep", w, y)

    # Build valid manifest then corrupt references
    manifest = scan_existing_cache(cache_dir)
    # Point all entries to a missing file
    for k in ("partial_seizure", "full_seizure", "no_seizure"):
        for item in manifest.get(k, []):
            item["cache_file"] = "missing_data.npy"
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    assert validate_manifest(cache_dir, manifest) is False
