from pathlib import Path

import json
import numpy as np

from src.brain_brr.data.cache_utils import scan_existing_cache, validate_manifest


def _save_npz(path: Path, windows: np.ndarray, labels: np.ndarray | None = None) -> None:
    if labels is not None:
        np.savez_compressed(path, windows=windows.astype(np.float32), labels=labels.astype(np.float32))
    else:
        np.savez_compressed(path, windows=windows.astype(np.float32))


def test_validate_manifest_ok(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    w = np.zeros((3, 19, 10), dtype=np.float32)
    y = np.zeros((3, 10), dtype=np.float32)
    y[1, :5] = 1.0
    _save_npz(cache_dir / "a_windows.npz", w, y)
    _save_npz(cache_dir / "b_windows.npz", w, y)

    manifest = scan_existing_cache(cache_dir)
    assert validate_manifest(cache_dir, manifest) is True


def test_validate_manifest_empty_false(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # No npz → scan creates empty manifest
    manifest = scan_existing_cache(cache_dir)
    assert validate_manifest(cache_dir, manifest) is False


def test_validate_manifest_missing_refs_false(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    w = np.zeros((2, 19, 10), dtype=np.float32)
    y = np.zeros((2, 10), dtype=np.float32)
    y[0, :] = 1.0
    _save_npz(cache_dir / "keep_windows.npz", w, y)

    # Build valid manifest then corrupt references
    manifest = scan_existing_cache(cache_dir)
    # Point all entries to a missing file
    for k in ("partial_seizure", "full_seizure", "no_seizure"):
        for item in manifest.get(k, []):
            item["cache_file"] = "missing_windows.npz"
    (cache_dir / "manifest.json").write_text(json.dumps(manifest))

    assert validate_manifest(cache_dir, manifest) is False

