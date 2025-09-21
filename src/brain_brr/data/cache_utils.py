"""Cache utilities for EEG datasets."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Guard tqdm import for Modal/subprocess environments
try:
    from tqdm import tqdm  # type: ignore[import-untyped]
except Exception:  # ImportError or runtime issues
    tqdm = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CacheStatus:
    total_files: int
    cached_files: int
    missing_files: int
    missing: list[Path]


def cache_file_path(cache_dir: Path, edf_path: Path) -> Path:
    """Return expected cache npz path for an EDF file."""
    return cache_dir / f"{edf_path.stem}_windows.npz"


def check_cache_completeness(edf_files: Iterable[Path], cache_dir: Path) -> CacheStatus:
    """Check how many EDF files have a corresponding cache npz file present.

    Args:
        edf_files: Iterable of EDF file paths
        cache_dir: Root directory where cache npz files live

    Returns:
        CacheStatus with counts and missing file list
    """
    edf_list = list(edf_files)
    missing: list[Path] = []
    cached = 0
    for p in edf_list:
        if cache_file_path(cache_dir, p).exists():
            cached += 1
        else:
            missing.append(p)
    total = len(edf_list)
    return CacheStatus(
        total_files=total, cached_files=cached, missing_files=total - cached, missing=missing
    )


def scan_existing_cache(cache_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Scan a cache directory of NPZ files and build a seizure-category manifest.

    The manifest has three keys: "partial_seizure", "full_seizure", and "no_seizure".
    Each item is a mapping with keys: {"cache_file": str, "window_idx": int}.
    """
    cache_dir = Path(cache_dir)
    manifest: dict[str, list[dict[str, Any]]] = {
        "partial_seizure": [],
        "full_seizure": [],
        "no_seizure": [],
    }

    npz_files = sorted(cache_dir.glob("*.npz"))
    if not npz_files:
        with (cache_dir / "manifest.json").open("w") as f:
            json.dump(manifest, f)
        return manifest

    # Centralized iterator choice (handles tqdm=None + env flag)
    disable_tqdm = os.getenv("BGB_DISABLE_TQDM", "").strip() == "1" or tqdm is None
    print(f"[CACHE] tqdm disabled={disable_tqdm} | files={len(npz_files)}", flush=True)

    iterator = npz_files if disable_tqdm else tqdm(npz_files, desc="Scanning cache", leave=False)
    for npz_path in iterator:
        try:
            with np.load(npz_path) as data:
                if "labels" not in data:
                    # No labels = assume all windows are no-seizure
                    n_windows = int(data["windows"].shape[0])
                    for w_idx in range(n_windows):
                        manifest["no_seizure"].append(
                            {"cache_file": npz_path.name, "window_idx": int(w_idx)}
                        )
                    continue
                labels = data["labels"]
        except (OSError, ValueError) as e:
            # Skip corrupted or inaccessible files
            print(f"Warning: Skipping {npz_path.name}: {e}")
            continue

        n_windows = int(labels.shape[0])
        for w_idx in range(n_windows):
            lbl = labels[w_idx]
            ratio = float((lbl > 0).mean())
            # Use relative path (just filename) for portability
            item = {"cache_file": npz_path.name, "window_idx": int(w_idx)}
            if ratio == 0.0:
                manifest["no_seizure"].append(item)
            elif ratio >= 0.99:
                manifest["full_seizure"].append(item)
            else:
                manifest["partial_seizure"].append(item)

    with (cache_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    n_partial = len(manifest["partial_seizure"])
    n_full = len(manifest["full_seizure"])
    n_none = len(manifest["no_seizure"])
    total = n_partial + n_full + n_none

    if n_partial == 0:
        print(f"WARNING: No partial seizure windows found in {len(npz_files)} files!")
        print(f"  Full seizure: {n_full}, No seizure: {n_none}")
    else:
        print(f"Manifest created: {n_partial} partial, {n_full} full, {n_none} no-seizure")
        print(f"  Seizure ratio: {(n_partial + n_full) / total:.1%}")

    return manifest
