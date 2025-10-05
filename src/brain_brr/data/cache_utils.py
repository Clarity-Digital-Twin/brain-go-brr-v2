"""Cache utilities for EEG datasets."""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from src.brain_brr.constants import MANIFEST_FILENAME

# Module logger
logger = logging.getLogger(__name__)

# Guard tqdm import for Modal/subprocess environments with typing-friendly fallbacks
tqdm: Any | None
try:
    # tqdm has no type stubs (third-party library)
    import tqdm as _tqdm  # type: ignore[import-untyped]

    tqdm = _tqdm.tqdm
except Exception:  # ImportError or runtime issues
    tqdm = None


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
    """Scan a cache directory and build a seizure-category manifest.

    Supports both NPZ format (legacy) and NPY format (mmap, production).

    NPZ format: filename_windows.npz (contains windows + labels)
    NPY format: filename_data.npy + filename_labels.npy (memory-mappable)

    The manifest has three keys: "partial_seizure", "full_seizure", and "no_seizure".
    Each item is a mapping with keys: {"cache_file": str, "window_idx": int}.
    """
    cache_dir = Path(cache_dir)
    manifest: dict[str, list[dict[str, Any]]] = {
        "partial_seizure": [],
        "full_seizure": [],
        "no_seizure": [],
    }

    # Check for NPY format (production mmap cache) first
    npy_data_files = sorted(cache_dir.glob("*_data.npy"))
    npz_files = sorted(cache_dir.glob("*.npz"))

    if npy_data_files:
        # NPY format (production): Use *_data.npy + *_labels.npy
        cache_files = npy_data_files
        is_npy_format = True
    elif npz_files:
        # NPZ format (legacy): Use *.npz
        cache_files = npz_files
        is_npy_format = False
    else:
        # No cache files found
        with (cache_dir / MANIFEST_FILENAME).open("w") as f:
            json.dump(manifest, f)
        return manifest

    # Centralized iterator choice (handles tqdm=None + env flag)
    from src.brain_brr.utils.env import env

    disable_tqdm = env.disable_tqdm() or tqdm is None
    logger.debug(f"[CACHE] tqdm disabled={disable_tqdm} | files={len(cache_files)}")

    if disable_tqdm:
        iterable = cache_files
    else:
        iterable = cast(Any, tqdm)(cache_files, desc="Scanning cache", leave=False)

    for cache_path in iterable:
        try:
            if is_npy_format:
                # NPY format: Load labels from separate file
                # cache_path is *_data.npy, find corresponding *_labels.npy
                stem = cache_path.stem.replace("_data", "")
                labels_path = cache_path.parent / f"{stem}_labels.npy"

                if not labels_path.exists():
                    warnings.warn(
                        f"⚠️  NPY file {cache_path.name} has NO LABELS FILE ({labels_path.name})! "
                        f"This indicates cache corruption or incomplete conversion. "
                        f"Excluding from balanced sampling.",
                        stacklevel=2,
                    )
                    continue

                labels = np.load(labels_path, mmap_mode="r")
                # For manifest, reference the stem without _data suffix
                manifest_filename = f"{stem}_windows.npz"  # Keep NPZ-style naming for compatibility
            else:
                # NPZ format (legacy)
                with np.load(cache_path) as data:
                    if "labels" not in data:
                        warnings.warn(
                            f"⚠️  NPZ file {cache_path.name} has NO LABELS! "
                            f"This indicates cache corruption or incomplete processing. "
                            f"Excluding from balanced sampling.",
                            stacklevel=2,
                        )
                        continue
                    labels = data["labels"]
                manifest_filename = cache_path.name

        except (OSError, ValueError) as e:
            logger.warning(f"Skipping {cache_path.name}: {e}")
            continue

        n_windows = int(labels.shape[0])
        for w_idx in range(n_windows):
            lbl = labels[w_idx]
            ratio = float((lbl > 0).mean())
            # Use relative path (just filename) for portability
            item = {"cache_file": manifest_filename, "window_idx": int(w_idx)}
            if ratio == 0.0:
                manifest["no_seizure"].append(item)
            elif ratio >= 0.99:
                manifest["full_seizure"].append(item)
            else:
                manifest["partial_seizure"].append(item)

    with (cache_dir / MANIFEST_FILENAME).open("w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    n_partial = len(manifest["partial_seizure"])
    n_full = len(manifest["full_seizure"])
    n_none = len(manifest["no_seizure"])
    total = n_partial + n_full + n_none

    format_type = "NPY (mmap)" if is_npy_format else "NPZ (legacy)"

    if n_partial == 0:
        logger.warning(
            f"No partial seizure windows found in {len(cache_files)} {format_type} files!"
        )
        logger.warning(f"  Full seizure: {n_full}, No seizure: {n_none}")
    else:
        logger.info(
            f"Manifest created from {format_type}: {n_partial} partial, {n_full} full, {n_none} no-seizure"
        )
        logger.info(f"  Seizure ratio: {(n_partial + n_full) / total:.1%}")

    return manifest


def validate_manifest(cache_dir: Path, manifest: dict[str, Any]) -> bool:
    """Validate that a manifest matches the current cache directory.

    Supports both NPZ format (legacy) and NPY format (mmap, production).

    Conditions for validity:
    - Manifest has at least one window total across categories
    - All referenced cache files exist in ``cache_dir`` (allowing a small
      fraction of missing files due to partial cache updates)

    Returns:
        True if manifest appears valid for ``cache_dir``; False otherwise.
    """
    try:
        cache_dir = Path(cache_dir)

        # Build set of available files (both NPZ and NPY formats)
        # NPZ format: filename_windows.npz
        # NPY format: filename_data.npy + filename_labels.npy
        #   → Manifest references filename_windows.npz (for compatibility)
        #   → We check if filename_data.npy exists
        npz_set = {p.name for p in cache_dir.glob("*.npz")}
        npy_data_files = cache_dir.glob("*_data.npy")
        # Convert NPY data files to NPZ-style names for manifest comparison
        npy_set = {p.stem.replace("_data", "") + "_windows.npz" for p in npy_data_files}

        available_files = npz_set | npy_set

        total = 0
        missing_refs = 0
        for key in ("partial_seizure", "full_seizure", "no_seizure"):
            entries = manifest.get(key, []) or []
            total += len(entries)
            for item in entries:
                cf = str(item.get("cache_file", ""))
                if cf not in available_files:
                    missing_refs += 1

        if total == 0:
            return False

        # If more than 5% of entries reference missing files, treat as invalid
        return not (total > 0 and (missing_refs / total) > 0.05)
    except Exception:
        return False
