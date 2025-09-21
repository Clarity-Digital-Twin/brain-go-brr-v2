"""Cache utilities for EEG datasets."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


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
