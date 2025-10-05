#!/usr/bin/env python3
"""Convert compressed NPZ cache to memory-mapped NPY format for production ML.

This script converts the existing compressed NPZ cache files to uncompressed,
memory-mappable NPY files for optimal performance and memory efficiency.

Performance Benefits:
- Zero decompression overhead (uncompressed)
- OS-managed memory (kernel page cache)
- Workers share physical memory automatically
- Scales to any dataset size
- Industry standard (NumPy mmap since 2005)

Memory Benefits:
- Compressed NPZ: 387 GB RAM needed (OOMs on Modal A100)
- Memory-mapped NPY: <1 GB RAM per worker (OS manages swapping)

Usage:
    python scripts/convert_cache_to_mmap.py \\
        --source cache/tusz/train \\
        --dest cache/tusz_mmap/train

    python scripts/convert_cache_to_mmap.py \\
        --source cache/tusz/dev \\
        --dest cache/tusz_mmap/dev
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def convert_npz_to_npy(npz_file: Path, dest_dir: Path) -> tuple[Path, Path]:
    """Convert single NPZ file to pair of uncompressed NPY files.

    Args:
        npz_file: Path to compressed NPZ file (e.g., "aaaaaajy_s001_t000_windows.npz")
        dest_dir: Destination directory for NPY files

    Returns:
        Tuple of (windows_file, labels_file) paths

    Raises:
        ValueError: If NPZ file is missing required arrays
        IOError: If file operations fail
    """
    # Load compressed data
    with np.load(npz_file) as data:
        if "windows" not in data:
            raise ValueError(f"NPZ file missing 'windows' array: {npz_file}")

        windows = data["windows"][:]  # (N, C, T) float32
        labels = data["labels"][:] if "labels" in data else None  # (N, T) float32

    # Generate output file names
    # Input:  "aaaaaajy_s001_t000_windows.npz"
    # Output: "aaaaaajy_s001_t000_data.npy", "aaaaaajy_s001_t000_labels.npy"
    stem = npz_file.stem.replace("_windows", "")
    windows_file = dest_dir / f"{stem}_data.npy"
    labels_file = dest_dir / f"{stem}_labels.npy"

    # Save as uncompressed NPY (memory-mappable!)
    np.save(windows_file, windows)
    if labels is not None:
        np.save(labels_file, labels)

    # Verify mmap works
    try:
        mmap_test = np.load(windows_file, mmap_mode='r')
        assert mmap_test.shape == windows.shape, f"Shape mismatch: {mmap_test.shape} != {windows.shape}"
        del mmap_test  # Close mmap handle
    except Exception as e:
        raise IOError(f"Mmap verification failed for {windows_file}: {e}")

    return windows_file, labels_file


def convert_cache_dir(source_dir: Path, dest_dir: Path) -> dict[str, int]:
    """Convert all NPZ files in directory to memory-mapped NPY format.

    Args:
        source_dir: Source directory containing NPZ files
        dest_dir: Destination directory for NPY files

    Returns:
        Dictionary with conversion statistics
    """
    # Validate source directory
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    npz_files = sorted(source_dir.glob("*_windows.npz"))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {source_dir}")

    # Create destination directory
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Converting {len(npz_files)} NPZ files to memory-mapped NPY format")
    print(f"Source: {source_dir}")
    print(f"Dest:   {dest_dir}")
    print(f"{'='*80}\n")

    # Convert files with progress bar
    stats = {
        "total_files": len(npz_files),
        "converted": 0,
        "failed": 0,
        "total_size_mb": 0,
    }

    failed_files = []

    for npz_file in tqdm(npz_files, desc="Converting", unit="file"):
        try:
            windows_file, labels_file = convert_npz_to_npy(npz_file, dest_dir)

            # Track statistics
            stats["converted"] += 1
            stats["total_size_mb"] += windows_file.stat().st_size / (1024**2)
            if labels_file.exists():
                stats["total_size_mb"] += labels_file.stat().st_size / (1024**2)

        except Exception as e:
            stats["failed"] += 1
            failed_files.append((npz_file, str(e)))
            tqdm.write(f"❌ Failed: {npz_file.name}: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print("CONVERSION SUMMARY")
    print(f"{'='*80}")
    print(f"Total files:     {stats['total_files']}")
    print(f"Converted:       {stats['converted']} ✅")
    print(f"Failed:          {stats['failed']} {'❌' if stats['failed'] > 0 else ''}")
    print(f"Total size:      {stats['total_size_mb']/1024:.1f} GB")
    print(f"{'='*80}\n")

    if failed_files:
        print("⚠️  FAILED FILES:")
        for npz_file, error in failed_files:
            print(f"  - {npz_file.name}: {error}")
        print()

    if stats["failed"] > 0:
        print("⚠️  Some files failed to convert. Review errors above.")
        return stats

    print("✅ All files converted successfully!")
    print("\n📋 NEXT STEPS:")
    print(f"1. Verify disk space: du -sh {dest_dir}")
    print(f"2. Regenerate manifests: python -m src scan-cache --cache-dir {dest_dir}")
    print("3. Update configs to point to new cache directory")
    print("4. Run local tests: make test && make s")
    print()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert compressed NPZ cache to memory-mapped NPY format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source directory containing NPZ files (e.g., cache/tusz/train)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination directory for NPY files (e.g., cache/tusz_mmap/train)",
    )

    args = parser.parse_args()

    try:
        stats = convert_cache_dir(args.source, args.dest)

        if stats["failed"] > 0:
            sys.exit(1)

        sys.exit(0)

    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
