"""Clean stray NPZ files from Modal SSD cache.

Context:
--------
During first failed smoke test (wrong cache path), datasets.py created 3 NPZ files
before we killed it. These are DUPLICATES of existing NPY files and must be removed.

Safety:
-------
- Only deletes NPZ files from /results/cache/tusz_mmap/
- Verifies corresponding NPY files exist before deleting
- Supports dry-run mode (default)
- Requires explicit --confirm flag for actual deletion

Usage:
------
# Dry run (list files to delete):
modal run deploy/modal/clean_stray_npz.py

# Actual deletion:
modal run deploy/modal/clean_stray_npz.py --confirm
"""

import logging
from pathlib import Path

import modal

# Create Modal app
app = modal.App("clean-stray-npz")

# Use same image as main training (has all dependencies)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==1.26.4",
)

# Mount results volume (where cache lives)
results_volume = modal.Volume.from_name(
    "brain-go-brr-results",
    create_if_missing=False,
)

logger = logging.getLogger(__name__)


@app.function(
    image=image,
    volumes={"/results": results_volume},
    timeout=600,  # 10 min max
)
def clean_npz_files(dry_run: bool = True) -> dict[str, int | list[str]]:
    """Find and optionally delete stray NPZ files.

    Args:
        dry_run: If True, only list files (don't delete). Default: True.

    Returns:
        Dict with:
            - found: Number of NPZ files found
            - deleted: Number of NPZ files deleted
            - files: List of file names processed
            - freed_mb: MB of disk space freed
    """
    cache_dir = Path("/results/cache/tusz_mmap")

    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found: {cache_dir}")

    logger.info("=" * 70)
    logger.info("[CLEANUP] Searching for stray NPZ files...")
    logger.info("=" * 70)

    # Find all NPZ files
    npz_files = []
    for split_dir in ["train", "dev"]:
        split_path = cache_dir / split_dir
        if split_path.exists():
            npz_files.extend(split_path.glob("*.npz"))

    if not npz_files:
        logger.info("[CLEANUP] ✅ No NPZ files found - cache is clean!")
        return {
            "found": 0,
            "deleted": 0,
            "files": [],
            "freed_mb": 0.0,
        }

    logger.info(f"[CLEANUP] Found {len(npz_files)} NPZ files:")

    # Verify each NPZ has corresponding NPY files
    files_to_delete = []
    total_size_bytes = 0

    for npz_file in npz_files:
        # Calculate expected NPY file names
        stem = npz_file.stem.replace("_windows", "")
        data_file = npz_file.parent / f"{stem}_data.npy"
        labels_file = npz_file.parent / f"{stem}_labels.npy"

        # Check if NPY files exist
        has_npy = data_file.exists() and labels_file.exists()
        size_mb = npz_file.stat().st_size / (1024**2)
        total_size_bytes += npz_file.stat().st_size

        status = "✓ NPY files exist" if has_npy else "⚠️  NPY files MISSING!"

        logger.info(f"  - {npz_file.name} ({size_mb:.1f} MiB) → {status}")

        if has_npy:
            files_to_delete.append(npz_file)
        else:
            logger.warning(
                f"[CLEANUP] ⚠️  Skipping {npz_file.name} - no NPY files found!"
            )

    if not files_to_delete:
        logger.warning("[CLEANUP] ⚠️  No NPZ files safe to delete")
        return {
            "found": len(npz_files),
            "deleted": 0,
            "files": [f.name for f in npz_files],
            "freed_mb": 0.0,
        }

    freed_mb = sum(f.stat().st_size for f in files_to_delete) / (1024**2)

    if dry_run:
        logger.info("=" * 70)
        logger.info("[CLEANUP] DRY RUN MODE - No files deleted")
        logger.info(f"[CLEANUP] Would delete {len(files_to_delete)} NPZ files ({freed_mb:.1f} MiB)")
        logger.info("[CLEANUP] Run with --confirm to actually delete")
        logger.info("=" * 70)
        return {
            "found": len(npz_files),
            "deleted": 0,
            "files": [f.name for f in files_to_delete],
            "freed_mb": freed_mb,
        }

    # Actually delete files
    logger.info("=" * 70)
    logger.info("[CLEANUP] DELETING NPZ FILES...")
    deleted_count = 0
    for npz_file in files_to_delete:
        try:
            npz_file.unlink()
            deleted_count += 1
            logger.info(f"[CLEANUP] ✅ Deleted {npz_file.name}")
        except Exception as e:
            logger.error(f"[CLEANUP] ❌ Failed to delete {npz_file.name}: {e}")

    logger.info("=" * 70)
    logger.info(f"[CLEANUP] ✅ Deleted {deleted_count}/{len(files_to_delete)} files")
    logger.info(f"[CLEANUP] ✅ Freed {freed_mb:.1f} MiB disk space")
    logger.info("[CLEANUP] ✅ Cache clean - only NPY files remain")
    logger.info("=" * 70)

    # Commit changes to volume
    results_volume.commit()

    return {
        "found": len(npz_files),
        "deleted": deleted_count,
        "files": [f.name for f in files_to_delete],
        "freed_mb": freed_mb,
    }


@app.local_entrypoint()
def main(confirm: bool = False):
    """Main entry point.

    Args:
        confirm: If True, actually delete files. Default: False (dry-run).
    """
    dry_run = not confirm

    if dry_run:
        print("\n🔍 Running in DRY-RUN mode (use --confirm to delete)")
    else:
        print("\n🗑️  DELETING NPZ FILES (confirm=True)")

    result = clean_npz_files.remote(dry_run=dry_run)

    print("\n" + "=" * 70)
    print("CLEANUP SUMMARY")
    print("=" * 70)
    print(f"NPZ files found:    {result['found']}")
    print(f"NPZ files deleted:  {result['deleted']}")
    print(f"Disk space freed:   {result['freed_mb']:.1f} MiB")
    print(f"Files: {', '.join(result['files'][:5])}")
    if len(result['files']) > 5:
        print(f"       ... and {len(result['files']) - 5} more")
    print("=" * 70)

    if result['deleted'] > 0:
        print("✅ Cleanup complete - cache is now 100% NPY format")
    elif result['found'] == 0:
        print("✅ Cache already clean - no NPZ files found")
    elif dry_run:
        print("ℹ️  Dry run complete - run with --confirm to delete")
    else:
        print("⚠️  No files deleted (safety checks failed)")
