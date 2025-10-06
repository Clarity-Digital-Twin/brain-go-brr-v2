#!/usr/bin/env python3
"""Safely clean up old smoke test outputs from Modal volume.

SAFETY: Only removes /results/smoke/* outputs, NEVER touches cache!
Cache lives at /results/cache/tusz_mmap and is needed for training.

What we clean:
- /results/smoke/ - Old smoke test checkpoints/logs (573 MB)
- /results/cache/smoke/ - Old smoke test cache attempt (WRONG LOCATION)
- /results/cache/tusz/ - Legacy NPZ cache (REPLACED by tusz_mmap)

What we KEEP:
- /results/cache/tusz_mmap/ - NPY mmap cache (529GB, CRITICAL)
- /results/v3_full_training/ - Current training outputs
"""

import logging
import os
import shutil
from pathlib import Path

import modal

logger = logging.getLogger(__name__)

app = modal.App("brain-go-brr-cleanup-smokes")

results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=False)

@app.function(
    timeout=600,
    volumes={"/results": results_volume},
    cpu=4,
    memory=4096,
)
def cleanup_old_outputs(dry_run: bool = True):
    """Clean up old smoke test outputs and legacy cache directories.

    Args:
        dry_run: If True, only show what would be deleted (safe default)
    """
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logger.info("=" * 70)
    logger.info("MODAL VOLUME CLEANUP - OLD SMOKE TESTS & LEGACY CACHE")
    logger.info("=" * 70)

    if dry_run:
        logger.info("🔍 DRY RUN MODE - No files will be deleted")
    else:
        logger.info("⚠️  LIVE MODE - Files will be permanently deleted!")

    logger.info("")

    # Directories to clean
    cleanup_targets = [
        ("/results/smoke", "Old smoke test outputs (checkpoints, wandb, tensorboard)"),
        ("/results/cache/smoke", "Incorrectly placed smoke cache"),
        ("/results/cache/tusz", "Legacy NPZ cache (replaced by tusz_mmap)"),
    ]

    # CRITICAL: Verify we're NOT touching production cache
    production_cache = Path("/results/cache/tusz_mmap")
    if not production_cache.exists():
        logger.error("❌ CRITICAL: Production cache missing at /results/cache/tusz_mmap")
        logger.error("❌ ABORTING cleanup to prevent data loss!")
        return "ABORTED: Production cache not found"

    # Count production cache files
    train_npy = len(list((production_cache / "train").glob("*_data.npy")))
    dev_npy = len(list((production_cache / "dev").glob("*_data.npy")))

    logger.info("✅ Production cache verified:")
    logger.info(f"   /results/cache/tusz_mmap/train: {train_npy} data files (expected 4667)")
    logger.info(f"   /results/cache/tusz_mmap/dev: {dev_npy} data files (expected 1832)")

    if train_npy != 4667 or dev_npy != 1832:
        logger.error("❌ CRITICAL: Production cache file counts don't match!")
        logger.error("❌ ABORTING cleanup to prevent data loss!")
        return "ABORTED: Production cache incomplete"

    logger.info("")

    # Calculate sizes and show what would be deleted
    total_size_mb = 0
    total_files = 0

    for dir_path, description in cleanup_targets:
        if os.path.exists(dir_path):
            size_mb = 0
            file_count = 0
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        size_mb += os.path.getsize(fp) / 1024 / 1024
                        file_count += 1
                    except:
                        pass

            total_size_mb += size_mb
            total_files += file_count

            if dry_run:
                logger.info(f"🗑️  Would delete: {dir_path}")
            else:
                logger.info(f"🗑️  Deleting: {dir_path}")

            logger.info(f"   {description}")
            logger.info(f"   Size: {size_mb:.1f} MB ({file_count} files)")

            if not dry_run:
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"   ✅ Deleted successfully")
                except Exception as e:
                    logger.error(f"   ❌ Error: {e}")

            logger.info("")
        else:
            logger.info(f"✓ {dir_path}: Already clean (doesn't exist)")
            logger.info("")

    # Summary
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total space to free: {total_size_mb:.1f} MB ({total_files} files)")
    logger.info("")

    # Verify what's kept
    logger.info("KEEPING (CRITICAL FOR TRAINING):")
    logger.info("  ✅ /results/cache/tusz_mmap/ - Production NPY mmap cache (529 GB)")
    logger.info("  ✅ /results/v3_full_training/ - Current training outputs")
    logger.info("")

    if not dry_run:
        # Commit changes
        results_volume.commit()
        logger.info("✅ Changes committed to volume")
        return f"Cleanup complete: {total_size_mb:.1f} MB freed"
    else:
        logger.info("🔍 DRY RUN - No changes made")
        logger.info("   Run with dry_run=False to actually delete files")
        return f"Dry run complete: {total_size_mb:.1f} MB would be freed"

@app.local_entrypoint()
def main(dry_run: bool = True):
    """Run cleanup (dry_run=True by default for safety)."""
    result = cleanup_old_outputs.remote(dry_run=dry_run)
    print(result)

if __name__ == "__main__":
    main()
