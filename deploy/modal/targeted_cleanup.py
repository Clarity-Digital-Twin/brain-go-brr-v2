"""Targeted cleanup of Modal volume to free space for mmap cache completion.

DELETES (safe to remove):
- /results/cache/tusz/          OLD NPZ cache (449 GB) - replaced by mmap
- /results/diag_*/              Old diagnostic runs (~minimal, but unused)
- /results/v3_full_training/    Old training run (50-200 GB, 7 days old)

KEEPS (preserve):
- /results/cache/tusz_mmap/     NEW mmap cache (keep existing train + partial dev)
- /results/smoke/               Recent smoke test results

This will free ~500-700 GB, allowing dev split copy to complete.
"""

import modal
from pathlib import Path
import shutil

app = modal.App("brain-go-brr-v2-cleanup")

results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=False)


@app.function(
    timeout=600,  # 10 minutes
    cpu=4,
    memory=4096,
    volumes={"/results": results_volume},
)
def targeted_cleanup():
    """Delete old NPZ cache and diagnostic runs to free space."""
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("TARGETED MODAL VOLUME CLEANUP")
    logger.info("=" * 70)

    deleted_total = 0

    # 1. Delete old NPZ cache (biggest waste of space)
    old_npz_cache = Path("/results/cache/tusz")
    if old_npz_cache.exists():
        logger.info(f"\n[DELETE] Old NPZ cache: {old_npz_cache}")
        logger.info(
            "[DELETE] This is the compressed cache, replaced by mmap format"
        )

        # Calculate size before deletion
        size_gb = sum(f.stat().st_size for f in old_npz_cache.rglob("*") if f.is_file()) / (
            1024**3
        )
        logger.info(f"[DELETE] Size: {size_gb:.1f} GB")

        shutil.rmtree(old_npz_cache)
        deleted_total += size_gb
        logger.info(f"[DELETE] ✅ Deleted old NPZ cache ({size_gb:.1f} GB freed)")
    else:
        logger.info(f"[SKIP] Old NPZ cache not found: {old_npz_cache}")

    # 2. Delete diagnostic runs
    diag_dirs = [
        Path("/results/diag_1a_amp_off"),
        Path("/results/diag_1b_fallback"),
        Path("/results/diag_2a_blocking"),
    ]

    for diag_dir in diag_dirs:
        if diag_dir.exists():
            size_gb = (
                sum(f.stat().st_size for f in diag_dir.rglob("*") if f.is_file()) / (1024**3)
            )
            logger.info(f"\n[DELETE] Diagnostic run: {diag_dir} ({size_gb:.2f} GB)")
            shutil.rmtree(diag_dir)
            deleted_total += size_gb
            logger.info(f"[DELETE] ✅ Deleted {diag_dir.name}")
        else:
            logger.info(f"[SKIP] Diagnostic dir not found: {diag_dir}")

    # 3. Delete old full training run (if exists)
    old_training = Path("/results/v3_full_training")
    if old_training.exists():
        size_gb = (
            sum(f.stat().st_size for f in old_training.rglob("*") if f.is_file()) / (1024**3)
        )
        logger.info(f"\n[DELETE] Old training run: {old_training} ({size_gb:.1f} GB)")
        logger.info("[DELETE] This is from 7 days ago, likely has 100 checkpoints")
        shutil.rmtree(old_training)
        deleted_total += size_gb
        logger.info(f"[DELETE] ✅ Deleted old training run ({size_gb:.1f} GB freed)")
    else:
        logger.info(f"[SKIP] Old training run not found: {old_training}")

    # 4. Verify what's kept
    logger.info("\n" + "=" * 70)
    logger.info("PRESERVED DIRECTORIES")
    logger.info("=" * 70)

    mmap_cache = Path("/results/cache/tusz_mmap")
    if mmap_cache.exists():
        mmap_size = (
            sum(f.stat().st_size for f in mmap_cache.rglob("*") if f.is_file()) / (1024**3)
        )
        train_files = len(list((mmap_cache / "train").glob("*_data.npy")))
        dev_files = (
            len(list((mmap_cache / "dev").glob("*_data.npy"))) if (mmap_cache / "dev").exists() else 0
        )
        logger.info(f"[KEEP] Mmap cache: {mmap_cache} ({mmap_size:.1f} GB)")
        logger.info(f"[KEEP]   - Train: {train_files} files")
        logger.info(f"[KEEP]   - Dev: {dev_files} files (partial - will complete next)")
    else:
        logger.info(f"[WARNING] Mmap cache not found: {mmap_cache}")

    smoke_dir = Path("/results/smoke")
    if smoke_dir.exists():
        smoke_size = sum(f.stat().st_size for f in smoke_dir.rglob("*") if f.is_file()) / (
            1024**3
        )
        logger.info(f"[KEEP] Recent smoke test: {smoke_dir} ({smoke_size:.2f} GB)")
    else:
        logger.info(f"[SKIP] Smoke dir not found: {smoke_dir}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("CLEANUP SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total space freed: {deleted_total:.1f} GB")
    logger.info(f"Estimated free space now: ~{deleted_total:.0f} GB")
    logger.info(f"Dev split needs: 170 GB")
    logger.info(f"Sufficient space: {'✅ YES' if deleted_total >= 170 else '❌ NO'}")
    logger.info("=" * 70)

    results_volume.commit()
    logger.info("[VOLUME] ✅ Changes committed to Modal volume")

    return {"deleted_gb": deleted_total, "sufficient_space": deleted_total >= 170}


@app.local_entrypoint()
def main():
    """Run targeted cleanup."""
    result = targeted_cleanup.remote()
    print(f"\n✅ Cleanup complete: {result['deleted_gb']:.1f} GB freed")
    if result["sufficient_space"]:
        print("✅ Sufficient space for dev split copy!")
    else:
        print("⚠️ May still need more cleanup")
