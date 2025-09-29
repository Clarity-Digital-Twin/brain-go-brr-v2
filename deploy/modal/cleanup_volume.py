#!/usr/bin/env python3
"""Clean up Modal persistence volume - remove unnecessary cache directories."""

import logging
import os
import shutil

import modal

# Module logger
logger = logging.getLogger(__name__)

app = modal.App("brain-go-brr-cleanup")

# Get the persistence volume
results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=False)

@app.function(
    timeout=600,
    volumes={"/results": results_volume},
    cpu=4,
    memory=4096,
)
def cleanup_volume():
    """Clean up unnecessary directories from Modal persistence volume."""
    # Setup logging for Modal function
    from src.brain_brr.utils.logging_config import setup_logging
    setup_logging(format_style="simple", force=True)

    logger.info("=== MODAL VOLUME CLEANUP ===")

    # Directories to DELETE (we use S3 mount for cache now!)
    dirs_to_delete = [
        "/results/cache",      # Old cache directory - we use S3 mount now!
        "/results/results",    # Confusing duplicate - only keep /results/smoke, /results/train etc
    ]

    for dir_path in dirs_to_delete:
        if os.path.exists(dir_path):
            logger.info(f"🗑️ DELETING: {dir_path}")
            try:
                shutil.rmtree(dir_path)
                logger.info(f"   ✅ Deleted successfully")
            except Exception as e:
                logger.error(f"   ❌ Error: {e}")
        else:
            logger.info(f"   ℹ️ {dir_path} does not exist (already clean)")

    # Directories to KEEP
    dirs_to_keep = [
        "/results/smoke",      # Smoke test results
        "/results/train",      # Full training results (when it happens)
        "/results/checkpoints", # Model checkpoints
        "/results/tensorboard", # Tensorboard logs
        "/results/wandb",      # W&B logs
    ]

    logger.info("=== KEEPING THESE DIRECTORIES ===")
    for dir_path in dirs_to_keep:
        if os.path.exists(dir_path):
            # Get size
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total_size += os.path.getsize(fp)
                        file_count += 1
                    except:
                        pass
            logger.info(f"✅ {dir_path}: {file_count} files, {total_size/1024/1024:.2f} MB")
        else:
            logger.info(f"   {dir_path}: Will be created when needed")

    # Commit changes to volume
    results_volume.commit()

    logger.info("=== FINAL VOLUME STRUCTURE ===")
    # Show what's left
    for root, dirs, files in os.walk("/results"):
        level = root.replace("/results", "").count(os.sep)
        if level <= 1:  # Only show top 2 levels
            indent = "  " * level
            logger.info(f"{indent}{os.path.basename(root)}/")

    logger.info("✅ CLEANUP COMPLETE!")
    logger.info("📝 IMPORTANT NOTES:")
    logger.info("1. Cache is now mounted from S3 at /cache (not /results/cache)")
    logger.info("2. Training outputs go to /results/{smoke,train,etc}")
    logger.info("3. The S3 mount provides the preprocessed NPZ files")

    return "Cleanup complete"

@app.local_entrypoint()
def main():
    """Run cleanup."""
    # Setup logging for local entrypoint
    from src.brain_brr.utils.logging_config import setup_logging
    setup_logging(format_style="simple", force=True)

    result = cleanup_volume.remote()
    logger.info(result)

if __name__ == "__main__":
    main()
