#!/usr/bin/env python3
"""Inspect Modal persistence volume contents."""

import logging
import os

import modal

# Module logger
logger = logging.getLogger(__name__)

app = modal.App("brain-go-brr-inspect")

# Get the persistence volume
results_volume = modal.Volume.from_name("brain-go-brr-results", create_if_missing=False)

@app.function(
    timeout=300,
    volumes={"/results": results_volume},
    cpu=2,
    memory=2048,
)
def inspect_volume():
    """List all contents of the Modal persistence volume."""
    # Simple logging setup for Modal
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logger.info("=== Modal Persistence Volume Contents ===")

    # Walk through the entire volume
    for root, dirs, files in os.walk("/results"):
        level = root.replace("/results", "").count(os.sep)
        indent = " " * 2 * level
        logger.info(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)

        # Show directories
        for d in dirs[:10]:  # Limit to first 10 dirs per level
            logger.info(f"{subindent}{d}/")
        if len(dirs) > 10:
            logger.info(f"{subindent}... and {len(dirs)-10} more directories")

        # Show files
        for f in files[:5]:  # Limit to first 5 files per level
            size = os.path.getsize(os.path.join(root, f))
            logger.info(f"{subindent}{f} ({size/1024/1024:.2f} MB)")
        if len(files) > 5:
            logger.info(f"{subindent}... and {len(files)-5} more files")

        # Don't go too deep
        if level >= 3:
            dirs[:] = []  # Don't recurse further

    # Summary statistics
    logger.info("=== Summary ===")

    # Check specific directories
    for check_dir in ["/results/cache", "/results/results", "/results/smoke"]:
        if os.path.exists(check_dir):
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(check_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total_size += os.path.getsize(fp)
                        file_count += 1
                    except:
                        pass
            logger.info(f"{check_dir}: {file_count} files, {total_size/1024/1024:.2f} MB")
        else:
            logger.info(f"{check_dir}: DOES NOT EXIST")

    # Check cache structure (CRITICAL for training startup)
    logger.info("\n=== Cache Structure (Training Readiness) ===")
    cache_root = "/results/cache/tusz"
    if os.path.exists(cache_root):
        import json

        # Check metadata
        metadata_path = os.path.join(cache_root, ".cache_metadata.json")
        logger.info(f"\nMetadata: {os.path.exists(metadata_path)}")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                meta = json.load(f)
            logger.info(f"  → Version: {meta.get('version')}")
            logger.info(f"  → Expected: {meta.get('train_files')} train, {meta.get('dev_files')} dev")

        # Check train split
        train_dir = os.path.join(cache_root, "train")
        if os.path.exists(train_dir):
            train_npz = len([f for f in os.listdir(train_dir) if f.endswith('.npz')])
            train_manifest = os.path.join(train_dir, "manifest.json")
            train_index = os.path.join(train_dir, "_dataset_index.json")
            logger.info(f"\nTrain split:")
            logger.info(f"  → NPZ files: {train_npz}")
            logger.info(f"  → manifest.json: {'✅ EXISTS' if os.path.exists(train_manifest) else '❌ MISSING'}")
            logger.info(f"  → _dataset_index.json: {'✅ EXISTS' if os.path.exists(train_index) else '❌ MISSING'}")
            if os.path.exists(train_manifest):
                size_mb = os.path.getsize(train_manifest) / 1024 / 1024
                logger.info(f"     Size: {size_mb:.1f} MB")

        # Check dev split
        dev_dir = os.path.join(cache_root, "dev")
        if os.path.exists(dev_dir):
            dev_npz = len([f for f in os.listdir(dev_dir) if f.endswith('.npz')])
            dev_manifest = os.path.join(dev_dir, "manifest.json")
            dev_index = os.path.join(dev_dir, "_dataset_index.json")
            logger.info(f"\nDev split:")
            logger.info(f"  → NPZ files: {dev_npz}")
            logger.info(f"  → manifest.json: {'✅ EXISTS' if os.path.exists(dev_manifest) else '❌ MISSING'}")
            logger.info(f"  → _dataset_index.json: {'✅ EXISTS' if os.path.exists(dev_index) else '❌ MISSING'}")
            if os.path.exists(dev_index):
                size_kb = os.path.getsize(dev_index) / 1024
                logger.info(f"     Size: {size_kb:.1f} KB")

        # Readiness check
        train_ready = (
            os.path.exists(train_dir) and
            os.path.exists(os.path.join(train_dir, "manifest.json")) and
            len([f for f in os.listdir(train_dir) if f.endswith('.npz')]) >= 4600
        )
        dev_ready = (
            os.path.exists(dev_dir) and
            os.path.exists(os.path.join(dev_dir, "_dataset_index.json")) and
            len([f for f in os.listdir(dev_dir) if f.endswith('.npz')]) >= 1800
        )

        logger.info(f"\n🚀 TRAINING READINESS:")
        logger.info(f"  → Train: {'✅ READY' if train_ready else '❌ NOT READY'}")
        logger.info(f"  → Dev: {'✅ READY' if dev_ready else '❌ NOT READY'}")
        logger.info(f"  → Modal: {'✅ READY FOR TRAINING' if (train_ready and dev_ready) else '❌ NEEDS SETUP'}")
    else:
        logger.info(f"{cache_root}: DOES NOT EXIST - Run populate_cache first!")

    return "Inspection complete"

@app.local_entrypoint()
def main():
    """Run inspection."""
    # Simple logging for local entrypoint
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    result = inspect_volume.remote()
    logger.info(result)

if __name__ == "__main__":
    main()
