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
    # Setup logging for Modal function
    from src.brain_brr.utils.logging_config import setup_logging
    setup_logging(format_style="simple", force=True)

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

    return "Inspection complete"

@app.local_entrypoint()
def main():
    """Run inspection."""
    # Setup logging for local entrypoint
    from src.brain_brr.utils.logging_config import setup_logging
    setup_logging(format_style="simple", force=True)

    result = inspect_volume.remote()
    logger.info(result)

if __name__ == "__main__":
    main()
