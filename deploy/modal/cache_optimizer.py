#!/usr/bin/env python
"""Cache optimization script for Modal deployment.

Copies cached NPZ files from S3 mount to local persistent volume for faster access.
S3 CloudBucketMount is optimized for sequential reads, not random access.
This script should be run once before training to localize the cache.
"""

import shutil
import time
from pathlib import Path
from typing import Optional


def optimize_cache_for_modal(
    s3_cache_dir: Path,
    local_cache_dir: Path,
    force: bool = False,
    verbose: bool = True,
) -> bool:
    """Copy cached NPZ files from S3 to local persistent volume.

    Args:
        s3_cache_dir: Source cache directory on S3 mount (e.g., /data/cache/tusz/train)
        local_cache_dir: Destination on persistent volume (e.g., /results/cache/tusz/train)
        force: Force recopy even if local cache exists
        verbose: Print progress messages

    Returns:
        True if cache was copied/updated, False if using existing
    """
    s3_cache_dir = Path(s3_cache_dir)
    local_cache_dir = Path(local_cache_dir)

    # Check if local cache already exists and is non-empty
    if local_cache_dir.exists() and not force:
        npz_files = list(local_cache_dir.glob("*.npz"))
        manifest = local_cache_dir / "manifest.json"

        if npz_files and manifest.exists():
            if verbose:
                print(f"[CACHE] Local cache already exists with {len(npz_files)} NPZ files", flush=True)
                print(f"[CACHE] Using existing cache at {local_cache_dir}", flush=True)
                print("[CACHE] To force recopy, set BGB_FORCE_CACHE_COPY=1", flush=True)
            return False

    # Check if S3 cache exists
    if not s3_cache_dir.exists():
        print(f"[ERROR] S3 cache directory not found: {s3_cache_dir}", flush=True)
        print("[ERROR] Build cache first or check S3 mount path", flush=True)
        return False

    # Get list of NPZ files to copy
    s3_npz_files = list(s3_cache_dir.glob("*.npz"))
    s3_manifest = s3_cache_dir / "manifest.json"

    if not s3_npz_files:
        print(f"[ERROR] No NPZ files found in S3 cache: {s3_cache_dir}", flush=True)
        return False

    if verbose:
        print(f"[CACHE] Found {len(s3_npz_files)} NPZ files in S3 cache", flush=True)
        total_size_gb = sum(f.stat().st_size for f in s3_npz_files) / (1024**3)
        print(f"[CACHE] Total size: {total_size_gb:.2f} GB", flush=True)

    # Create local cache directory
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    # Copy files with progress tracking
    start_time = time.time()
    copied_files = 0
    failed_files = []

    print(f"[CACHE] Copying cache from S3 to local volume...", flush=True)
    print(f"[CACHE] From: {s3_cache_dir}", flush=True)
    print(f"[CACHE] To: {local_cache_dir}", flush=True)

    for i, s3_file in enumerate(s3_npz_files, 1):
        local_file = local_cache_dir / s3_file.name

        # Skip if already exists and has same size
        if local_file.exists() and not force:
            if local_file.stat().st_size == s3_file.stat().st_size:
                if verbose and i % 100 == 0:
                    print(f"[CACHE] Skipping {i}/{len(s3_npz_files)}: {s3_file.name} (already exists)", flush=True)
                continue

        try:
            # Copy file
            shutil.copy2(s3_file, local_file)
            copied_files += 1

            # Progress update
            if verbose and (copied_files % 10 == 0 or copied_files == 1):
                elapsed = time.time() - start_time
                rate = copied_files / elapsed if elapsed > 0 else 0
                eta = (len(s3_npz_files) - i) / rate if rate > 0 else 0
                print(
                    f"[CACHE] Copied {copied_files} files | "
                    f"Progress: {i}/{len(s3_npz_files)} | "
                    f"Rate: {rate:.1f} files/s | "
                    f"ETA: {eta/60:.1f} min",
                    flush=True,
                )
        except Exception as e:
            print(f"[ERROR] Failed to copy {s3_file.name}: {e}", flush=True)
            failed_files.append(s3_file.name)

    # Copy manifest if exists
    if s3_manifest.exists():
        try:
            shutil.copy2(s3_manifest, local_cache_dir / "manifest.json")
            if verbose:
                print("[CACHE] Copied manifest.json", flush=True)
        except Exception as e:
            print(f"[ERROR] Failed to copy manifest: {e}", flush=True)

    # Summary
    elapsed_time = time.time() - start_time
    print(f"[CACHE] Cache optimization complete!", flush=True)
    print(f"[CACHE] Copied {copied_files} new/updated files in {elapsed_time/60:.1f} minutes", flush=True)

    if failed_files:
        print(f"[WARNING] Failed to copy {len(failed_files)} files", flush=True)
        for name in failed_files[:10]:  # Show first 10
            print(f"  - {name}", flush=True)

    # Verify local cache
    local_npz_files = list(local_cache_dir.glob("*.npz"))
    local_manifest = local_cache_dir / "manifest.json"

    if local_npz_files and local_manifest.exists():
        print(f"[CACHE] Local cache ready with {len(local_npz_files)} NPZ files", flush=True)
        print(f"[CACHE] Training will now use fast local cache at {local_cache_dir}", flush=True)
        return True
    else:
        print("[ERROR] Local cache verification failed!", flush=True)
        return False


def should_optimize_cache() -> Optional[tuple[Path, Path]]:
    """Check if cache optimization is needed based on environment.

    Returns:
        (s3_cache_path, local_cache_path) if optimization needed, None otherwise
    """
    import os

    # Only optimize on Modal (check for Modal environment)
    if not os.getenv("MODAL_FUNCTION_ID"):
        print("[CACHE] Not running on Modal (no MODAL_FUNCTION_ID)", flush=True)
        return None

    # Check if forcing optimization
    force = os.getenv("BGB_FORCE_CACHE_COPY", "").strip() == "1"
    if force:
        print("[CACHE] Forcing cache optimization (BGB_FORCE_CACHE_COPY=1)", flush=True)

    # Standard paths for Modal deployment
    s3_train_cache = Path("/data/cache/tusz/train")
    s3_val_cache = Path("/data/cache/tusz/val")
    local_train_cache = Path("/results/cache/tusz/train")
    local_val_cache = Path("/results/cache/tusz/val")

    # Log what we're checking
    print(f"[CACHE] Checking S3 train cache: {s3_train_cache}", flush=True)
    print(f"[CACHE]   Exists: {s3_train_cache.exists()}", flush=True)
    print(f"[CACHE] Checking local train cache: {local_train_cache}", flush=True)
    print(f"[CACHE]   Exists: {local_train_cache.exists()}", flush=True)

    # Check if S3 cache exists but local doesn't (or force copy)
    if s3_train_cache.exists() and (not local_train_cache.exists() or force):
        print(f"[CACHE] Will optimize: S3 cache exists, local {'forced' if force else 'missing'}", flush=True)
        return (s3_train_cache, local_train_cache)

    # Check validation cache too
    if s3_val_cache.exists() and (not local_val_cache.exists() or force):
        print(f"[CACHE] Will optimize VAL: S3 val cache exists, local {'forced' if force else 'missing'}", flush=True)
        return (s3_val_cache, local_val_cache)

    # Explain why we're not optimizing
    if not s3_train_cache.exists() and not s3_val_cache.exists():
        # S3 mount only has raw EDF files, not NPZ cache
        print("[CACHE] No cache on S3 mount (/data/cache/ doesn't exist)", flush=True)

        # Check if cache already exists locally
        if local_train_cache.exists():
            npz_count = len(list(local_train_cache.glob("*.npz")))
            if npz_count > 0:
                print(f"[CACHE] ✅ Using existing Modal volume cache: {npz_count} NPZ files", flush=True)
                print(f"[CACHE] Location: {local_train_cache}", flush=True)
                print("[CACHE] Training will use fast local cache!", flush=True)
            else:
                print("[CACHE] ⚠️ Cache directory exists but is empty", flush=True)
                print("[CACHE] Training will build cache on-the-fly (30-60 min)", flush=True)
        else:
            print("[CACHE] ⚠️ No cache found, will build on first epoch", flush=True)
            print("[CACHE] This will take 30-60 minutes but only happens once", flush=True)

    elif local_train_cache.exists() and not force:
        npz_count = len(list(local_train_cache.glob("*.npz")))
        print(f"[CACHE] ✅ Local cache already exists with {npz_count} NPZ files", flush=True)
        print("[CACHE] Using existing optimized cache (fast!)", flush=True)

    return None


if __name__ == "__main__":
    import sys

    # Check if we should optimize
    paths = should_optimize_cache()
    if paths:
        s3_path, local_path = paths
        print(f"[CACHE] Optimizing cache for Modal deployment...", flush=True)
        success = optimize_cache_for_modal(s3_path, local_path, verbose=True)
        if not success:
            print("[ERROR] Cache optimization failed!", flush=True)
            sys.exit(1)
    else:
        print("[CACHE] No cache optimization needed", flush=True)