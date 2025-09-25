#!/usr/bin/env python3
"""Clean up Modal persistence volume - remove unnecessary cache directories."""

import modal
import shutil
import os

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

    print("=== MODAL VOLUME CLEANUP ===\n")

    # Directories to DELETE (we use S3 mount for cache now!)
    dirs_to_delete = [
        "/results/cache",      # Old cache directory - we use S3 mount now!
        "/results/results",    # Confusing duplicate - only keep /results/smoke, /results/train etc
    ]

    for dir_path in dirs_to_delete:
        if os.path.exists(dir_path):
            print(f"🗑️ DELETING: {dir_path}")
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ Deleted successfully")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ℹ️ {dir_path} does not exist (already clean)")

    # Directories to KEEP
    dirs_to_keep = [
        "/results/smoke",      # Smoke test results
        "/results/train",      # Full training results (when it happens)
        "/results/checkpoints", # Model checkpoints
        "/results/tensorboard", # Tensorboard logs
        "/results/wandb",      # W&B logs
    ]

    print("\n=== KEEPING THESE DIRECTORIES ===")
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
            print(f"✅ {dir_path}: {file_count} files, {total_size/1024/1024:.2f} MB")
        else:
            print(f"   {dir_path}: Will be created when needed")

    # Commit changes to volume
    results_volume.commit()

    print("\n=== FINAL VOLUME STRUCTURE ===")
    # Show what's left
    for root, dirs, files in os.walk("/results"):
        level = root.replace("/results", "").count(os.sep)
        if level <= 1:  # Only show top 2 levels
            indent = "  " * level
            print(f"{indent}{os.path.basename(root)}/")

    print("\n✅ CLEANUP COMPLETE!")
    print("\n📝 IMPORTANT NOTES:")
    print("1. Cache is now mounted from S3 at /cache (not /results/cache)")
    print("2. Training outputs go to /results/{smoke,train,etc}")
    print("3. The S3 mount provides the preprocessed NPZ files")

    return "Cleanup complete"

@app.local_entrypoint()
def main():
    """Run cleanup."""
    result = cleanup_volume.remote()
    print(f"\n{result}")

if __name__ == "__main__":
    main()