#!/usr/bin/env python
"""Explore Modal volumes to see what's actually stored."""

import modal
import os
from pathlib import Path

app = modal.App("volume-explorer")

# Mount the volumes
data_volume = modal.Volume.from_name("brain-go-brr-data")
results_volume = modal.Volume.from_name("brain-go-brr-results")

@app.function(volumes={"/data_vol": data_volume, "/results": results_volume})
def explore_volumes():
    """List all contents of Modal volumes."""

    print("\n" + "="*80)
    print("MODAL VOLUME EXPLORATION")
    print("="*80)

    # Check brain-go-brr-data volume
    print("\n[1] brain-go-brr-data volume (/data_vol):")
    print("-"*40)
    if os.path.exists("/data_vol"):
        data_contents = list(Path("/data_vol").rglob("*"))
        if data_contents:
            # Show first 20 items
            for item in data_contents[:20]:
                rel_path = item.relative_to("/data_vol")
                if item.is_dir():
                    print(f"  📁 {rel_path}/")
                else:
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"  📄 {rel_path} ({size_mb:.2f} MB)")
            if len(data_contents) > 20:
                print(f"  ... and {len(data_contents) - 20} more items")
        else:
            print("  ⚠️ EMPTY - No files found!")
    else:
        print("  ❌ Volume not mounted")

    # Check brain-go-brr-results volume
    print("\n[2] brain-go-brr-results volume (/results):")
    print("-"*40)

    # Check cache specifically
    cache_dir = Path("/results/cache/tusz/train")
    if cache_dir.exists():
        npz_files = list(cache_dir.glob("*.npz"))
        manifest = cache_dir / "manifest.json"
        print(f"\n  📁 /results/cache/tusz/train/")
        print(f"     ✅ {len(npz_files)} NPZ files found")
        if manifest.exists():
            print(f"     ✅ manifest.json exists ({manifest.stat().st_size / 1024:.1f} KB)")
        if npz_files:
            # Show sample NPZ files
            for npz in npz_files[:3]:
                size_mb = npz.stat().st_size / (1024 * 1024)
                print(f"     📄 {npz.name} ({size_mb:.2f} MB)")
            if len(npz_files) > 3:
                print(f"     ... and {len(npz_files) - 3} more NPZ files")

    # Check other directories
    for subdir in ["/results/smoke", "/results/tusz_a100_100ep", "/results/cache"]:
        if os.path.exists(subdir):
            items = list(Path(subdir).iterdir())
            print(f"\n  📁 {subdir}/")
            for item in items[:5]:
                if item.is_dir():
                    sub_count = len(list(item.iterdir()))
                    print(f"     📁 {item.name}/ ({sub_count} items)")
                else:
                    size_mb = item.stat().st_size / (1024 * 1024)
                    print(f"     📄 {item.name} ({size_mb:.2f} MB)")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY:")
    print("-"*40)

    # Results volume stats
    results_total_size = 0
    results_file_count = 0
    for item in Path("/results").rglob("*"):
        if item.is_file():
            results_total_size += item.stat().st_size
            results_file_count += 1

    print(f"brain-go-brr-results volume:")
    print(f"  Total files: {results_file_count}")
    print(f"  Total size: {results_total_size / (1024**3):.2f} GB")

    # Data volume stats
    data_total_size = 0
    data_file_count = 0
    if os.path.exists("/data_vol"):
        for item in Path("/data_vol").rglob("*"):
            if item.is_file():
                data_total_size += item.stat().st_size
                data_file_count += 1

    print(f"\nbrain-go-brr-data volume:")
    print(f"  Total files: {data_file_count}")
    print(f"  Total size: {data_total_size / (1024**3):.2f} GB")

    if data_file_count == 0:
        print("  ⚠️ This volume is EMPTY and can be deleted!")

    print("="*80)

if __name__ == "__main__":
    with app.run():
        explore_volumes.remote()