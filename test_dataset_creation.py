#!/usr/bin/env python3
"""Test dataset creation exactly as pipeline does."""

from pathlib import Path
from src.experiment.data import EEGWindowDataset

data_root = Path('data/tusz/edf/dev')
edf_files = sorted(data_root.glob("**/*.edf"))

val_split = int(len(edf_files) * 0.2)
train_files = edf_files[val_split:]
val_files = edf_files[:val_split]

print(f"Creating dataset with {len(train_files)} train files...")
print(f"First 5 train files:")
for f in train_files[:5]:
    print(f"  {f}")

# Exactly replicate the pipeline
train_label_files = [p.with_suffix(".csv") for p in train_files]

# Use a temp cache dir
cache_dir = Path("temp_cache_test")
cache_dir.mkdir(exist_ok=True)

try:
    train_dataset = EEGWindowDataset(
        train_files,
        label_files=train_label_files,
        cache_dir=cache_dir / "train",
    )
    print(f"\n✅ Dataset created successfully!")
    print(f"Dataset length: {len(train_dataset)}")
except Exception as e:
    print(f"\n❌ Dataset creation failed: {e}")

    # Find which file failed
    print("\nFinding problematic file...")
    for i, f in enumerate(train_files[:20]):  # Test first 20
        try:
            test_dataset = EEGWindowDataset(
                [f],
                label_files=[f.with_suffix(".csv")],
                cache_dir=cache_dir / f"test_{i}",
            )
            print(f"  ✅ {f.name}")
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")
            print(f"\n  Full path: {f}")
            break