#!/usr/bin/env python3
"""Check which files are in train vs val split."""

from pathlib import Path

data_root = Path('data/tusz/edf/dev')
edf_files = sorted(data_root.glob("**/*.edf"))

print(f"Total files: {len(edf_files)}")

val_split = int(len(edf_files) * 0.2)
train_files = edf_files[val_split:]
val_files = edf_files[:val_split]

print(f"Train: {len(train_files)}, Val: {len(val_files)}")
print(f"\nFirst train file: {train_files[0] if train_files else 'None'}")
print(f"First val file: {val_files[0] if val_files else 'None'}")

# Test first train file
if train_files:
    from src.experiment.data import load_edf_file
    print(f"\nTesting first TRAIN file: {train_files[0]}")
    try:
        data, fs = load_edf_file(train_files[0])
        print(f"✅ SUCCESS: {data.shape}")
    except Exception as e:
        print(f"❌ FAILED: {e}")