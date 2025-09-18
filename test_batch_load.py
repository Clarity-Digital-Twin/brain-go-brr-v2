#!/usr/bin/env python3
"""Test loading multiple files to find the problematic one."""

from pathlib import Path
from src.experiment.data import load_edf_file
import traceback

data_root = Path('data/tusz/edf/dev')
edf_files = sorted(data_root.glob("**/*.edf"))

val_split = int(len(edf_files) * 0.2)
train_files = edf_files[val_split:]

print(f"Testing first 10 train files...")
print("=" * 60)

failed_files = []

for i, f in enumerate(train_files[:10]):
    print(f"\n[{i+1}/10] {f.name}")
    try:
        data, fs = load_edf_file(f)
        print(f"  ✅ OK: {data.shape}")
    except Exception as e:
        print(f"  ❌ FAIL: {e}")
        failed_files.append((f, str(e)))

        # For first failure, show more details
        if len(failed_files) == 1:
            print("\nDETAILED ERROR:")
            try:
                load_edf_file(f)
            except Exception:
                traceback.print_exc()
            break

if failed_files:
    print("\n" + "=" * 60)
    print(f"FAILED FILES: {len(failed_files)}")
    for f, err in failed_files:
        print(f"  {f}: {err}")
else:
    print("\n✅ All files loaded successfully!")