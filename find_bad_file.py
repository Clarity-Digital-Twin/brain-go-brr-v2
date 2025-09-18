#!/usr/bin/env python3
"""Find the file that's missing Fz/Pz channels."""

from pathlib import Path
from src.experiment.data import load_edf_file
import sys

data_root = Path('data/tusz/edf/dev')
edf_files = sorted(data_root.glob("**/*.edf"))

# Split exactly as pipeline does
val_split = int(len(edf_files) * 0.2)
train_files = edf_files[val_split:]

print(f"Testing {len(train_files)} training files...")
print(f"First file: {train_files[0]}")
print(f"Last file: {train_files[-1]}")
print("=" * 60)

# Test each file
for i, f in enumerate(train_files):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(train_files)}...")

    try:
        data, fs = load_edf_file(f)
    except ValueError as e:
        if "Missing required channels" in str(e):
            print(f"\n❌ FOUND THE PROBLEM FILE!")
            print(f"File #{i}: {f}")
            print(f"Error: {e}")
            print(f"\nFull path: {f.absolute()}")
            print(f"Parent dir: {f.parent}")

            # Check what channels it has
            import mne
            mne.set_log_level('ERROR')
            raw = mne.io.read_raw_edf(f, preload=False, verbose=False)
            print(f"\nActual channels in file ({len(raw.ch_names)}):")
            for ch in raw.ch_names:
                if 'FZ' in ch.upper() or 'PZ' in ch.upper() or 'CZ' in ch.upper():
                    print(f"  - {ch}")

            sys.exit(1)
    except Exception as e:
        print(f"Other error on file {f}: {e}")
        sys.exit(1)

print("\n✅ All files loaded successfully!")