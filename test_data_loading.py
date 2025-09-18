#!/usr/bin/env python
"""Quick test of data loading with interpolation."""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("Testing data loading with interpolation...")

# Find first EDF file
data_dir = Path("data/tusz/edf/dev")
edf_files = list(data_dir.glob("**/*.edf"))[:5]  # Just test 5 files

print(f"Found {len(edf_files)} EDF files to test")

if edf_files:
    from src.experiment.data import load_edf_file

    for i, edf_file in enumerate(edf_files):
        print(f"\n[{i+1}/{len(edf_files)}] Testing {edf_file.name}...")
        try:
            data, fs = load_edf_file(edf_file)
            print(f"  ✓ Success! Shape: {data.shape}, Sampling rate: {fs} Hz")
        except Exception as e:
            print(f"  ✗ Error: {e}")

print("\nDone!")