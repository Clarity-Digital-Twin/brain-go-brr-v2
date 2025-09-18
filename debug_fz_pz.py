#!/usr/bin/env python3
"""Debug script to investigate missing Fz/Pz channels."""

import mne
from pathlib import Path
import sys

# Suppress MNE verbose output
mne.set_log_level('ERROR')

# Get first dev file
dev_files = list(Path('data/tusz/edf/dev').glob('**/*.edf'))
if not dev_files:
    print("No EDF files found in data/tusz/edf/dev/")
    sys.exit(1)

test_file = dev_files[0]
print(f"Testing file: {test_file}")
print(f"Total dev files: {len(dev_files)}")
print("=" * 60)

# Load with MNE
raw = mne.io.read_raw_edf(test_file, preload=False, verbose=False)
print(f"\nChannel count: {len(raw.ch_names)}")
print(f"\nFirst 10 channels: {raw.ch_names[:10]}")
print(f"Last 10 channels: {raw.ch_names[-10:]}")

# Check for Fz/Pz variations
print("\n" + "=" * 60)
print("SEARCHING FOR Fz/Pz VARIANTS:")
fz_variants = [ch for ch in raw.ch_names if 'FZ' in ch.upper()]
pz_variants = [ch for ch in raw.ch_names if 'PZ' in ch.upper()]
cz_variants = [ch for ch in raw.ch_names if 'CZ' in ch.upper()]

print(f"\nFz variants found: {fz_variants}")
print(f"Pz variants found: {pz_variants}")
print(f"Cz variants found (for comparison): {cz_variants}")

# Import our canonicalization
print("\n" + "=" * 60)
print("TESTING OUR CANONICALIZATION:")

from src.experiment import constants

# Recreate the canonicalization logic to debug
def debug_canonical(name: str) -> str | None:
    """Debug version of canonicalization with print statements."""
    original = name
    s = name.strip().upper()

    # Filter out known non-EEG channels early
    non_eeg_prefixes = (
        "EKG", "ECG", "EOG", "EMG", "RESP", "PHOTIC", "IBI",
        "BURSTS", "SUPPR", "LOC", "ROC"
    )

    if any(s.startswith(p) for p in non_eeg_prefixes):
        print(f"  FILTERED (non-EEG): {original}")
        return None

    # Handle "EEG " prefix
    if s.startswith("EEG "):
        s = s[4:]
        print(f"  Stripped 'EEG ': {original} -> {s}")

    # Remove reference suffixes
    for suf in ("-REF", "-LE", "-AR", "-AVG", "-DC"):
        if s.endswith(suf):
            before = s
            s = s[: -len(suf)]
            print(f"  Stripped suffix '{suf}': {before} -> {s}")
            break

    # Map to canonical
    upper_to_canon = {ch.upper(): ch for ch in constants.CHANNEL_NAMES_10_20}
    # Add common synonyms
    upper_to_canon.update({
        "T7": "T3",
        "T8": "T4",
        "P7": "T5",
        "P8": "T6",
    })

    result = upper_to_canon.get(s)
    if result and result in ['Fz', 'Pz', 'Cz']:
        print(f"  ✓ MAPPED: {original} -> {result}")
    return result

# Test canonicalization on all channels
canonical_names = {}
for ch in raw.ch_names:
    canon = debug_canonical(ch)
    if canon:
        canonical_names[ch] = canon

print(f"\n{len(canonical_names)} channels mapped to canonical names")

# Check what's missing
missing = sorted(set(constants.CHANNEL_NAMES_10_20) - set(canonical_names.values()))
print(f"\nMISSING CHANNELS: {missing}")

if missing:
    print("\n" + "=" * 60)
    print("INVESTIGATING WHY THESE ARE MISSING:")
    for m in missing:
        print(f"\nLooking for {m}:")
        # Find any channel containing this pattern
        pattern = m.upper()
        matches = [ch for ch in raw.ch_names if pattern in ch.upper()]
        print(f"  Channels containing '{pattern}': {matches}")

        # Check if it's there but not canonicalizing
        for ch in matches:
            print(f"  Testing canonicalization of '{ch}':")
            result = debug_canonical(ch)
            if not result:
                print(f"    -> Failed to canonicalize!")

print("\n" + "=" * 60)
print("SUMMARY:")
print(f"File: {test_file}")
print(f"Total channels in file: {len(raw.ch_names)}")
print(f"Channels mapped: {len(canonical_names)}")
print(f"Required channels: {len(constants.CHANNEL_NAMES_10_20)}")
print(f"Missing: {missing if missing else 'None! ✓'}")