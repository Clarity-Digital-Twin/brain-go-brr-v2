#!/usr/bin/env python3
"""Test loading EDF file directly with our load_edf_file function."""

from pathlib import Path
from src.experiment.data import load_edf_file

# Get first dev file
dev_files = list(Path('data/tusz/edf/dev').glob('**/*.edf'))
test_file = dev_files[0]

print(f"Testing: {test_file}")
print("=" * 60)

try:
    data, fs = load_edf_file(test_file)
    print(f"✅ SUCCESS!")
    print(f"Data shape: {data.shape}")
    print(f"Sampling rate: {fs}")
except Exception as e:
    print(f"❌ FAILED: {e}")

    # Try with verbose debugging
    print("\nDEBUGGING:")
    import mne
    mne.set_log_level('ERROR')

    raw = mne.io.read_raw_edf(test_file, preload=False, verbose=False)
    print(f"Raw channel names: {raw.ch_names[:5]} ... {raw.ch_names[-5:]}")

    # Try canonicalization manually
    from src.experiment import constants

    # Build uppercase → canonical map
    upper_to_canon = {c.upper(): c for c in constants.CHANNEL_NAMES_10_20}
    upper_to_canon.update({"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"})

    def _to_canonical(name: str) -> str | None:
        s = name.strip().upper()

        # Filter non-EEG
        non_eeg_prefixes = ("EKG", "ECG", "EOG", "EMG", "RESP", "PHOTIC", "IBI", "BURSTS", "SUPPR", "LOC", "ROC")
        if any(s.startswith(p) for p in non_eeg_prefixes):
            return None

        # Handle EEG prefix
        if s.startswith("EEG "):
            s = s[4:]

        # Remove suffixes
        for suf in ("-REF", "-LE", "-AR", "-AVG", "-DC"):
            if s.endswith(suf):
                s = s[: -len(suf)]
                break

        return upper_to_canon.get(s)

    rename_map = {}
    for ch in raw.ch_names:
        canonical = _to_canonical(ch)
        if canonical and canonical != ch:
            rename_map[ch] = canonical

    print(f"Rename map entries: {len(rename_map)}")

    # Check for Fz/Pz specifically
    fz_entries = {k: v for k, v in rename_map.items() if 'Fz' in v}
    pz_entries = {k: v for k, v in rename_map.items() if 'Pz' in v}
    print(f"Fz mappings: {fz_entries}")
    print(f"Pz mappings: {pz_entries}")

    # Try renaming
    try:
        raw.rename_channels(rename_map)
        print(f"After rename: {sorted(raw.ch_names)[:5]} ... {sorted(raw.ch_names)[-5:]}")

        # Check if our channels are there
        target = constants.CHANNEL_NAMES_10_20
        available = [ch for ch in target if ch in raw.ch_names]
        missing = set(target) - set(available)
        print(f"Available: {len(available)}/{len(target)}")
        print(f"Missing: {missing}")
    except Exception as e:
        print(f"Rename failed: {e}")