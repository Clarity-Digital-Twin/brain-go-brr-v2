#!/usr/bin/env python
"""Test channel selection fix on all montage types."""

from pathlib import Path
from src.experiment.data import load_edf_file
from src.experiment.constants import CHANNEL_NAMES_10_20
import traceback


def test_montage_files():
    """Test one file from each montage type."""
    test_patterns = {
        'tcp_ar': 'data/tusz/edf/dev/*/s*/01_tcp_ar/*.edf',
        'tcp_le': 'data/tusz/edf/dev/*/s*/02_tcp_le/*.edf',
        'tcp_ar_a': 'data/tusz/edf/dev/*/s*/03_tcp_ar_a/*.edf'
    }

    results = {}
    for montage, pattern in test_patterns.items():
        files = list(Path('.').glob(pattern))[:3]  # Test 3 files per montage
        if not files:
            print(f"⚠️  {montage}: No files found")
            results[montage] = "NO_FILES"
            continue

        success_count = 0
        for f in files:
            try:
                print(f"  Testing {f.name}...", end=" ")
                data, fs = load_edf_file(f)
                assert data.shape[0] == 19, f"Expected 19 channels, got {data.shape[0]}"
                assert fs > 0, f"Invalid sampling rate: {fs}"
                print(f"✅ Shape: {data.shape}, Fs: {fs} Hz")
                success_count += 1
            except Exception as e:
                print(f"❌ Error: {e}")
                traceback.print_exc()

        if success_count == len(files):
            print(f"✅ {montage}: All {success_count} files loaded successfully!")
            results[montage] = "SUCCESS"
        else:
            print(f"⚠️  {montage}: {success_count}/{len(files)} files loaded")
            results[montage] = f"PARTIAL_{success_count}/{len(files)}"

    print("\n=== SUMMARY ===")
    for montage, status in results.items():
        print(f"{montage}: {status}")

    return all(v == "SUCCESS" for v in results.values())


if __name__ == "__main__":
    print("Testing channel selection fix on TUSZ montages...")
    print(f"Target channels: {CHANNEL_NAMES_10_20}")
    print()

    success = test_montage_files()
    if success:
        print("\n🎉 ALL TESTS PASSED! Channel selection fix is working!")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")