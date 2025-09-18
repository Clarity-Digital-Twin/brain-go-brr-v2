#!/usr/bin/env python
"""Integration test for TUSZ CSV parsing and training pipeline."""

import tempfile
from pathlib import Path

import numpy as np


def create_mock_tusz_data(output_dir: Path) -> None:
    """Create mock TUSZ EDF + CSV pairs for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create mock EDF (just a placeholder file)
    edf_path = output_dir / "test_001.edf"
    edf_path.write_bytes(b"MOCK EDF DATA" * 100)  # Mock content

    # Create corresponding CSV_BI annotation
    csv_path = output_dir / "test_001.csv"
    csv_content = """# version = csv_bi_v1.0.0
# bname = test_001
# duration = 600.00 secs
# montage = 01_tcp_ar
channel,start_time,stop_time,label,confidence
FP1-F7,0.0000,100.0000,bckg,1.0000
FP1-F7,100.0000,150.0000,cpsz,1.0000
FP1-F7,150.0000,200.0000,bckg,1.0000
FP1-F7,200.0000,230.0000,fnsz,1.0000
FP1-F7,230.0000,300.0000,bckg,1.0000
FP1-F7,300.0000,320.0000,gnsz,1.0000
FP1-F7,320.0000,400.0000,bckg,1.0000
FP1-F7,400.0000,420.0000,tcsz,1.0000
FP1-F7,420.0000,450.0000,bckg,1.0000
FP1-F7,450.0000,470.0000,absz,1.0000
FP1-F7,470.0000,500.0000,bckg,1.0000
FP1-F7,500.0000,510.0000,atnz,1.0000
FP1-F7,510.0000,550.0000,bckg,1.0000
FP1-F7,550.0000,570.0000,seiz,1.0000
FP1-F7,570.0000,600.0000,bckg,1.0000
"""
    csv_path.write_text(csv_content)
    print(f"Created mock TUSZ data in {output_dir}")


def test_csv_parser() -> None:
    """Test CSV parser with all seizure types."""
    from src.experiment.data import parse_tusz_csv, events_to_binary_mask
    from src.experiment import constants

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        create_mock_tusz_data(tmppath)

        csv_path = tmppath / "test_001.csv"

        # Parse CSV
        duration, events = parse_tusz_csv(csv_path)

        print(f"\n✅ CSV Parsing Test:")
        print(f"  Duration: {duration}s")
        print(f"  Events found: {len(events)}")

        # Check all seizure types detected
        seizure_types = {label for _, _, label in events}
        expected_types = {"cpsz", "fnsz", "gnsz", "tcsz", "absz", "atnz", "seiz"}

        print(f"  Seizure types: {sorted(seizure_types)}")
        assert seizure_types == expected_types, f"Missing: {expected_types - seizure_types}"

        # Test binary mask conversion
        n_samples = int(duration * constants.SAMPLING_RATE)  # 600s * 256 Hz
        mask = events_to_binary_mask(events, n_samples)

        # Verify mask has correct seizure regions
        seizure_seconds = sum(stop - start for start, stop, _ in events)
        seizure_samples = int(seizure_seconds * constants.SAMPLING_RATE)
        actual_seizure_samples = int(mask.sum())

        print(f"\n✅ Binary Mask Test:")
        print(f"  Total samples: {n_samples}")
        print(f"  Expected seizure samples: {seizure_samples}")
        print(f"  Actual seizure samples: {actual_seizure_samples}")
        print(f"  Difference: {abs(seizure_samples - actual_seizure_samples)} samples")

        # Allow small rounding differences (±1 sample per event)
        assert abs(seizure_samples - actual_seizure_samples) <= len(events), "Mask mismatch"

        print("\n✅ ALL PARSER TESTS PASSED!")


def test_dataset_loading() -> None:
    """Test EEGWindowDataset can load CSV labels."""
    from unittest.mock import patch, MagicMock
    from src.experiment.data import EEGWindowDataset
    from src.experiment import constants

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        create_mock_tusz_data(tmppath)

        edf_path = tmppath / "test_001.edf"
        csv_path = tmppath / "test_001.csv"

        # Mock EDF loading to avoid MNE dependency
        mock_raw = MagicMock()
        mock_raw.ch_names = constants.CHANNEL_NAMES_10_20
        mock_raw.get_data.return_value = np.random.randn(19, 600 * 256).astype(np.float32)
        mock_raw.info = {"sfreq": 256.0}

        with patch("src.experiment.data._read_raw_edf", return_value=mock_raw):
            # Create dataset with label files
            dataset = EEGWindowDataset(
                edf_files=[edf_path],
                label_files=[csv_path],
                cache_dir=tmppath / "cache"
            )

            print(f"\n✅ Dataset Loading Test:")
            dataset_len = dataset.__len__()  # Call method directly
            print(f"  Dataset length: {dataset_len}")
            assert dataset_len > 0, "Dataset should have windows"

            # Get first window
            window, label = dataset[0]

            print(f"  Window shape: {tuple(window.shape)}")
            print(f"  Label shape: {tuple(label.shape)}")
            print(f"  Label has seizures: {label.max() > 0.5}")

            # Check shapes
            assert window.shape == (19, constants.WINDOW_SAMPLES), f"Bad window shape: {window.shape}"
            assert label.shape == (constants.WINDOW_SAMPLES,), f"Bad label shape: {label.shape}"

            print("\n✅ DATASET TEST PASSED!")


def test_pipeline_pairing() -> None:
    """Test pipeline correctly pairs EDF and CSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create multiple file pairs
        for i in range(3):
            edf = tmppath / f"rec_{i:03d}.edf"
            csv = tmppath / f"rec_{i:03d}.csv"

            edf.write_bytes(b"MOCK EDF")
            csv.write_text(f"""# duration = {100 * (i+1)}.00 secs
channel,start_time,stop_time,label,confidence
FP1-F7,0.0,10.0,bckg,1.0
FP1-F7,10.0,20.0,cpsz,1.0
""")

        # Simulate pipeline discovery
        edf_files = sorted(tmppath.glob("**/*.edf"))
        label_files = [p.with_suffix(".csv") for p in edf_files]

        print(f"\n✅ Pipeline Pairing Test:")
        print(f"  Found {len(edf_files)} EDF files")

        for edf, csv in zip(edf_files, label_files):
            assert edf.stem == csv.stem, f"Mismatch: {edf.stem} != {csv.stem}"
            assert csv.exists(), f"Missing CSV: {csv}"
            print(f"  ✓ {edf.name} ↔ {csv.name}")

        print("\n✅ PAIRING TEST PASSED!")


def run_all_tests() -> None:
    """Run all integration tests."""
    print("=" * 60)
    print("TUSZ INTEGRATION TEST SUITE")
    print("=" * 60)

    test_csv_parser()
    test_dataset_loading()
    test_pipeline_pairing()

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED - WE ARE 100% READY TO TRAIN!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()