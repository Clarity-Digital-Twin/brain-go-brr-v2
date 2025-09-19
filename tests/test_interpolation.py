"""Test interpolation of missing Fz/Pz channels."""

from unittest.mock import patch

import mne
import numpy as np
import pytest


def test_interpolation_fz_pz_missing():
    """Test that missing Fz/Pz channels are interpolated correctly."""
    from pathlib import Path

    from src.brain_brr.data import load_edf_file

    # Create synthetic Raw with 17 channels (missing Fz, Pz)
    sfreq = 256.0
    n_samples = 1000
    channels_without_fz_pz = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Cz",
    ]

    # Create realistic-looking EEG data
    np.random.seed(42)
    data = np.random.randn(len(channels_without_fz_pz), n_samples) * 50  # µV scale

    info = mne.create_info(ch_names=channels_without_fz_pz, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data / 1e6, info)  # MNE uses V, not µV
    raw.set_montage("standard_1020", on_missing="ignore")

    # Mock _read_raw_edf to return our synthetic Raw
    with patch("src.brain_brr.data.io._read_raw_edf") as mock_read:
        mock_read.return_value = raw

        # Call load_edf_file
        result_data, result_fs = load_edf_file(Path("dummy.edf"))

        # Verify all 19 channels are present
        assert result_data.shape[0] == 19, f"Expected 19 channels, got {result_data.shape[0]}"
        assert result_fs == sfreq

        # Verify Fz and Pz were added and have finite values
        # The order should match CHANNEL_NAMES_10_20 from constants.py
        from src.brain_brr.constants import CHANNEL_NAMES_10_20

        fz_idx = CHANNEL_NAMES_10_20.index("Fz")
        pz_idx = CHANNEL_NAMES_10_20.index("Pz")

        assert np.all(np.isfinite(result_data[fz_idx])), "Fz has non-finite values"
        assert np.all(np.isfinite(result_data[pz_idx])), "Pz has non-finite values"

        # Verify interpolated channels have non-zero values
        assert not np.allclose(result_data[fz_idx], 0), "Fz is all zeros after interpolation"
        assert not np.allclose(result_data[pz_idx], 0), "Pz is all zeros after interpolation"


def test_interpolation_other_channel_missing_raises():
    """Test that missing channels other than Fz/Pz raise an error."""
    from pathlib import Path

    from src.brain_brr.data import load_edf_file

    # Create synthetic Raw missing O1 (not allowed)
    sfreq = 256.0
    n_samples = 1000
    channels_without_o1 = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Fz",
        "Cz",
        "Pz",  # Missing O1
    ]

    data = np.random.randn(len(channels_without_o1), n_samples) * 50
    info = mne.create_info(ch_names=channels_without_o1, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data / 1e6, info)

    with patch("src.brain_brr.data.io._read_raw_edf") as mock_read:
        mock_read.return_value = raw

        # Should raise ValueError for missing O1
        with pytest.raises(ValueError, match=r"Missing required channels.*O1"):
            load_edf_file(Path("dummy.edf"))


def test_interpolation_with_montage_disabled():
    """Test that interpolation fails gracefully when montage is disabled."""
    from pathlib import Path

    from src.brain_brr.data import load_edf_file

    # Create synthetic Raw missing Fz
    sfreq = 256.0
    n_samples = 1000
    channels_without_fz = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Cz",
        "Pz",
    ]

    data = np.random.randn(len(channels_without_fz), n_samples) * 50
    info = mne.create_info(ch_names=channels_without_fz, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data / 1e6, info)

    with patch("src.brain_brr.data.io._read_raw_edf") as mock_read:
        mock_read.return_value = raw

        # Should raise ValueError when montage is disabled
        with pytest.raises(ValueError, match="montage disabled"):
            load_edf_file(Path("dummy.edf"), apply_montage=False)


def test_interpolation_warning_logged(caplog):
    """Test that interpolation emits a warning."""
    import logging
    from pathlib import Path

    from src.brain_brr.data import load_edf_file

    # Set up logging to capture warnings
    caplog.set_level(logging.WARNING)

    # Create synthetic Raw missing both Fz and Pz
    sfreq = 256.0
    n_samples = 1000
    channels = [
        "Fp1",
        "Fp2",
        "F3",
        "F4",
        "C3",
        "C4",
        "P3",
        "P4",
        "O1",
        "O2",
        "F7",
        "F8",
        "T3",
        "T4",
        "T5",
        "T6",
        "Cz",
    ]

    data = np.random.randn(len(channels), n_samples) * 50
    info = mne.create_info(ch_names=channels, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data / 1e6, info)
    raw.set_montage("standard_1020", on_missing="ignore")

    with patch("src.brain_brr.data.io._read_raw_edf") as mock_read:
        mock_read.return_value = raw

        # Call load_edf_file
        load_edf_file(Path("test_file.edf"))

        # Check that warning was logged
        assert any("Interpolated channels" in record.message for record in caplog.records)
        assert any("['Fz', 'Pz']" in record.message for record in caplog.records)
        assert any("test_file.edf" in record.message for record in caplog.records)
