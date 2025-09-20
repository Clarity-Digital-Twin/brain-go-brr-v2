"""Integration tests for data IO edge cases with REAL TUSZ files."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.brain_brr.data.io import load_edf_file


@pytest.mark.integration
class TestRealTUSZFiles:
    """Test handling of REAL TUSZ files including corrupted ones."""

    @pytest.fixture
    def data_dir(self):
        """Get REAL TUSZ data directory."""
        data_dir = Path("data_ext4/tusz/edf/train")
        if not data_dir.exists():
            pytest.skip("TUSZ data not found - need real data for these tests")
        return data_dir

    def test_load_real_tusz_file(self, data_dir):
        """Test loading actual TUSZ EDF files."""
        # Find first EDF file
        edf_files = list(data_dir.glob("**/*.edf"))[:5]
        if not edf_files:
            pytest.skip("No EDF files found")

        for edf_file in edf_files:
            data, sample_rate, channel_names, start_time, annotations = load_edf_file(
                Path(edf_file),
                target_channels=None,  # Load all channels
            )

            # Verify data loaded correctly
            assert data is not None
            assert data.shape[0] > 0  # Has channels
            assert data.shape[1] > 0  # Has samples
            assert sample_rate > 0  # Has valid sample rate
            assert not np.isnan(data).any()
            assert not np.isinf(data).any()

    def test_missing_channels_interpolation(self, data_dir):
        """Test files with missing standard channels."""
        from src.brain_brr.constants import CHANNEL_NAMES_10_20

        edf_files = list(data_dir.glob("**/*.edf"))[:10]

        for edf_file in edf_files:
            data, sample_rate, channel_names, _, _ = load_edf_file(
                Path(edf_file),
                target_channels=CHANNEL_NAMES_10_20,
            )

            if data is not None:
                # Should have all requested channels after interpolation
                assert data.shape[0] == len(CHANNEL_NAMES_10_20)
                assert sample_rate > 0

    def test_extreme_class_imbalance_in_dataset(self, data_dir):
        """Test REAL class imbalance in TUSZ dataset."""
        from src.brain_brr.data.datasets import SeizureDataset

        # Use limited files for speed
        os.environ["BGB_LIMIT_FILES"] = "10"

        try:
            dataset = SeizureDataset(
                data_dir=str(data_dir),
                split="train",
                sampling_rate=256,
                window_size=60,
                stride=10,
                cache_dir="cache/test_imbalance",
            )

            # Sample some windows to check imbalance
            positive_windows = 0
            total_windows = min(50, len(dataset))

            for i in range(total_windows):
                _, label = dataset[i]
                if label.sum() > 0:  # Has any positive samples
                    positive_windows += 1

            positive_ratio = positive_windows / total_windows if total_windows > 0 else 0
            print(f"Positive ratio: {positive_ratio:.3f} ({positive_windows}/{total_windows})")

            # TUSZ is extremely imbalanced
            assert positive_ratio < 0.5, "Expected extreme imbalance in TUSZ"

        finally:
            if "BGB_LIMIT_FILES" in os.environ:
                del os.environ["BGB_LIMIT_FILES"]