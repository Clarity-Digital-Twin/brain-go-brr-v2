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
            data, sample_rate = load_edf_file(
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
            data, sample_rate = load_edf_file(
                Path(edf_file),
                target_channels=CHANNEL_NAMES_10_20,
            )

            if data is not None:
                # Should have all requested channels after interpolation
                assert data.shape[0] == len(CHANNEL_NAMES_10_20)
                assert sample_rate > 0

    def test_extreme_class_imbalance_in_dataset(self, data_dir):
        """Test REAL class imbalance in TUSZ dataset."""
        # This would need the full dataset loading infrastructure
        # For now, test class imbalance by loading raw label files
        label_files = list(data_dir.glob("**/*.tse"))[:10]
        if not label_files:
            pytest.skip("No TSE label files found")

        total_time = 0
        seizure_time = 0

        for label_file in label_files:
            # Read TSE file and count seizure vs background time
            with open(label_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start = float(parts[0])
                        end = float(parts[1])
                        label = parts[2]
                        duration = end - start
                        total_time += duration
                        if label == 'seiz':
                            seizure_time += duration

        if total_time > 0:
            seizure_ratio = seizure_time / total_time
            print(f"Seizure ratio: {seizure_ratio:.3f} ({seizure_time:.1f}s/{total_time:.1f}s)")
            # TUSZ is extremely imbalanced
            assert seizure_ratio < 0.15, "Expected extreme imbalance in TUSZ"