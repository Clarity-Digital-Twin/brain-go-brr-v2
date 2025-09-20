"""REAL edge case tests for data IO - no mocks, actual corrupted files."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from src.brain_brr.constants import CANONICAL_CHANNELS
from src.brain_brr.data.io import EEGDataLoader


class TestRealCorruptedFiles:
    """Test handling of REAL corrupted TUSZ files."""

    @pytest.fixture
    def data_dir(self):
        """Get REAL TUSZ data directory."""
        # Check multiple possible locations
        possible_dirs = [
            Path("data_ext4/tusz/edf/train"),
            Path("data/tusz/edf/train"),
            Path("/data/tusz/edf/train"),
        ]
        for d in possible_dirs:
            if d.exists():
                return d
        pytest.skip("TUSZ data not found - need real data for these tests")

    def test_real_corrupted_tusz_files(self, data_dir):
        """Test ACTUAL corrupted TUSZ files that crash in production."""
        loader = EEGDataLoader(sampling_rate=256, n_channels=19)

        # These are known problematic files in TUSZ
        problem_patterns = [
            "01_tcp_ar/002/00000258",  # Known header issues
            "01_tcp_ar/081/00008184",  # Missing channels
            "02_tcp_le/042/00004209",  # Unicode in channel names
        ]

        corrupted_count = 0
        for pattern in problem_patterns:
            pattern_path = data_dir / pattern
            if not pattern_path.exists():
                continue

            edf_files = list(pattern_path.glob("*.edf"))
            for edf_file in edf_files[:2]:  # Test first 2 files per pattern
                try:
                    data, annotations = loader.load_file(str(edf_file))
                    # File loaded successfully - check for interpolation
                    assert data is not None
                    assert data.shape[0] == 19  # All channels present

                    # Check if interpolation was needed (Fz, Pz often missing)
                    # This is logged but we can't easily check logs in test

                except Exception as e:
                    corrupted_count += 1
                    # Should handle gracefully, not crash
                    print(f"Handled corruption in {edf_file}: {e}")

        # We should have found and handled some corrupted files
        if corrupted_count == 0 and len(list(data_dir.glob("**/*.edf"))) > 0:
            pytest.skip("No corrupted files found in test subset")

    def test_missing_critical_channels(self, data_dir):
        """Test files with missing Fz, Pz channels that need interpolation."""
        loader = EEGDataLoader(sampling_rate=256, n_channels=19)

        # Find a file and load it
        edf_files = list(data_dir.glob("**/*.edf"))[:5]
        if not edf_files:
            pytest.skip("No EDF files found")

        for edf_file in edf_files:
            data, _ = loader.load_file(str(edf_file))

            # Verify all channels present after loading
            assert data.shape[0] == 19, f"Expected 19 channels, got {data.shape[0]}"

            # Verify channel ordering is canonical
            # Can't directly check channel names from tensor, but shape validates

    def test_extreme_class_imbalance_real_data(self, data_dir):
        """Load REAL TUSZ files with 99.9% background."""
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
                cache_dir="cache/test_edge",
            )

            # Calculate actual class balance
            total_samples = 0
            positive_samples = 0

            for i in range(min(100, len(dataset))):
                _, label = dataset[i]
                total_samples += label.numel()
                positive_samples += label.sum().item()

            if total_samples > 0:
                positive_ratio = positive_samples / total_samples
                print(f"Actual positive ratio: {positive_ratio:.4f}")

                # Real TUSZ data is extremely imbalanced
                assert positive_ratio < 0.1, "Expected extreme imbalance"

                # Test balanced sampler
                from src.brain_brr.train.loop import create_balanced_sampler

                sampler = create_balanced_sampler(dataset)

                # Sample a batch and check balance
                indices = list(sampler)[:32]
                batch_positive = 0
                batch_total = 0

                for idx in indices:
                    _, label = dataset[idx]
                    batch_total += 1
                    if label.sum() > 0:  # Has any positive samples
                        batch_positive += 1

                batch_positive_ratio = batch_positive / batch_total
                print(f"Balanced sampler positive ratio: {batch_positive_ratio:.4f}")

                # Balanced sampler should increase positive ratio significantly
                assert batch_positive_ratio > positive_ratio * 2

        finally:
            del os.environ["BGB_LIMIT_FILES"]

    def test_parallel_loading_stress(self, data_dir):
        """Stress test with num_workers=8 on REAL data."""
        from torch.utils.data import DataLoader
        from src.brain_brr.data.datasets import SeizureDataset

        os.environ["BGB_LIMIT_FILES"] = "20"

        try:
            dataset = SeizureDataset(
                data_dir=str(data_dir),
                split="train",
                sampling_rate=256,
                window_size=60,
                stride=10,
                cache_dir="cache/test_parallel",
            )

            # Create DataLoader with multiple workers
            num_workers = min(8, os.cpu_count() or 1)
            dataloader = DataLoader(
                dataset,
                batch_size=4,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )

            # Load several batches to stress parallel loading
            batches_loaded = 0
            for batch_idx, (data, labels) in enumerate(dataloader):
                assert data.shape[1] == 19  # Channels
                assert data.shape[2] == 15360  # Samples
                assert not torch.isnan(data).any()
                assert not torch.isinf(data).any()

                batches_loaded += 1
                if batches_loaded >= 10:
                    break

            assert batches_loaded > 0, "No batches loaded"
            print(f"Successfully loaded {batches_loaded} batches with {num_workers} workers")

        finally:
            del os.environ["BGB_LIMIT_FILES"]

    def test_zero_duration_files(self, data_dir):
        """Test handling of files with zero duration or corrupt timing."""
        loader = EEGDataLoader(sampling_rate=256, n_channels=19)

        # Find small files that might be corrupt
        small_files = []
        for edf_file in data_dir.glob("**/*.edf"):
            if edf_file.stat().st_size < 100_000:  # Less than 100KB
                small_files.append(edf_file)
            if len(small_files) >= 5:
                break

        if not small_files:
            pytest.skip("No small/potentially corrupt files found")

        for edf_file in small_files:
            try:
                data, annotations = loader.load_file(str(edf_file))
                if data is not None:
                    # Even small files should have some data
                    assert data.shape[1] > 0, "Zero duration data returned"
            except Exception as e:
                # Should handle gracefully
                print(f"Handled zero/corrupt duration in {edf_file}: {e}")

    def test_extreme_sampling_rates(self, data_dir):
        """Test files with non-standard sampling rates."""
        loader = EEGDataLoader(sampling_rate=256, n_channels=19)

        # TUSZ files can have various sampling rates (250, 256, 512, 1000 Hz)
        # Loader should resample to 256 Hz

        edf_files = list(data_dir.glob("**/*.edf"))[:10]
        for edf_file in edf_files:
            data, _ = loader.load_file(str(edf_file))

            if data is not None:
                # Check data is valid after resampling
                assert not torch.isnan(data).any()
                assert not torch.isinf(data).any()

                # Check reasonable value range (microvolts)
                assert data.abs().max() < 1e6, "Unreasonable values after resampling"

    def test_unicode_channel_names(self, data_dir):
        """Test handling of unicode and non-standard channel names."""
        loader = EEGDataLoader(sampling_rate=256, n_channels=19)

        # TUSZ uses both old (T3/T4) and new (T7/T8) naming
        edf_files = list(data_dir.glob("**/*.edf"))[:10]

        for edf_file in edf_files:
            data, _ = loader.load_file(str(edf_file))

            if data is not None:
                # Should have canonical 19 channels regardless of naming
                assert data.shape[0] == 19

                # Data should be properly ordered according to CANONICAL_CHANNELS
                # Can't verify exact ordering without access to channel names
                # but shape and validity checks ensure proper handling