"""Unit tests for RecordingStorage - Zero-copy verification.

Clean Code Principles:
- No mocks (test real behavior)
- Minimal setup (just what's needed)
- Clear assertions (exact memory targets)
- Single responsibility per test
"""

import gc
from pathlib import Path

import numpy as np
import psutil
import pytest
import torch

from src.brain_brr.train.recording_storage import RecordingStorage


@pytest.fixture
def sample_recording() -> tuple[torch.Tensor, torch.Tensor]:
    """Create one sample recording (19 channels x 122,880 samples = 2.33M samples)."""
    np.random.seed(42)
    probs = torch.from_numpy(np.random.rand(2_334_720).astype(np.float32))
    labels = torch.from_numpy(np.random.randint(0, 2, 2_334_720).astype(np.float32))
    return probs, labels


def get_rss_gb() -> float:
    """Get current RSS in GB."""
    return psutil.Process().memory_info().rss / (1024**3)


def force_gc() -> None:
    """Force garbage collection (3 passes for safety)."""
    for _ in range(3):
        gc.collect()


class TestRecordingStorageZeroAccumulation:
    """Verify RecordingStorage maintains 0GB resident memory during writes."""

    def test_write_100_recordings_no_accumulation(self, sample_recording: tuple) -> None:
        """Write 100 recordings (900MB on disk) and verify <50MB RSS increase."""
        force_gc()
        initial_rss = get_rss_gb()

        with RecordingStorage() as storage:
            for i in range(100):
                probs, labels = sample_recording
                storage.write_recording(f"rec_{i:03d}", probs, labels)

            force_gc()
            final_rss = get_rss_gb()
            rss_increase = final_rss - initial_rss

            assert rss_increase < 0.05, (
                f"RSS increased by {rss_increase:.3f}GB (expected <0.05GB). "
                f"Storage is accumulating in RAM instead of writing to disk!"
            )

            assert len(storage.recording_ids) == 100
            for i in range(100):
                assert (storage.cache_dir / f"rec_{i:03d}_probs.npy").exists()
                assert (storage.cache_dir / f"rec_{i:03d}_labels.npy").exists()

    def test_write_recording_cleans_up_transient(self, sample_recording: tuple) -> None:
        """Verify transient memory is freed after each write."""
        force_gc()

        with RecordingStorage() as storage:
            baseline_rss = get_rss_gb()

            for i in range(10):
                probs, labels = sample_recording
                storage.write_recording(f"rec_{i}", probs, labels)

                force_gc()
                current_rss = get_rss_gb()
                growth = current_rss - baseline_rss

                assert growth < 0.02, (
                    f"After {i + 1} writes: RSS grew {growth:.3f}GB. "
                    f"Transient buffers not being freed!"
                )


class TestRecordingStoragePreallocation:
    """Verify get_all_concatenated() uses pre-allocation (no double-buffer)."""

    def test_get_all_concatenated_single_pass(self, sample_recording: tuple) -> None:
        """Load 10 recordings and verify ~180MB allocation (not 360MB double-buffer)."""
        with RecordingStorage() as storage:
            for i in range(10):
                probs, labels = sample_recording
                storage.write_recording(f"rec_{i}", probs, labels)

            force_gc()
            before_rss = get_rss_gb()

            probs_all, labels_all = storage.get_all_concatenated()

            force_gc()
            after_rss = get_rss_gb()
            allocated = after_rss - before_rss

            assert 0.15 < allocated < 0.25, (
                f"Expected ~0.18GB allocation (pre-allocated), got {allocated:.3f}GB. "
                f"If >0.3GB, this indicates double-buffering (list+concat pattern)!"
            )

            assert probs_all.shape == (23_347_200,)
            assert labels_all.shape == (23_347_200,)
            assert probs_all.dtype == np.float32
            assert labels_all.dtype == np.float32

            del probs_all, labels_all

    def test_get_all_concatenated_data_integrity(self, sample_recording: tuple) -> None:
        """Verify pre-allocation doesn't corrupt data."""
        with RecordingStorage() as storage:
            test_data = []
            for i in range(50):
                probs = torch.arange(2_334_720, dtype=torch.float32) + i * 1000
                labels = torch.ones(2_334_720, dtype=torch.float32) * i
                storage.write_recording(f"rec_{i:02d}", probs, labels)
                test_data.append((probs.numpy(), labels.numpy()))

            probs_all, labels_all = storage.get_all_concatenated()

            offset = 0
            for i in range(50):
                n = 2_334_720
                probs_chunk = probs_all[offset : offset + n]
                labels_chunk = labels_all[offset : offset + n]

                np.testing.assert_array_almost_equal(probs_chunk, test_data[i][0], decimal=5)
                np.testing.assert_array_almost_equal(labels_chunk, test_data[i][1], decimal=5)

                offset += n

            del probs_all, labels_all


class TestRecordingStorageStreaming:
    """Verify iter_recordings() uses O(1) memory (memory-mapped)."""

    def test_iter_recordings_no_accumulation(self, sample_recording: tuple) -> None:
        """Iterate 100 recordings and verify no RSS growth."""
        with RecordingStorage() as storage:
            for i in range(100):
                probs, labels = sample_recording
                storage.write_recording(f"rec_{i:02d}", probs, labels)

            force_gc()
            initial_rss = get_rss_gb()

            count = 0
            for probs, labels in storage.iter_recordings():
                assert probs.shape == (2_334_720,)
                assert labels.shape == (2_334_720,)
                count += 1

            force_gc()
            final_rss = get_rss_gb()
            rss_change = abs(final_rss - initial_rss)

            assert rss_change < 0.05, (
                f"RSS changed by {rss_change:.3f}GB during iteration. "
                f"Expected O(1) memory with mmap, not accumulation!"
            )

            assert count == 100

    def test_iter_recordings_data_integrity(self, sample_recording: tuple) -> None:
        """Verify streaming doesn't corrupt data."""
        with RecordingStorage() as storage:
            test_data = []
            for i in range(50):
                probs = torch.rand(2_334_720) * i
                labels = torch.randint(0, 2, (2_334_720,)).float()
                storage.write_recording(f"rec_{i:02d}", probs, labels)
                test_data.append((probs.numpy(), labels.numpy()))

            for i, (probs, labels) in enumerate(storage.iter_recordings()):
                np.testing.assert_array_almost_equal(probs, test_data[i][0], decimal=5)
                np.testing.assert_array_almost_equal(labels, test_data[i][1], decimal=5)


class TestRecordingStorageZeroCopyTensors:
    """Verify get_all_as_torch_tensors() uses copy-on-write (zero-copy)."""

    def test_get_all_as_torch_tensors_minimal_allocation(self, sample_recording: tuple) -> None:
        """Get 10 recordings as tensors and verify <50MB allocation (not 180MB copies)."""
        with RecordingStorage() as storage:
            for i in range(10):
                probs, labels = sample_recording
                storage.write_recording(f"rec_{i}", probs, labels)

            force_gc()
            before_rss = get_rss_gb()

            probs_list, labels_list = storage.get_all_as_torch_tensors()

            force_gc()
            after_rss = get_rss_gb()
            allocated = after_rss - before_rss

            assert allocated < 0.05, (
                f"Expected <50MB allocation (zero-copy), got {allocated:.3f}GB. "
                f"If >0.1GB, mmap data is being copied into RAM!"
            )

            assert len(probs_list) == 10
            assert len(labels_list) == 10

            for probs_t in probs_list:
                assert probs_t.shape == (2_334_720,)
                assert probs_t.dtype == torch.float32

            del probs_list, labels_list

    def test_copy_on_write_tensors_are_writeable(self, sample_recording: tuple) -> None:
        """Verify mmap_mode='c' makes tensors writeable (PyTorch requirement)."""
        with RecordingStorage() as storage:
            probs, labels = sample_recording
            storage.write_recording("test", probs, labels)

            probs_list, labels_list = storage.get_all_as_torch_tensors()

            probs_t = probs_list[0]
            labels_t = labels_list[0]

            assert probs_t.numpy().flags.writeable, "Tensor must be writeable for PyTorch"
            assert labels_t.numpy().flags.writeable, "Tensor must be writeable for PyTorch"

            del probs_list, labels_list

    def test_read_only_operations_dont_trigger_copy(self, sample_recording: tuple) -> None:
        """Verify read-only operations (like FA sweep) don't trigger copy-on-write."""
        with RecordingStorage() as storage:
            probs, labels = sample_recording
            storage.write_recording("test", probs, labels)

            original_probs = np.load(storage.cache_dir / "test_probs.npy")
            original_checksum = original_probs.sum()

            probs_list, _ = storage.get_all_as_torch_tensors()
            probs_t = probs_list[0]

            threshold_result = probs_t >= 0.5

            assert threshold_result.shape == probs_t.shape
            assert threshold_result.dtype == torch.bool

            reloaded_probs = np.load(storage.cache_dir / "test_probs.npy")
            reloaded_checksum = reloaded_probs.sum()

            assert abs(original_checksum - reloaded_checksum) < 1e-5, (
                "Original file was modified! Copy-on-write should prevent this."
            )

            del probs_list, threshold_result


class TestRecordingStorageContextManager:
    """Verify context manager cleanup."""

    def test_context_manager_cleanup(self, sample_recording: tuple, tmp_path: Path) -> None:
        """Verify __exit__ cleans up temp directory."""
        cache_dir = tmp_path / "test_cache"

        with RecordingStorage(cache_dir=cache_dir) as storage:
            probs, labels = sample_recording
            storage.write_recording("test", probs, labels)

            assert cache_dir.exists()
            assert (cache_dir / "test_probs.npy").exists()

        assert not cache_dir.exists(), "Context manager should clean up cache_dir"

    def test_automatic_temp_cleanup(self, sample_recording: tuple) -> None:
        """Verify automatic temp directory cleanup (None cache_dir)."""
        temp_path = None

        with RecordingStorage() as storage:
            probs, labels = sample_recording
            storage.write_recording("test", probs, labels)

            temp_path = storage.cache_dir
            assert temp_path.exists()

        assert not temp_path.exists(), "Temp directory should be cleaned up"


class TestRecordingStorageEdgeCases:
    """Edge cases and error conditions."""

    def test_empty_storage(self) -> None:
        """Verify empty storage behaves correctly."""
        with RecordingStorage() as storage:
            assert len(storage.recording_ids) == 0

            probs_all, labels_all = storage.get_all_concatenated()
            assert probs_all.shape == (0,)
            assert labels_all.shape == (0,)

            count = sum(1 for _ in storage.iter_recordings())
            assert count == 0

            probs_list, labels_list = storage.get_all_as_torch_tensors()
            assert len(probs_list) == 0
            assert len(labels_list) == 0

    def test_non_contiguous_tensors(self) -> None:
        """Verify non-contiguous tensors are handled correctly."""
        with RecordingStorage() as storage:
            base = torch.randn(4_669_440)
            probs = base[::2]
            labels = torch.zeros(2_334_720)

            assert not probs.is_contiguous(), "Test requires non-contiguous tensor"
            assert probs.shape == (2_334_720,)

            storage.write_recording("test", probs, labels)

            probs_loaded, _ = storage.get_all_concatenated()
            assert probs_loaded.shape == (2_334_720,)

            torch.testing.assert_close(
                torch.from_numpy(probs_loaded), probs.contiguous(), rtol=1e-5, atol=1e-5
            )
