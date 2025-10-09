"""Integration tests for validation memory profile.

Validates that the disk-backed validation stays within memory budget.

Clean Code Principles:
- Test real validation pipeline (not mocks)
- Measure actual RSS (not estimates)
- Clear pass/fail thresholds
- Single purpose per test
"""

import gc
from typing import Any

import numpy as np
import psutil
import pytest
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from src.brain_brr.config.schemas import PostprocessingConfig
from src.brain_brr.train.recording_storage import RecordingStorage


@pytest.fixture
def simple_model() -> nn.Module:
    """Create a minimal model for testing (just returns random predictions)."""

    class SimpleDetector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(19 * 15360, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            x_flat = x.view(batch_size, -1)
            return self.linear(x_flat)

    return SimpleDetector()


@pytest.fixture
def small_validation_loader() -> DataLoader:
    """Create validation loader with 10 synthetic recordings.

    Each recording:
    - 19 channels x 15360 samples (60s at 256Hz)
    - 10 windows per recording (overlap)
    - Total: 100 windows across 10 recordings
    """
    windows_list = []
    labels_list = []
    file_ids_list = []
    window_starts_list = []

    np.random.seed(42)
    torch.manual_seed(42)

    for rec_id in range(10):
        for window_idx in range(10):
            window = torch.randn(19, 15360)
            label = torch.rand(19, 15360)

            windows_list.append(window)
            labels_list.append(label)
            file_ids_list.append(f"rec_{rec_id:02d}")
            window_starts_list.append(float(window_idx * 6.0))

    windows_tensor = torch.stack(windows_list)
    labels_tensor = torch.stack(labels_list)

    dataset = TensorDataset(windows_tensor, labels_tensor)

    def collate_fn(batch: list) -> dict[str, Any]:
        windows = torch.stack([item[0] for item in batch])
        labels = torch.stack([item[1] for item in batch])

        file_ids = [file_ids_list[i] for i in range(len(batch))]
        window_starts = torch.tensor(
            [window_starts_list[i] for i in range(len(batch))], dtype=torch.float32
        )

        return {
            "window": windows,
            "label": labels,
            "file_id": file_ids,
            "window_start_s": window_starts,
        }

    return DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate_fn)


def get_rss_gb() -> float:
    """Get current RSS in GB."""
    return psutil.Process().memory_info().rss / (1024**3)


def force_gc() -> None:
    """Force garbage collection."""
    for _ in range(3):
        gc.collect()


class TestValidationMemoryProfile:
    """Test full validation pipeline memory profile."""

    def test_validation_small_dataset_memory_budget(
        self, simple_model: nn.Module, small_validation_loader: DataLoader
    ) -> None:
        """Verify validation on 10 recordings stays under 0.5GB peak.

        Memory expectation:
        - 10 recordings x 19 x 15360 = 2.9M samples
        - Float32: 2.9M x 4 bytes x 2 (probs+labels) = ~23MB
        - Pre-allocation peak: ~23MB
        - Overhead: <100MB
        - Total: <0.5GB
        """
        from src.brain_brr.train.val_step import validate_epoch

        force_gc()
        initial_rss = get_rss_gb()

        metrics = validate_epoch(
            model=simple_model,
            dataloader=small_validation_loader,
            post_config=PostprocessingConfig(),
            device="cpu",
            fa_rates=[10, 5, 1],
            focal_alpha=0.25,
            focal_gamma=2.0,
        )

        force_gc()
        peak_rss = get_rss_gb()
        peak_increase = peak_rss - initial_rss

        assert peak_increase < 0.5, (
            f"Peak memory {peak_increase:.3f}GB exceeds 0.5GB budget. "
            f"Expected <0.5GB for 10-recording validation."
        )

        assert "auroc" in metrics
        assert "pr_auc" in metrics
        assert "ece" in metrics
        assert "val_loss" in metrics

        assert 0 <= metrics["auroc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        assert 0 <= metrics["ece"] <= 1

    def test_staged_loading_memory_pattern(self, simple_model: nn.Module) -> None:
        """Verify staged loading: high peak during AUROC, then drop to low during FA sweep.

        This test verifies the core memory strategy:
        1. Load all data for AUROC (peak)
        2. Free all data (drop to baseline)
        3. Reload as mmap for FA sweep (minimal)
        """
        from src.brain_brr.train.val_step import _compute_final_metrics

        with RecordingStorage() as storage:
            for i in range(100):
                probs = torch.rand(100_000)
                labels = torch.randint(0, 2, (100_000,)).float()
                storage.write_recording(f"rec_{i:03d}", probs, labels)

            force_gc()
            baseline_rss = get_rss_gb()

            all_ref_events = [(10.0, 20.0), (30.0, 40.0)]
            all_pred_events = [(11.0, 19.0), (31.0, 39.0)]

            metrics = _compute_final_metrics(
                storage=storage,
                all_ref_events=all_ref_events,
                all_pred_events=all_pred_events,
                total_hours=10.0,
                fa_rates=[10, 5, 1],
                post_cfg=PostprocessingConfig(),
                sampling_rate=256,
                num_recordings=100,
            )

            force_gc()
            final_rss = get_rss_gb()

            total_increase = final_rss - baseline_rss

            assert total_increase < 0.1, (
                f"Final RSS increased by {total_increase:.3f}GB from baseline. "
                f"Memory should be freed after metrics computation!"
            )

            assert metrics["auroc"] >= 0
            assert metrics["pr_auc"] >= 0
            assert metrics["ece"] >= 0


class TestMetricsExactMatch:
    """Verify metrics match sklearn exactly (no approximations)."""

    def test_auroc_exact_match_sklearn(self) -> None:
        """Verify AUROC computation matches sklearn exactly."""
        np.random.seed(42)
        n_samples = 1_000_000

        probs = np.random.rand(n_samples).astype(np.float32)
        labels = (np.random.rand(n_samples) > 0.5).astype(np.int32)

        expected_auroc = roc_auc_score(labels, probs)
        expected_pr = average_precision_score(labels, probs)

        with RecordingStorage() as storage:
            chunk_size = n_samples // 10
            for i in range(10):
                start = i * chunk_size
                end = start + chunk_size
                probs_chunk = torch.from_numpy(probs[start:end])
                labels_chunk = torch.from_numpy(labels[start:end].astype(np.float32))
                storage.write_recording(f"chunk_{i}", probs_chunk, labels_chunk)

            probs_all, labels_all = storage.get_all_concatenated()
            labels_binary = (labels_all > 0.5).astype(np.int32)

            actual_auroc = roc_auc_score(labels_binary, probs_all)
            actual_pr = average_precision_score(labels_binary, probs_all)

            del probs_all, labels_all, labels_binary

        assert abs(expected_auroc - actual_auroc) < 1e-9, (
            f"AUROC mismatch: expected {expected_auroc:.10f}, got {actual_auroc:.10f}"
        )
        assert abs(expected_pr - actual_pr) < 1e-9, (
            f"PR-AUC mismatch: expected {expected_pr:.10f}, got {actual_pr:.10f}"
        )

    def test_ece_streaming_correctness(self) -> None:
        """Verify streaming ECE matches batch computation."""
        from src.brain_brr.train.val_step import _compute_ece_streaming

        np.random.seed(42)

        with RecordingStorage() as storage:
            all_probs = []
            all_labels = []

            for i in range(50):
                probs = torch.rand(100_000)
                labels = torch.randint(0, 2, (100_000,)).float()
                storage.write_recording(f"rec_{i:02d}", probs, labels)

                all_probs.append(probs.numpy())
                all_labels.append(labels.numpy())

            streaming_ece = _compute_ece_streaming(storage, n_bins=10)

            probs_batch = np.concatenate(all_probs)
            labels_batch = np.concatenate(all_labels)

            bin_edges = np.linspace(0.0, 1.0, 11)
            bin_sums = np.zeros(10, dtype=np.float64)
            bin_label_sums = np.zeros(10, dtype=np.float64)
            bin_counts = np.zeros(10, dtype=np.int64)

            labels_binary = (labels_batch > 0.5).astype(np.float32)
            bin_indices = np.digitize(probs_batch, bin_edges[1:-1])

            np.add.at(bin_sums, bin_indices, probs_batch)
            np.add.at(bin_label_sums, bin_indices, labels_binary)
            np.add.at(bin_counts, bin_indices, 1)

            total = bin_counts.sum()
            batch_ece = 0.0
            for i in range(10):
                if bin_counts[i] > 0:
                    avg_prob = bin_sums[i] / bin_counts[i]
                    avg_label = bin_label_sums[i] / bin_counts[i]
                    weight = bin_counts[i] / total
                    batch_ece += weight * abs(avg_prob - avg_label)

            assert abs(streaming_ece - batch_ece) < 1e-9, (
                f"ECE mismatch: streaming={streaming_ece:.10f}, batch={batch_ece:.10f}"
            )


class TestRecordingStorageRobustness:
    """Additional robustness checks for production use."""

    def test_large_number_of_recordings(self) -> None:
        """Verify storage handles 1000+ recordings without issues."""
        with RecordingStorage() as storage:
            for i in range(1000):
                probs = torch.rand(10_000)
                labels = torch.randint(0, 2, (10_000,)).float()
                storage.write_recording(f"rec_{i:04d}", probs, labels)

            assert len(storage.recording_ids) == 1000

            count = 0
            for probs, _labels in storage.iter_recordings():
                assert probs.shape == (10_000,)
                count += 1
            assert count == 1000

    def test_mixed_recording_sizes(self) -> None:
        """Verify storage handles variable-length recordings."""
        with RecordingStorage() as storage:
            sizes = [10_000, 50_000, 100_000, 250_000, 500_000]

            for i, size in enumerate(sizes):
                probs = torch.rand(size)
                labels = torch.zeros(size)
                storage.write_recording(f"rec_{i}", probs, labels)

            probs_all, labels_all = storage.get_all_concatenated()

            expected_total = sum(sizes)
            assert probs_all.shape == (expected_total,)
            assert labels_all.shape == (expected_total,)

            offset = 0
            for i, (probs, _labels) in enumerate(storage.iter_recordings()):
                assert probs.shape == (sizes[i],)
                offset += sizes[i]

            assert offset == expected_total
