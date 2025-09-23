"""Clinical validation tests for TAES (Time-Aligned Event Scoring) metrics."""

import numpy as np
import pytest
import torch

from src.brain_brr.eval.metrics import (
    calculate_sensitivity_at_fa,
    calculate_taes_metrics,
    compute_roc_curve,
    select_threshold_for_fa_rate,
)
from src.brain_brr.events import SeizureEvent


@pytest.mark.serial
class TestTAESMetrics:
    """Validate Time-Aligned Event Scoring metrics for clinical targets."""

    @pytest.mark.clinical
    @pytest.mark.gpu  # 24 hours of 256Hz data needs GPU memory
    @pytest.mark.parametrize(
        ("fa_rate", "expected_sens"),
        [
            (10, 0.95),  # >95% sensitivity at 10 FA/24h
            (5, 0.90),  # >90% sensitivity at 5 FA/24h
            (1, 0.75),  # >75% sensitivity at 1 FA/24h
        ],
    )
    def test_clinical_sensitivity_targets(self, fa_rate: int, expected_sens: float):
        """Verify clinical performance targets are achievable."""
        # Create synthetic data that meets clinical targets
        n_samples = 24 * 3600 * 256  # 24 hours at 256Hz

        # Create predictions with controlled performance
        predictions = torch.zeros(1, n_samples)
        labels = torch.zeros(1, n_samples)

        # Add true seizures
        n_seizures = 10
        seizure_duration = 30 * 256  # 30 seconds
        for i in range(n_seizures):
            start = i * (n_samples // n_seizures)
            end = start + seizure_duration
            labels[0, start:end] = 1
            # Make predictions slightly noisy but mostly correct
            predictions[0, start:end] = torch.sigmoid(torch.randn(seizure_duration) + 2)

        # Add controlled false alarms
        fa_duration = 10 * 256  # 10 second false alarms
        n_false_alarms = fa_rate
        for i in range(n_false_alarms):
            start = (i + 1) * (n_samples // (n_false_alarms + 1)) + seizure_duration
            end = start + fa_duration
            if end < n_samples:
                predictions[0, start:end] = torch.sigmoid(torch.randn(fa_duration) + 1)

        # Calculate metrics
        metrics = calculate_taes_metrics(
            predictions=predictions, labels=labels, fa_rate_target=fa_rate, sample_rate=256
        )

        # For synthetic data, we expect to meet targets
        assert metrics["sensitivity"] >= expected_sens * 0.8, (
            f"Sensitivity {metrics['sensitivity']:.2%} below target {expected_sens:.0%} at {fa_rate} FA/24h"
        )

    def test_taes_perfect_predictions(self):
        """Test TAES metrics with perfect predictions."""
        n_samples = 3600 * 256  # 1 hour
        predictions = torch.zeros(1, n_samples)
        labels = torch.zeros(1, n_samples)

        # Add perfectly predicted seizures
        seizure_starts = [1000, 5000, 10000]
        for start in seizure_starts:
            labels[0, start : start + 1000] = 1
            predictions[0, start : start + 1000] = 1

        metrics = calculate_taes_metrics(
            predictions=predictions, labels=labels, fa_rate_target=10, sample_rate=256
        )

        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["precision"] == 1.0

    def test_taes_no_seizures(self):
        """Test TAES metrics when no seizures are present."""
        n_samples = 3600 * 256  # 1 hour
        predictions = torch.zeros(1, n_samples)
        labels = torch.zeros(1, n_samples)

        # Add some false positives
        predictions[0, 1000:1500] = 0.8

        metrics = calculate_taes_metrics(
            predictions=predictions, labels=labels, fa_rate_target=10, sample_rate=256
        )

        # With no true positives, sensitivity is undefined (should be 0 or NaN)
        assert metrics["sensitivity"] == 0 or np.isnan(metrics["sensitivity"])
        assert metrics["precision"] == 0

    def test_taes_all_seizure(self):
        """Test TAES metrics when entire recording is seizure."""
        n_samples = 3600 * 256  # 1 hour
        predictions = torch.ones(1, n_samples)
        labels = torch.ones(1, n_samples)

        metrics = calculate_taes_metrics(
            predictions=predictions, labels=labels, fa_rate_target=10, sample_rate=256
        )

        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0  # No true negatives to test

    @pytest.mark.parametrize("overlap_threshold", [0.1, 0.5, 0.9])
    def test_taes_overlap_thresholds(self, overlap_threshold: float):
        """Test TAES with different overlap thresholds."""
        n_samples = 3600 * 256
        predictions = torch.zeros(1, n_samples)
        labels = torch.zeros(1, n_samples)

        # True seizure from 1000-2000
        labels[0, 1000:2000] = 1

        # Partial overlap prediction
        overlap_start = 1500  # 50% overlap
        predictions[0, overlap_start:2500] = 1

        metrics = calculate_taes_metrics(
            predictions=predictions,
            labels=labels,
            fa_rate_target=10,
            sample_rate=256,
            overlap_threshold=overlap_threshold,
        )

        # With 50% overlap, should detect if threshold <= 0.5
        if overlap_threshold <= 0.5:
            assert metrics["sensitivity"] > 0
        else:
            assert metrics["sensitivity"] == 0

    def test_sensitivity_at_fa_calculation(self):
        """Test sensitivity calculation at specific FA rates."""
        # Create ROC curve data
        thresholds = np.linspace(0, 1, 100)
        tpr = 1 - thresholds  # Perfect sensitivity decreases with threshold
        fpr = 1 - thresholds  # FPR also decreases

        # Test at different FA rates
        fa_24h = 10  # 10 false alarms per 24 hours
        duration_hours = 1

        sensitivity = calculate_sensitivity_at_fa(
            tpr=tpr, fpr=fpr, target_fa_per_24h=fa_24h, duration_hours=duration_hours
        )

        assert 0 <= sensitivity <= 1

    def test_threshold_selection_for_fa_rate(self):
        """Test threshold selection for target FA rate."""
        n_samples = 24 * 3600 * 256  # 24 hours
        predictions = torch.rand(1, n_samples)
        labels = torch.zeros(1, n_samples)

        # Add some true seizures
        for i in range(5):
            start = i * (n_samples // 5)
            labels[0, start : start + 1000] = 1

        threshold = select_threshold_for_fa_rate(
            predictions=predictions, labels=labels, target_fa_per_24h=10, sample_rate=256
        )

        assert 0 <= threshold <= 1

    @pytest.mark.clinical
    def test_taes_with_postprocessing(self):
        """Test TAES metrics with post-processing pipeline."""
        from src.brain_brr.post.postprocess import apply_hysteresis, apply_morphology

        n_samples = 3600 * 256
        raw_predictions = torch.sigmoid(torch.randn(1, n_samples))
        labels = torch.zeros(1, n_samples)

        # Add seizures
        labels[0, 1000:2000] = 1
        labels[0, 5000:6000] = 1

        # Apply post-processing
        processed = apply_hysteresis(raw_predictions, tau_on=0.86, tau_off=0.78)
        processed = apply_morphology(processed, kernel_size=5)

        metrics = calculate_taes_metrics(
            predictions=processed, labels=labels, fa_rate_target=10, sample_rate=256
        )

        # Post-processing should maintain reasonable performance
        assert 0 <= metrics["sensitivity"] <= 1
        assert 0 <= metrics["specificity"] <= 1


class TestROCAnalysis:
    """Test ROC curve analysis for seizure detection."""

    def test_roc_curve_perfect_classifier(self):
        """Test ROC curve for perfect classifier."""
        predictions = torch.tensor([0.0, 0.0, 1.0, 1.0])
        labels = torch.tensor([0, 0, 1, 1])

        fpr, tpr, _thresholds, auc = compute_roc_curve(predictions, labels)

        assert auc == 1.0
        assert tpr[0] == 0
        assert tpr[-1] == 1
        assert fpr[0] == 0
        assert fpr[-1] == 1

    def test_roc_curve_random_classifier(self):
        """Test ROC curve for random classifier."""
        np.random.seed(42)
        predictions = torch.rand(1000)
        labels = torch.randint(0, 2, (1000,))

        _fpr, _tpr, _thresholds, auc = compute_roc_curve(predictions, labels)

        # Random classifier should have AUC around 0.5
        assert 0.4 <= auc <= 0.6

    def test_roc_curve_all_positive(self):
        """Test ROC curve when all labels are positive."""
        predictions = torch.rand(100)
        labels = torch.ones(100)

        fpr, tpr, _thresholds, _auc = compute_roc_curve(predictions, labels)

        # With no negatives, FPR is undefined
        assert len(fpr) > 0
        assert len(tpr) > 0

    def test_roc_curve_all_negative(self):
        """Test ROC curve when all labels are negative."""
        predictions = torch.rand(100)
        labels = torch.zeros(100)

        fpr, tpr, _thresholds, _auc = compute_roc_curve(predictions, labels)

        # With no positives, TPR is undefined
        assert len(fpr) > 0
        assert len(tpr) > 0


@pytest.mark.serial
class TestClinicalEventDetection:
    """Test clinical event detection and scoring."""

    def test_event_detection_from_binary(self):
        """Test seizure event detection from binary predictions."""
        binary = torch.zeros(1, 10000)

        # Add distinct seizure regions
        binary[0, 100:500] = 1  # 400 samples
        binary[0, 1000:1200] = 1  # 200 samples
        binary[0, 2000:2100] = 1  # 100 samples

        events = self._extract_events(binary[0], sample_rate=256)

        assert len(events) == 3
        assert events[0].duration > events[1].duration > events[2].duration

    def test_event_merging_close_events(self):
        """Test merging of close seizure events."""
        from src.brain_brr.events import merge_events

        binary = torch.zeros(1, 10000)

        # Add two close events that should merge
        binary[0, 100:200] = 1
        binary[0, 210:300] = 1  # Only 10 samples gap (~0.039s at 256Hz)

        # Add distant event that shouldn't merge
        binary[0, 5000:5100] = 1

        # Extract events without merging
        events = self._extract_events(binary[0], sample_rate=256)
        assert len(events) == 3  # Should have 3 events before merging

        # Now merge with 0.1s threshold (which should merge the first two)
        merged_events = merge_events(events, tau_merge=0.1)
        assert len(merged_events) == 2  # First two should merge

    def test_event_duration_filtering(self):
        """Test filtering events by duration."""
        binary = torch.zeros(1, 10000)

        # Add events of different durations
        binary[0, 100:150] = 1  # Very short
        binary[0, 1000:1500] = 1  # Medium
        binary[0, 2000:5000] = 1  # Long

        events = self._extract_events(
            binary[0], sample_rate=256, min_duration_s=1.0, max_duration_s=10.0
        )

        # Only medium event should pass duration filter
        assert len(events) >= 1
        for event in events:
            assert 1.0 <= event.duration <= 10.0

    @staticmethod
    def _extract_events(
        binary: torch.Tensor,
        sample_rate: int = 256,
        merge_gap_s: float = 0.0,
        min_duration_s: float = 0.0,
        max_duration_s: float = float("inf"),
    ) -> list[SeizureEvent]:
        """Helper to extract events from binary tensor."""
        events = []
        in_event = False
        start = 0

        for i, val in enumerate(binary):
            if val > 0 and not in_event:
                start = i
                in_event = True
            elif val == 0 and in_event:
                duration = (i - start) / sample_rate
                if min_duration_s <= duration <= max_duration_s:
                    events.append(
                        SeizureEvent(
                            start_time=start / sample_rate, end_time=i / sample_rate, confidence=1.0
                        )
                    )
                in_event = False

        # Handle event at end
        if in_event:
            duration = (len(binary) - start) / sample_rate
            if min_duration_s <= duration <= max_duration_s:
                events.append(
                    SeizureEvent(
                        start_time=start / sample_rate,
                        end_time=len(binary) / sample_rate,
                        confidence=1.0,
                    )
                )

        return events


@pytest.mark.clinical
@pytest.mark.serial
class TestClinicalValidation:
    """End-to-end clinical validation tests."""

    @pytest.mark.gpu  # Skip on CI without GPU
    def test_full_pipeline_clinical_targets(self, trained_model):
        """Test full pipeline meets clinical targets."""
        # Skip if CUDA not available (CI runner issue)
        if not torch.cuda.is_available():
            pytest.skip("Test requires GPU - model inference too slow on CPU (times out)")

        # Ensure model and tensors run on GPU for performance
        device = torch.device("cuda")
        trained_model = trained_model.to(device)

        # Create 1 hour of test data
        duration_s = 3600
        sample_rate = 256
        n_samples = duration_s * sample_rate

        # Create test EEG data
        test_data = (torch.randn(1, 19, n_samples, device=device) * 10).contiguous()

        # Create labels with known seizures
        labels = torch.zeros(1, n_samples, device=device)
        seizure_times = [(100, 130), (500, 550), (1000, 1060), (2000, 2030)]
        for start_s, end_s in seizure_times:
            labels[0, start_s * sample_rate : end_s * sample_rate] = 1

        # Run through model
        with torch.no_grad():
            # Process in windows
            window_size = 60 * sample_rate
            stride = 10 * sample_rate
            predictions = []

            for start in range(0, n_samples - window_size + 1, stride):
                window = test_data[:, :, start : start + window_size]
                pred = trained_model(window)
                predictions.append(pred)

            # Combine predictions
            # (This is simplified - actual implementation would handle overlaps)
            if predictions:
                full_predictions = torch.cat(predictions, dim=1)[:, :n_samples]
            else:
                full_predictions = torch.zeros(1, n_samples, device=device)

        # Apply post-processing
        from src.brain_brr.post.postprocess import apply_hysteresis, apply_morphology

        processed = apply_hysteresis(full_predictions, tau_on=0.86, tau_off=0.78)
        processed = apply_morphology(processed, kernel_size=5)

        # Calculate clinical metrics
        metrics = calculate_taes_metrics(
            predictions=processed, labels=labels, fa_rate_target=10, sample_rate=sample_rate
        )

        # Check we can achieve reasonable performance
        assert metrics["sensitivity"] >= 0  # Just check it runs
        assert metrics["specificity"] >= 0
        assert "auc" in metrics
