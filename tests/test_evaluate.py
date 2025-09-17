"""Tests for evaluation metrics (TAES, FA/24h, sensitivity@FA)."""

import pytest
import torch

from src.experiment.evaluate import (
    calculate_taes,
    fa_per_24h,
    batch_masks_to_events,
    sensitivity_at_fa_rates,
    evaluate_predictions,
)
from src.experiment.schemas import PostprocessingConfig


class TestTAES:
    """Test Time-Aligned Event Scoring metric."""

    def test_perfect_overlap(self) -> None:
        """TAES should be 1.0 for perfect overlap."""
        ref_events = [(10.0, 20.0), (30.0, 40.0)]
        pred_events = [(10.0, 20.0), (30.0, 40.0)]
        taes = calculate_taes(pred_events, ref_events)
        assert abs(taes - 1.0) < 1e-6

    def test_no_overlap(self) -> None:
        """TAES should be low for no overlap."""
        ref_events = [(10.0, 20.0)]
        pred_events = [(30.0, 40.0)]
        taes = calculate_taes(pred_events, ref_events)
        assert taes < 0.5  # Penalty for false alarm

    def test_partial_overlap(self) -> None:
        """TAES should reflect partial overlap."""
        ref_events = [(10.0, 20.0)]  # 10s duration
        pred_events = [(15.0, 25.0)]  # 5s overlap
        taes = calculate_taes(pred_events, ref_events)
        assert 0.3 < taes < 0.7  # ~0.5 overlap minus FP penalty

    def test_empty_reference(self) -> None:
        """TAES should be 0 with no reference events."""
        ref_events = []
        pred_events = [(10.0, 20.0)]
        taes = calculate_taes(pred_events, ref_events)
        assert taes == 0.0

    def test_empty_predictions(self) -> None:
        """TAES should be 0 with no predictions."""
        ref_events = [(10.0, 20.0)]
        pred_events = []
        taes = calculate_taes(pred_events, ref_events)
        assert taes == 0.0

    def test_false_alarm_penalty(self) -> None:
        """TAES should penalize false alarms."""
        ref_events = [(10.0, 20.0)]
        # One perfect match + one false alarm
        pred_events = [(10.0, 20.0), (30.0, 40.0)]
        taes = calculate_taes(pred_events, ref_events)
        assert taes < 1.0  # Penalty for FA


class TestFARate:
    """Test false alarm per 24h calculation."""

    def test_no_false_alarms(self) -> None:
        """FA/24h should be 0 with perfect predictions."""
        pred_events = [[(10.0, 20.0), (30.0, 40.0)]]
        ref_events = [[(10.0, 20.0), (30.0, 40.0)]]
        fa_rate = fa_per_24h(pred_events, ref_events, total_hours=1.0)
        assert fa_rate == 0.0

    def test_all_false_alarms(self) -> None:
        """FA/24h should count all events when no refs."""
        pred_events = [[(i * 300.0, i * 300.0 + 5.0) for i in range(10)]]
        ref_events = [[]]
        fa_rate = fa_per_24h(pred_events, ref_events, total_hours=1.0)
        assert abs(fa_rate - 240.0) < 1e-3  # 10 FA/hr * 24 hr

    def test_mixed_events(self) -> None:
        """FA/24h should only count non-overlapping predictions."""
        pred_events = [[(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]]
        ref_events = [[(10.0, 20.0)]]  # Only first pred matches
        fa_rate = fa_per_24h(pred_events, ref_events, total_hours=1.0)
        assert abs(fa_rate - 48.0) < 1e-3  # 2 FA/hr * 24 hr

    def test_multi_record(self) -> None:
        """FA/24h should aggregate across records."""
        pred_events = [
            [(10.0, 20.0), (30.0, 40.0)],
            [(50.0, 60.0), (70.0, 80.0)],
        ]
        ref_events = [
            [(10.0, 20.0)],  # 1 FA in first record
            [],  # 2 FA in second record
        ]
        fa_rate = fa_per_24h(pred_events, ref_events, total_hours=2.0)
        assert abs(fa_rate - 36.0) < 1e-3  # 3 FA/2hr * 24hr


class TestEventization:
    """Test mask to event conversion."""

    def test_simple_mask_to_events(self) -> None:
        """Test basic binary mask to events."""
        # Create mask with two seizures
        mask = torch.zeros(1, 2560)  # 10s at 256Hz
        mask[0, 256:512] = 1  # 1-2s
        mask[0, 1024:1536] = 1  # 4-6s

        events = batch_masks_to_events(mask, fs=256)
        assert len(events) == 1
        assert len(events[0]) == 2
        assert abs(events[0][0][0] - 1.0) < 0.1
        assert abs(events[0][0][1] - 2.0) < 0.1
        assert abs(events[0][1][0] - 4.0) < 0.1
        assert abs(events[0][1][1] - 6.0) < 0.1

    def test_batch_masks_to_events(self) -> None:
        """Test batch conversion."""
        masks = torch.zeros(2, 2560)
        masks[0, 0:256] = 1  # First record: 0-1s
        masks[1, 256:512] = 1  # Second record: 1-2s

        events = batch_masks_to_events(masks, fs=256)
        assert len(events) == 2
        assert len(events[0]) == 1
        assert len(events[1]) == 1


class TestSensitivityAtFA:
    """Test sensitivity at FA operating points."""

    @pytest.fixture
    def post_cfg(self) -> PostprocessingConfig:
        """Create post-processing config."""
        from src.experiment.schemas import HysteresisConfig

        return PostprocessingConfig(
            hysteresis=HysteresisConfig(tau_on=0.86, tau_off=0.78),
            morphology={"kernel_size": 5, "operation": "closing"},
            min_duration=1.0,
        )

    def test_sensitivity_monotonic(self, post_cfg: PostprocessingConfig) -> None:
        """Sensitivity should decrease with stricter FA targets."""
        # Create synthetic data with varying probabilities
        probs = torch.linspace(0, 1, 15360).unsqueeze(0)
        labels = (probs > 0.6).float()

        metrics = sensitivity_at_fa_rates(
            probs, labels, [10, 5, 1], post_cfg, sampling_rate=256
        )

        # Monotonicity check
        assert metrics["sensitivity_at_10fa"] >= metrics["sensitivity_at_5fa"]
        assert metrics["sensitivity_at_5fa"] >= metrics["sensitivity_at_1fa"]

    def test_perfect_predictions(self, post_cfg: PostprocessingConfig) -> None:
        """Perfect predictions should have high sensitivity."""
        labels = torch.zeros(1, 15360)
        labels[0, 5000:6000] = 1  # Single seizure

        # Perfect prediction
        probs = labels.clone()

        metrics = sensitivity_at_fa_rates(
            probs, labels, [10], post_cfg, sampling_rate=256
        )

        assert metrics["sensitivity_at_10fa"] > 0.9


class TestEvaluatePredictions:
    """Test complete evaluation function."""

    @pytest.fixture
    def post_cfg(self) -> PostprocessingConfig:
        """Create post-processing config."""
        from src.experiment.schemas import HysteresisConfig

        return PostprocessingConfig(
            hysteresis=HysteresisConfig(tau_on=0.86, tau_off=0.78),
            morphology={"kernel_size": 5, "operation": "closing"},
            min_duration=1.0,
        )

    def test_evaluate_output_keys(self, post_cfg: PostprocessingConfig) -> None:
        """Test that evaluate returns expected keys."""
        probs = torch.rand(2, 15360)
        labels = (probs > 0.7).float()

        results = evaluate_predictions(
            probs, labels, [10, 5, 1], post_cfg
        )

        expected_keys = {
            "taes",
            "auroc",
            "sensitivity_at_10fa",
            "sensitivity_at_5fa",
            "sensitivity_at_1fa",
            "fa_curve",
        }
        assert set(results.keys()) >= expected_keys

    def test_evaluate_value_ranges(self, post_cfg: PostprocessingConfig) -> None:
        """Test that metrics are in valid ranges."""
        probs = torch.rand(2, 15360)
        labels = (probs > 0.7).float()

        results = evaluate_predictions(
            probs, labels, [10, 5, 1], post_cfg
        )

        # All metrics should be in [0, 1]
        assert 0 <= results["taes"] <= 1
        assert 0 <= results["auroc"] <= 1
        assert 0 <= results["sensitivity_at_10fa"] <= 1
        assert 0 <= results["sensitivity_at_5fa"] <= 1
        assert 0 <= results["sensitivity_at_1fa"] <= 1

        # FA curve should be list of tuples
        assert isinstance(results["fa_curve"], list)
        if results["fa_curve"]:
            assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in results["fa_curve"])