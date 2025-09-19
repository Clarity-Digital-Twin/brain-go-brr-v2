"""Tests for post-processing operations."""

import pytest
import torch

from src.brain_brr.post import (
    apply_hysteresis,
    apply_morphology,
    filter_duration,
    postprocess_predictions,
    stitch_windows,
)
from src.brain_brr.config.schemas import PostprocessingConfig


class TestHysteresis:
    """Test hysteresis thresholding."""

    def test_basic_hysteresis(self):
        """Test basic ON/OFF transitions."""
        # Create simple probability sequence
        probs = torch.tensor(
            [[0.2, 0.9, 0.9, 0.2, 0.2, 0.9, 0.2]]  # Below, above, above, below, below, above, below
        )

        masks = apply_hysteresis(
            probs, tau_on=0.85, tau_off=0.3, min_onset_samples=1, min_offset_samples=1
        )

        # With tau_off=0.3, it exits at index 3 (0.2 < 0.3)
        # Then re-enters at index 5 (0.9 > 0.85)
        expected = torch.tensor([[False, True, True, False, False, True, False]])
        assert torch.equal(masks, expected)

    def test_stability_windows(self):
        """Test min_onset and min_offset stability."""
        # Oscillating signal
        probs = torch.tensor(
            [[0.9, 0.9, 0.9, 0.2, 0.2, 0.2, 0.9, 0.2]]  # 3 high, 3 low, 1 high, 1 low
        )

        # Require 2 samples for onset, 2 for offset
        masks = apply_hysteresis(
            probs, tau_on=0.85, tau_off=0.3, min_onset_samples=2, min_offset_samples=2
        )

        # Should enter at index 0-1 (retroactive after 2 samples above tau_on)
        # Continue through index 2, then at 3-4 have 2 samples below tau_off, exit
        # Single 0.9 at index 6 is not enough (needs 2 for onset)
        expected = torch.tensor([[True, True, True, True, False, False, False, False]])
        assert torch.equal(masks, expected)

    def test_invalid_thresholds(self):
        """Test that invalid thresholds raise error."""
        probs = torch.randn(1, 10).sigmoid()
        with pytest.raises(ValueError, match=r"tau_on.*must be > tau_off"):
            apply_hysteresis(probs, tau_on=0.5, tau_off=0.7)

    def test_hysteresis_equality_edge_cases(self):
        """Test hysteresis behavior at exact threshold values."""
        # Test exact equality cases: should trigger at tau_on, stay on at tau_off
        probs = torch.tensor([[0.85, 0.86, 0.86, 0.78, 0.78, 0.77]])
        masks = apply_hysteresis(probs, tau_on=0.86, tau_off=0.78)

        # Should trigger at 0.86 (>=), stay on at 0.78 (not <), off at 0.77 (<)
        expected = torch.tensor([[False, True, True, True, True, False]])
        assert torch.equal(masks, expected)


class TestMorphology:
    """Test morphological operations."""

    def test_opening_removes_spikes(self):
        """Test that opening removes isolated spikes."""
        # Single spike in middle
        masks = torch.tensor([[0, 0, 0, 1, 0, 0, 0]], dtype=torch.bool)

        # Opening with kernel size 3 should remove single spike
        cleaned = apply_morphology(masks, opening_kernel=3, closing_kernel=1)
        expected = torch.zeros_like(masks)
        assert torch.equal(cleaned, expected)

    def test_closing_fills_gaps(self):
        """Test that closing fills small gaps."""
        # Gap in middle with padding
        masks = torch.tensor([[0, 1, 1, 0, 1, 1, 0]], dtype=torch.bool)

        # Closing with kernel size 3 should fill single gap
        cleaned = apply_morphology(masks, opening_kernel=1, closing_kernel=3)

        # After closing, the gap should be filled but edges may erode
        # Check that the gap at position 3 is filled
        assert cleaned[0, 3]  # Gap is filled
        # The continuous region should be preserved
        assert cleaned[0, 1:6].all()  # Main region preserved

    def test_odd_kernel_validation(self):
        """Test that even kernels raise error."""
        masks = torch.randn(1, 10) > 0
        with pytest.raises(ValueError, match="Kernel sizes must be odd"):
            apply_morphology(masks, opening_kernel=4, closing_kernel=3)


class TestDurationFilter:
    """Test duration filtering."""

    def test_filter_short_events(self):
        """Test removal of short events."""
        # Two events: 2 samples (short) and 5 samples (keep)
        masks = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)

        # Filter out events < 3 samples
        filtered = filter_duration(masks, min_duration_samples=3, max_duration_samples=100)
        expected = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1]], dtype=torch.bool)
        assert torch.equal(filtered, expected)

    def test_segment_long_events(self):
        """Test segmentation of long events."""
        # One long event (10 samples)
        masks = torch.ones(1, 10, dtype=torch.bool)

        # Max duration 4 samples - should create 3 segments
        filtered = filter_duration(masks, min_duration_samples=1, max_duration_samples=4)
        # All samples should still be True (segmented but not removed)
        assert torch.equal(filtered, masks)


class TestStitching:
    """Test window stitching."""

    def test_overlap_add(self):
        """Test overlap-add stitching."""
        # Two overlapping windows
        window1 = torch.tensor([0.8, 0.9, 0.7, 0.6])
        window2 = torch.tensor([0.5, 0.4, 0.3, 0.2])

        stitched = stitch_windows(
            window_probs=[window1, window2],
            window_starts=[0, 2],
            total_length=6,
            method="overlap_add",
        )

        # Check shape and range
        assert len(stitched) == 6
        assert torch.all((stitched >= 0) & (stitched <= 1))

        # Check averaging in overlap region
        assert torch.allclose(stitched[2], (window1[2] + window2[0]) / 2)
        assert torch.allclose(stitched[3], (window1[3] + window2[1]) / 2)

    def test_max_stitching(self):
        """Test max stitching."""
        window1 = torch.tensor([0.8, 0.9, 0.7, 0.6])
        window2 = torch.tensor([0.5, 0.95, 0.3, 0.2])

        stitched = stitch_windows(
            window_probs=[window1, window2], window_starts=[0, 2], total_length=6, method="max"
        )

        # Check max is taken in overlap
        assert stitched[2] == max(window1[2], window2[0])
        assert stitched[3] == max(window1[3], window2[1])


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline(self):
        """Test complete post-processing pipeline."""
        # Create test data with clear seizure pattern
        probs = torch.zeros(2, 100)
        probs[0, 20:40] = 0.9  # Clear seizure
        probs[1, 60:65] = 0.9  # Short spike (should be filtered)

        config = PostprocessingConfig()
        config.hysteresis.tau_on = 0.85
        config.hysteresis.tau_off = 0.2
        config.hysteresis.min_onset_samples = 1  # For test with 100 samples
        config.hysteresis.min_offset_samples = 1  # For test with 100 samples
        config.morphology.opening_kernel = 3
        config.morphology.closing_kernel = 5
        config.duration.min_duration_s = 0.05  # ~13 samples at 256 Hz

        masks = postprocess_predictions(probs, config, sampling_rate=256)

        # First batch should have seizure preserved
        assert masks[0, 20:40].any()

        # Second batch spike should be removed (too short)
        assert not masks[1, 60:65].any()
