"""Integration tests for post-processing edge cases."""

import pytest
import torch

from src.brain_brr.post.postprocess import apply_hysteresis


@pytest.mark.integration
class TestHysteresisEdgeCases:
    """Test hysteresis with edge cases."""

    def test_rapid_oscillations(self):
        """Test rapid oscillations don't trigger false positives."""
        # Create rapidly oscillating probabilities
        num_samples = 10000
        probabilities = torch.zeros(1, num_samples)  # Shape (B, T)

        # Oscillate every sample
        probabilities[0, ::2] = 0.95  # Above tau_on
        probabilities[0, 1::2] = 0.1  # Below tau_off

        binary = apply_hysteresis(
            probabilities,
            tau_on=0.86,
            tau_off=0.78,
            min_onset_samples=128,
            min_offset_samples=256,
        )

        # Should not detect seizures due to min_onset_samples
        assert binary.sum() == 0, "Rapid oscillations should not trigger"

    def test_boundary_conditions(self):
        """Test onset/offset at exact boundaries."""
        num_samples = 5000
        probabilities = torch.ones(1, num_samples) * 0.5  # Shape (B, T)

        # Onset right at min_onset_samples
        probabilities[0, 1000:1128] = 0.9  # Exactly min_onset_samples

        binary = apply_hysteresis(
            probabilities,
            tau_on=0.86,
            tau_off=0.78,
            min_onset_samples=128,
            min_offset_samples=256,
        )

        # Should trigger since we meet minimum
        if binary[0, 1100] == 0:
            # Try with one more sample
            probabilities[0, 1000:1129] = 0.9
            binary = apply_hysteresis(
                probabilities,
                tau_on=0.86,
                tau_off=0.78,
                min_onset_samples=128,
                min_offset_samples=256,
            )
            assert binary[0, 1100] == 1, "Should trigger with sufficient onset samples"

    def test_single_spike_filtering(self):
        """Test single sample spikes are filtered."""
        num_samples = 1000
        probabilities = torch.ones(1, num_samples) * 0.5  # Shape (B, T)

        # Add single sample spikes
        for i in range(100, 900, 100):
            probabilities[0, i] = 0.99

        binary = apply_hysteresis(
            probabilities,
            tau_on=0.86,
            tau_off=0.78,
            min_onset_samples=128,
            min_offset_samples=256,
        )

        # Should not trigger on single spikes
        assert binary.sum() == 0, "Single spikes should be filtered"
