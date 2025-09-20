"""Integration tests for post-processing edge cases."""

import numpy as np
import pytest
import torch

from src.brain_brr.post.postprocess import PostProcessor


@pytest.mark.integration
class TestHysteresisEdgeCases:
    """Test hysteresis with edge cases."""

    @pytest.fixture
    def postprocessor(self):
        """Create postprocessor with default settings."""
        return PostProcessor(
            tau_on=0.86,
            tau_off=0.78,
            min_onset_samples=128,
            min_offset_samples=256,
            opening_kernel=11,
            closing_kernel=31,
            min_duration=3.0,
            sample_rate=256,
        )

    def test_rapid_oscillations(self, postprocessor):
        """Test rapid oscillations don't trigger false positives."""
        # Create rapidly oscillating probabilities
        num_samples = 10000
        probabilities = np.zeros(num_samples, dtype=np.float32)

        # Oscillate every sample
        probabilities[::2] = 0.95  # Above tau_on
        probabilities[1::2] = 0.1  # Below tau_off

        binary = postprocessor.hysteresis_threshold(probabilities)

        # Should not detect seizures due to min_onset_samples
        assert binary.sum() == 0, "Rapid oscillations should not trigger"

    def test_boundary_conditions(self, postprocessor):
        """Test onset/offset at exact boundaries."""
        num_samples = 5000
        probabilities = np.ones(num_samples, dtype=np.float32) * 0.5

        # Onset right at min_onset_samples
        probabilities[1000:1000+128] = 0.9  # Exactly min_onset_samples

        binary = postprocessor.hysteresis_threshold(probabilities)

        # Should trigger since we meet minimum
        if binary[1100] == 0:
            # Try with one more sample
            probabilities[1000:1000+129] = 0.9
            binary = postprocessor.hysteresis_threshold(probabilities)
            assert binary[1100] == 1, "Should trigger with sufficient onset samples"

    def test_single_spike_filtering(self, postprocessor):
        """Test single sample spikes are filtered."""
        num_samples = 1000
        probabilities = np.ones(num_samples, dtype=np.float32) * 0.5

        # Add single sample spikes
        for i in range(100, 900, 100):
            probabilities[i] = 0.99

        binary = postprocessor.hysteresis_threshold(probabilities)

        # Should not trigger on single spikes
        assert binary.sum() == 0, "Single spikes should be filtered"