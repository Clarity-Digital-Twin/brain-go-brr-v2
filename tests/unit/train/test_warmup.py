"""Tests for warmup schedule utilities.

Testing philosophy: Test BEHAVIOR, not implementation.
No mocking config objects - use real dataclasses.
"""

from __future__ import annotations

from src.brain_brr.config.schemas import WarmupScheduleConfig
from src.brain_brr.train.warmup import get_focal_gamma


class TestFocalGammaWarmup:
    """Test focal loss gamma warmup schedule."""

    def test_no_warmup_returns_target_gamma(self):
        """When warmup disabled, should return target gamma unchanged."""
        result = get_focal_gamma(global_step=100, warmup_config=None, target_gamma=2.0)
        assert result == 2.0

    def test_warmup_disabled_returns_target_gamma(self):
        """When warmup config exists but disabled, should return target gamma."""
        config = WarmupScheduleConfig(
            enabled=False,
            warmup_steps=1000,
            focal_gamma_enabled=True,
            focal_gamma_start=0.5,
            focal_gamma_end=2.0,
        )
        result = get_focal_gamma(global_step=500, warmup_config=config, target_gamma=2.0)
        assert result == 2.0

    def test_focal_gamma_disabled_returns_target_gamma(self):
        """When focal gamma warmup disabled, should return target gamma."""
        config = WarmupScheduleConfig(
            enabled=True,
            warmup_steps=1000,
            focal_gamma_enabled=False,
            focal_gamma_start=0.5,
            focal_gamma_end=2.0,
        )
        result = get_focal_gamma(global_step=500, warmup_config=config, target_gamma=2.0)
        assert result == 2.0

    def test_warmup_at_step_zero(self):
        """At step 0, should return start gamma."""
        config = WarmupScheduleConfig(
            enabled=True,
            warmup_steps=1000,
            focal_gamma_enabled=True,
            focal_gamma_start=0.5,
            focal_gamma_end=2.0,
        )
        result = get_focal_gamma(global_step=0, warmup_config=config, target_gamma=2.0)
        assert result == 0.5

    def test_warmup_at_halfway(self):
        """At 50% warmup, should return midpoint gamma."""
        config = WarmupScheduleConfig(
            enabled=True,
            warmup_steps=1000,
            focal_gamma_enabled=True,
            focal_gamma_start=0.5,
            focal_gamma_end=2.0,
        )
        result = get_focal_gamma(global_step=500, warmup_config=config, target_gamma=2.0)
        # 0.5 + 0.5 * (2.0 - 0.5) = 0.5 + 0.75 = 1.25
        assert abs(result - 1.25) < 1e-6

    def test_warmup_at_completion(self):
        """At warmup_steps, should return end gamma."""
        config = WarmupScheduleConfig(
            enabled=True,
            warmup_steps=1000,
            focal_gamma_enabled=True,
            focal_gamma_start=0.5,
            focal_gamma_end=2.0,
        )
        result = get_focal_gamma(global_step=1000, warmup_config=config, target_gamma=2.0)
        assert result == 2.0

    def test_warmup_after_completion(self):
        """After warmup_steps, should return target gamma."""
        config = WarmupScheduleConfig(
            enabled=True,
            warmup_steps=1000,
            focal_gamma_enabled=True,
            focal_gamma_start=0.5,
            focal_gamma_end=2.0,
        )
        result = get_focal_gamma(global_step=2000, warmup_config=config, target_gamma=2.0)
        assert result == 2.0

    def test_linear_interpolation(self):
        """Gamma should increase linearly during warmup."""
        config = WarmupScheduleConfig(
            enabled=True,
            warmup_steps=1000,
            focal_gamma_enabled=True,
            focal_gamma_start=1.0,
            focal_gamma_end=3.0,
        )

        # Test multiple points along warmup
        gamma_at_0 = get_focal_gamma(0, config, target_gamma=3.0)
        gamma_at_250 = get_focal_gamma(250, config, target_gamma=3.0)
        gamma_at_500 = get_focal_gamma(500, config, target_gamma=3.0)
        gamma_at_750 = get_focal_gamma(750, config, target_gamma=3.0)
        gamma_at_1000 = get_focal_gamma(1000, config, target_gamma=3.0)

        # Check linearity
        assert gamma_at_0 == 1.0
        assert abs(gamma_at_250 - 1.5) < 1e-6  # 1.0 + 0.25 * 2.0
        assert abs(gamma_at_500 - 2.0) < 1e-6  # 1.0 + 0.5 * 2.0
        assert abs(gamma_at_750 - 2.5) < 1e-6  # 1.0 + 0.75 * 2.0
        assert gamma_at_1000 == 3.0

    def test_warmup_with_decreasing_gamma(self):
        """Warmup can decrease gamma (start > end)."""
        config = WarmupScheduleConfig(
            enabled=True,
            warmup_steps=100,
            focal_gamma_enabled=True,
            focal_gamma_start=3.0,
            focal_gamma_end=1.0,
        )

        gamma_at_0 = get_focal_gamma(0, config, target_gamma=1.0)
        gamma_at_50 = get_focal_gamma(50, config, target_gamma=1.0)
        gamma_at_100 = get_focal_gamma(100, config, target_gamma=1.0)

        assert gamma_at_0 == 3.0
        assert abs(gamma_at_50 - 2.0) < 1e-6  # 3.0 + 0.5 * (1.0 - 3.0) = 3.0 - 1.0 = 2.0
        assert gamma_at_100 == 1.0


class TestWarmupEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_step_number(self):
        """Negative step should not crash (though invalid in practice)."""
        config = WarmupScheduleConfig(
            enabled=True,
            warmup_steps=1000,
            focal_gamma_enabled=True,
            focal_gamma_start=0.5,
            focal_gamma_end=2.0,
        )
        # Should handle gracefully (negative progress ratio)
        result = get_focal_gamma(global_step=-100, warmup_config=config, target_gamma=2.0)
        # Extrapolates below start_gamma
        assert isinstance(result, float)

    def test_same_start_and_end_gamma(self):
        """When start == end, gamma should be constant."""
        config = WarmupScheduleConfig(
            enabled=True,
            warmup_steps=1000,
            focal_gamma_enabled=True,
            focal_gamma_start=2.0,
            focal_gamma_end=2.0,
        )

        gamma_at_0 = get_focal_gamma(0, config, target_gamma=2.0)
        gamma_at_500 = get_focal_gamma(500, config, target_gamma=2.0)
        gamma_at_1000 = get_focal_gamma(1000, config, target_gamma=2.0)

        assert gamma_at_0 == 2.0
        assert gamma_at_500 == 2.0
        assert gamma_at_1000 == 2.0
