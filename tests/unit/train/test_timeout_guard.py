"""Tests for wall-clock timeout guard.

Testing philosophy: Test BEHAVIOR, not implementation.
Minimal mocks - only time.monotonic() for deterministic tests.
"""

from __future__ import annotations

import time

from src.brain_brr.train.timeout_guard import TimeoutGuard


class TestTimeoutGuardNoLimit:
    """Test behavior when no timeout limit is set."""

    def test_no_limit_always_returns_false(self):
        """Guard with no limit should never trigger timeout."""
        guard = TimeoutGuard(limit_seconds=None)

        assert guard.check() is False
        assert guard.check() is False

    def test_no_limit_remaining_returns_none(self):
        """Remaining time should be None when no limit set."""
        guard = TimeoutGuard(limit_seconds=None)

        assert guard.remaining_seconds() is None

    def test_no_limit_tracks_elapsed_time(self):
        """Should still track elapsed time even without limit."""
        guard = TimeoutGuard(limit_seconds=None)

        time.sleep(0.01)
        elapsed = guard.elapsed_seconds()

        assert elapsed > 0
        assert elapsed < 1.0


class TestTimeoutGuardBasicBehavior:
    """Test basic timeout detection logic."""

    def test_returns_false_before_timeout(self, monkeypatch):
        """Should return False when plenty of time remains."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100)

        current_time = 500.0
        assert guard.check() is False

    def test_returns_true_at_timeout(self, monkeypatch):
        """Should return True when timeout imminent (within safety margin)."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100)

        current_time = 899.0
        assert guard.check() is False

        current_time = 900.0
        assert guard.check() is True

    def test_returns_true_after_timeout(self, monkeypatch):
        """Should continue returning True after timeout triggered."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100)

        current_time = 950.0
        assert guard.check() is True
        assert guard.check() is True


class TestTimeoutGuardTimingCalculations:
    """Test remaining and elapsed time calculations."""

    def test_remaining_seconds_decreases(self, monkeypatch):
        """Remaining time should decrease as time passes."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100)

        current_time = 0.0
        assert guard.remaining_seconds() == 1000.0

        current_time = 200.0
        assert guard.remaining_seconds() == 800.0

        current_time = 900.0
        assert guard.remaining_seconds() == 100.0

    def test_remaining_seconds_clamped_at_zero(self, monkeypatch):
        """Remaining time should not go negative."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100)

        current_time = 1100.0
        assert guard.remaining_seconds() == 0.0

    def test_elapsed_seconds_increases(self, monkeypatch):
        """Elapsed time should increase as time passes."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100)

        current_time = 0.0
        assert guard.elapsed_seconds() == 0.0

        current_time = 300.0
        assert guard.elapsed_seconds() == 300.0

        current_time = 1000.0
        assert guard.elapsed_seconds() == 1000.0


class TestTimeoutGuardCallback:
    """Test callback invocation on timeout."""

    def test_callback_invoked_on_first_timeout(self, monkeypatch):
        """Callback should be called exactly once on first timeout."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        callback_count = 0

        def on_timeout():
            nonlocal callback_count
            callback_count += 1

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100, on_timeout=on_timeout)

        current_time = 900.0
        guard.check()
        assert callback_count == 1

        current_time = 950.0
        guard.check()
        assert callback_count == 1

    def test_no_callback_when_not_provided(self, monkeypatch):
        """Should not crash when callback is None."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100, on_timeout=None)

        current_time = 900.0
        assert guard.check() is True


class TestTimeoutGuardReset:
    """Test reset functionality."""

    def test_reset_restarts_timer(self, monkeypatch):
        """Reset should restart the timer from current time."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100)

        current_time = 800.0
        assert guard.remaining_seconds() == 200.0

        current_time = 800.0
        guard.reset()

        assert guard.remaining_seconds() == 1000.0

        current_time = 900.0
        assert guard.remaining_seconds() == 900.0

    def test_reset_clears_triggered_flag(self, monkeypatch):
        """Reset should allow callback to be invoked again."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        callback_count = 0

        def on_timeout():
            nonlocal callback_count
            callback_count += 1

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=100, on_timeout=on_timeout)

        current_time = 900.0
        guard.check()
        assert callback_count == 1

        current_time = 900.0
        guard.reset()

        current_time = 1800.0
        guard.check()
        assert callback_count == 2


class TestTimeoutGuardSafetyMargin:
    """Test safety margin behavior."""

    def test_different_safety_margins(self, monkeypatch):
        """Different safety margins should trigger at different times."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard_10min = TimeoutGuard(limit_seconds=3600, safety_margin_seconds=600)
        guard_1min = TimeoutGuard(limit_seconds=3600, safety_margin_seconds=60)

        current_time = 2999.0
        assert guard_10min.check() is False
        assert guard_1min.check() is False

        current_time = 3000.0
        assert guard_10min.check() is True
        assert guard_1min.check() is False

        current_time = 3540.0
        assert guard_10min.check() is True
        assert guard_1min.check() is True

    def test_zero_safety_margin(self, monkeypatch):
        """Zero safety margin should trigger exactly at limit."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1000, safety_margin_seconds=0)

        current_time = 999.0
        assert guard.check() is False

        current_time = 1000.0
        assert guard.check() is True


class TestTimeoutGuardEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_timeout(self, monkeypatch):
        """Should handle very small timeout values."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=1, safety_margin_seconds=0)

        current_time = 0.5
        assert guard.check() is False

        current_time = 1.0
        assert guard.check() is True

    def test_safety_margin_larger_than_limit(self, monkeypatch):
        """Safety margin larger than limit should trigger immediately."""
        current_time = 0.0

        def mock_monotonic():
            return current_time

        monkeypatch.setattr(time, "monotonic", mock_monotonic)

        guard = TimeoutGuard(limit_seconds=100, safety_margin_seconds=200)

        current_time = 0.0
        assert guard.check() is True


class TestTimeoutGuardRealWorld:
    """Test real-world scenarios without mocking."""

    def test_actual_timeout_detection(self):
        """Test with real time (short timeout for speed)."""
        guard = TimeoutGuard(limit_seconds=1, safety_margin_seconds=0)

        assert guard.check() is False

        time.sleep(0.1)
        assert guard.check() is False

        time.sleep(1.0)
        assert guard.check() is True

    def test_elapsed_and_remaining_consistency(self):
        """Elapsed + remaining should equal limit (before timeout)."""
        guard = TimeoutGuard(limit_seconds=10, safety_margin_seconds=1)

        time.sleep(0.01)

        elapsed = guard.elapsed_seconds()
        remaining = guard.remaining_seconds()

        assert remaining is not None
        assert abs((elapsed + remaining) - 10.0) < 0.1
