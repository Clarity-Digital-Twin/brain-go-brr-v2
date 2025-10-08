"""Wall-clock timeout guard for graceful training exit.

Prevents hard kills by monitoring elapsed time and triggering graceful exit
before reaching platform timeouts (e.g., Modal's 24-hour limit).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TimeoutGuard:
    """Monitor wall-clock time and trigger graceful exit before timeout.

    Designed for cloud platforms with hard timeout limits (e.g., Modal's 24h).
    Checks elapsed time and triggers graceful exit with safety margin to allow
    checkpoint saving and cleanup.

    Example:
        >>> guard = TimeoutGuard(limit_seconds=86400, safety_margin_seconds=600)
        >>> for epoch in range(100):
        ...     if guard.check():
        ...         print("Timeout imminent, exiting gracefully")
        ...         save_checkpoint()
        ...         break
        ...     train_epoch()
    """

    def __init__(
        self,
        limit_seconds: int | None,
        safety_margin_seconds: int = 600,
        on_timeout: Callable[[], None] | None = None,
    ):
        """Initialize timeout guard.

        Args:
            limit_seconds: Wall-clock timeout in seconds (None = no limit)
            safety_margin_seconds: Exit N seconds before timeout (default: 10 min)
            on_timeout: Optional callback when timeout imminent
        """
        self.limit = limit_seconds
        self.margin = safety_margin_seconds
        self.start_time = time.monotonic()  # Use monotonic clock (immune to system clock changes)
        self.callback = on_timeout
        self._triggered = False

    def check(self) -> bool:
        """Check if timeout is imminent.

        Returns:
            True if should exit gracefully, False otherwise
        """
        if self.limit is None:
            return False

        elapsed = time.monotonic() - self.start_time
        imminent = elapsed >= (self.limit - self.margin)

        if imminent and not self._triggered:
            self._triggered = True
            if self.callback:
                self.callback()

        return imminent

    def remaining_seconds(self) -> float | None:
        """Get remaining time before timeout.

        Returns:
            Seconds remaining, or None if no limit set
        """
        if self.limit is None:
            return None

        elapsed = time.monotonic() - self.start_time
        return max(0.0, self.limit - elapsed)

    def elapsed_seconds(self) -> float:
        """Get elapsed time since guard creation.

        Returns:
            Seconds elapsed
        """
        return time.monotonic() - self.start_time

    def reset(self) -> None:
        """Reset the timeout guard to current time.

        Useful if you want to restart the timer after a checkpoint.
        """
        self.start_time = time.monotonic()
        self._triggered = False
