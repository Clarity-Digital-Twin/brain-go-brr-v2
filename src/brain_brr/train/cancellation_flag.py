"""Cancellation flag for graceful shutdown during long-running operations.

Allows signal handlers to request cancellation without immediately terminating,
giving validation loops and other operations a chance to exit cleanly.
"""

from __future__ import annotations


class CancellationFlag:
    """Thread-safe flag for requesting graceful cancellation.

    Used to coordinate between signal handlers and long-running operations like validation.
    Signal handler sets the flag, operations check it periodically and exit gracefully.

    Example:
        cancel_flag = CancellationFlag()

        def signal_handler(sig, frame):
            cancel_flag.set()

        signal.signal(signal.SIGTERM, signal_handler)

        for batch in dataloader:
            if cancel_flag.is_set():
                logger.warning("Cancellation requested, exiting")
                break
            # Process batch...
    """

    def __init__(self) -> None:
        """Initialize cancellation flag as not set."""
        self._cancelled = False

    def set(self) -> None:
        """Set the cancellation flag to request graceful exit."""
        self._cancelled = True

    def is_set(self) -> bool:
        """Check if cancellation has been requested.

        Returns:
            True if cancellation requested, False otherwise
        """
        return self._cancelled

    def clear(self) -> None:
        """Clear the cancellation flag (for testing/reuse)."""
        self._cancelled = False
