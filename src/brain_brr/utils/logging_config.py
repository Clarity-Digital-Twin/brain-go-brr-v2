"""Central logging configuration for Brain-Go-Brr V3."""

import atexit
import contextlib
import importlib.util
import logging
import os
import sys
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console, Optional, cast

from src.brain_brr.constants import LOG_EVERY_N_STEPS

# Constants from environment with sensible defaults
LOG_LEVEL = os.getenv("BGB_LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("BGB_LOG_FILE", None)


def _rich_available() -> bool:
    """Check if Rich is importable without importing it."""
    return importlib.util.find_spec("rich") is not None


# Default to simple unless explicitly requested and conditions are met
def _get_default_format() -> str:
    """Determine default log format based on environment."""
    # Force simple in CI, pytest, Modal
    if os.getenv("CI") or os.getenv("PYTEST_CURRENT_TEST") or os.getenv("MODAL_FUNCTION_ID"):
        return "simple"
    # Force simple if explicitly requested
    if os.getenv("BGB_FORCE_SIMPLE") == "1":
        return "simple"
    # Only use rich if TTY and available and not forced off
    if sys.stderr.isatty() and _rich_available() and os.getenv("BGB_FORCE_RICH") != "0":
        return "rich"
    return "simple"


LOG_FORMAT = os.getenv("BGB_LOG_FORMAT", _get_default_format())
LOG_RING_BUFFER_SIZE = int(os.getenv("BGB_LOG_RING_BUFFER_SIZE", "1000"))

# Thread-safe singleton instance
_lock = threading.Lock()
_instance: Optional["LoggingConfig"] = None


class RingBufferHandler(logging.Handler):
    """High-performance ring buffer handler for debugging.

    Keeps last N log records in memory for post-mortem analysis.
    Zero allocation after buffer fills - reuses existing slots.
    """

    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.buffer: deque[logging.LogRecord] = deque(maxlen=capacity)
        self.lock = cast(Any, threading.RLock())  # Use RLock for re-entrancy

    def emit(self, record: logging.LogRecord) -> None:
        """Add record to ring buffer with thread safety."""
        try:
            if self.lock:
                with self.lock:
                    self.buffer.append(record)
            else:
                self.buffer.append(record)
        except Exception:
            # Never log here - just swallow to avoid re-entrancy
            self.handleError(record)

    def get_records(self, n: int | None = None) -> list[logging.LogRecord]:
        """Get last n records (or all if n is None)."""
        if self.lock:
            with self.lock:
                if n is None:
                    return list(self.buffer)
                return list(self.buffer)[-n:]
        else:
            if n is None:
                return list(self.buffer)
            return list(self.buffer)[-n:]

    def clear(self) -> None:
        """Clear the buffer."""
        if self.lock:
            with self.lock:
                self.buffer.clear()
        else:
            self.buffer.clear()


class PerformanceFilter(logging.Filter):
    """Performance-conscious filter for high-frequency logs.

    Gates logs by step count to prevent spam while maintaining visibility.
    Uses efficient modulo check with configurable step size.
    """

    def __init__(self, every_n_steps: int = LOG_EVERY_N_STEPS):
        super().__init__()
        self.every_n_steps = every_n_steps
        self.step_counters: dict[str, int] = {}
        self.lock = cast(Any, threading.RLock())  # Use RLock for re-entrancy

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter based on step count if record has step attribute."""
        # Fast path: no step attribute means pass through
        step = getattr(record, "step", None)
        if step is None:
            return True

        # Fast path: always pass first/last steps
        if step == 0 or hasattr(record, "is_last"):
            return True

        # Check if we should log this step
        if step % self.every_n_steps == 0:
            # Only track if we're actually logging
            if self.lock:
                with self.lock:
                    key = f"{record.name}:{getattr(record, 'funcName', '')}"
                    self.step_counters[key] = step
            else:
                key = f"{record.name}:{getattr(record, 'funcName', '')}"
                self.step_counters[key] = step
            return True

        return False


class LoggingConfig:
    """Elite logging configuration with singleton pattern.

    Thread-safe, lazy initialization, zero overhead when disabled.
    Follows Google's internal logging best practices.
    """

    def __init__(self) -> None:
        self.is_configured = False
        self.handlers: dict[str, logging.Handler] = {}
        self.ring_buffer: RingBufferHandler | None = None
        self.performance_filter: PerformanceFilter | None = None
        self.console: Console | None = None
        self._original_levels: dict[str, int] = {}

    def setup(
        self,
        level: str | int = LOG_LEVEL,
        log_file: str | Path | None = LOG_FILE,
        format_style: str = LOG_FORMAT,
        force: bool = False,
        enable_ring_buffer: bool = True,
        enable_performance_filter: bool = True,
    ) -> None:
        """Configure logging with production-grade settings."""
        with _lock:
            if self.is_configured and not force:
                return

            # Clear any existing handlers (clean slate principle)
            root_logger = logging.getLogger()
            root_logger.handlers.clear()

            # Convert string level to int
            if isinstance(level, str):
                level = getattr(logging, level.upper(), logging.INFO)

            # Auto-raise to DEBUG for smoke/NaN debug modes
            if os.getenv("BGB_SMOKE_TEST", "0") == "1" or os.getenv("BGB_NAN_DEBUG", "0") == "1":
                level = min(level, logging.DEBUG)

            # Configure console handler based on format
            if format_style == "rich":
                self._setup_rich_handler(root_logger, level)
            elif format_style == "json":
                self._setup_json_handler(root_logger, level)
            else:
                self._setup_simple_handler(root_logger, level)

            # Add file handler if specified
            if log_file:
                self._setup_file_handler(root_logger, log_file)

            # Add ring buffer for debugging (zero overhead when not accessed)
            if enable_ring_buffer:
                self.ring_buffer = RingBufferHandler(LOG_RING_BUFFER_SIZE)
                self.ring_buffer.setLevel(logging.DEBUG)
                # Dynamic attribute for tracking BGB-owned handlers (cleanup safety)
                self.ring_buffer._bgb_owned = True  # type: ignore[attr-defined]
                root_logger.addHandler(self.ring_buffer)

            # Add performance filter for high-frequency paths
            if enable_performance_filter:
                self.performance_filter = PerformanceFilter(LOG_EVERY_N_STEPS)
                for handler in root_logger.handlers:
                    handler.addFilter(self.performance_filter)

            # Set root logger level
            root_logger.setLevel(level)

            # Suppress noisy third-party loggers (Clean Code: reduce noise)
            self._suppress_noisy_loggers()

            # Capture Python warnings into logging system
            logging.captureWarnings(True)

            # Register cleanup on exit (skip in tests to avoid hangs)
            if not os.getenv("PYTEST_CURRENT_TEST"):
                atexit.register(self.cleanup)

            self.is_configured = True

    def _setup_rich_handler(self, logger: logging.Logger, level: int) -> None:
        """Configure Rich handler (falls back to simple when unavailable)."""
        # Multiple safety checks
        if not sys.stderr.isatty():
            self._setup_simple_handler(logger, level)
            return

        # Don't use Rich in CI/pytest/Modal
        if os.getenv("CI") or os.getenv("PYTEST_CURRENT_TEST") or os.getenv("MODAL_FUNCTION_ID"):
            self._setup_simple_handler(logger, level)
            return

        # Lazy import Rich components only when actually needed
        try:
            from rich.console import Console as RichConsole
            from rich.logging import RichHandler as RichLogHandler
        except ImportError:
            self._setup_simple_handler(logger, level)
            return

        self.console = RichConsole(stderr=True, force_terminal=False)

        handler = RichLogHandler(
            console=self.console,
            rich_tracebacks=True,
            show_path=True,
            markup=True,
            log_time_format="[%Y-%m-%d %H:%M:%S.%f]",
            omit_repeated_times=False,
        )
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(message)s"))
        # Dynamic attribute for tracking BGB-owned handlers (cleanup safety)
        handler._bgb_owned = True  # type: ignore[attr-defined]

        logger.addHandler(handler)
        self.handlers["console"] = handler

    def _setup_simple_handler(self, logger: logging.Logger, level: int) -> None:
        """Configure simple stream handler for basic output."""
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        # Professional format with microsecond precision
        formatter = logging.Formatter(
            "[%(asctime)s.%(msecs)03d][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        # Dynamic attribute for tracking BGB-owned handlers (cleanup safety)
        handler._bgb_owned = True  # type: ignore[attr-defined]

        logger.addHandler(handler)
        self.handlers["console"] = handler

    def _setup_json_handler(self, logger: logging.Logger, level: int) -> None:
        """Configure JSON handler for structured logging."""
        # For now, use simple format with structured message
        # Full JSON can be added with python-json-logger if needed
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)

        # Structured format that's easily parseable
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","name":"%(name)s","level":"%(levelname)s",'
            '"message":"%(message)s","module":"%(module)s","function":"%(funcName)s"}'
        )
        handler.setFormatter(formatter)
        # Dynamic attribute for tracking BGB-owned handlers (cleanup safety)
        handler._bgb_owned = True  # type: ignore[attr-defined]

        logger.addHandler(handler)
        self.handlers["console"] = handler

    def _setup_file_handler(self, logger: logging.Logger, log_file: str | Path) -> None:
        """Configure file handler with rotation support."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Use simple FileHandler in tests to avoid xdist worker crashes
        # RotatingFileHandler can cause issues with parallel test execution
        if os.getenv("PYTEST_CURRENT_TEST"):
            # xdist/WSL-safe: lazy open, no rotation in tests
            handler = logging.FileHandler(log_path, encoding="utf-8", delay=True)
        else:
            # Use RotatingFileHandler for production
            from logging.handlers import RotatingFileHandler

            handler = RotatingFileHandler(
                log_path,
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=5,
                encoding="utf-8",
                delay=True,  # Lazy file opening prevents races
            )

        # Detailed format for files
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s"
        )
        handler.setFormatter(formatter)

        # Dynamic attribute for tracking BGB-owned handlers (cleanup safety)
        handler._bgb_owned = True  # type: ignore[attr-defined]

        logger.addHandler(handler)
        self.handlers["file"] = handler

    def _suppress_noisy_loggers(self) -> None:
        """Suppress known noisy third-party loggers."""
        noisy_loggers = [
            "matplotlib",
            "matplotlib.pyplot",
            "matplotlib.font_manager",
            "PIL",
            "PIL.Image",
            "torch.nn.parallel.distributed",
            "torch.distributed",
            "torch.nn.parallel",
            "transformers",
            "urllib3",
            "requests",
            "botocore",
            "boto3",
            "asyncio",
        ]

        for logger_name in noisy_loggers:
            logger = logging.getLogger(logger_name)
            self._original_levels[logger_name] = logger.level
            logger.setLevel(logging.WARNING)

    @contextmanager
    def temporary_level(self, level: str | int) -> Iterator[None]:
        """Context manager for temporary log level change.

        Usage:
            with logging_config.temporary_level("DEBUG"):
                # Debug logs enabled here
                debug_function()
            # Original level restored
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        root_logger = logging.getLogger()
        original_level = root_logger.level

        try:
            root_logger.setLevel(level)
            yield
        finally:
            root_logger.setLevel(original_level)

    @contextmanager
    def suppress(self, *logger_names: str) -> Iterator[None]:
        """Context manager to temporarily suppress specific loggers.

        Usage:
            with logging_config.suppress("torch", "transformers"):
                # Torch and transformers logs suppressed here
                noisy_function()
            # Original levels restored
        """
        original_levels = {}

        try:
            for name in logger_names:
                logger = logging.getLogger(name)
                original_levels[name] = logger.level
                logger.setLevel(logging.CRITICAL + 1)
            yield
        finally:
            for name, level in original_levels.items():
                logging.getLogger(name).setLevel(level)

    def get_last_logs(self, n: int = 100) -> list[str]:
        """Get last n log messages from ring buffer.

        Useful for debugging and crash analysis.
        """
        if not self.ring_buffer:
            return []

        records = self.ring_buffer.get_records(n)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        return [formatter.format(record) for record in records]

    def cleanup(self) -> None:
        """Clean up resources on exit."""
        root_logger = logging.getLogger()

        # First, remove our handlers from root to prevent emits to closed files
        for handler in list(root_logger.handlers):
            if getattr(handler, "_bgb_owned", False):
                with contextlib.suppress(Exception):
                    root_logger.removeHandler(handler)
                with contextlib.suppress(Exception):
                    handler.flush()
                    handler.close()

        # Also clean up any handlers we tracked explicitly
        for handler in list(self.handlers.values()):
            with contextlib.suppress(Exception):
                if handler in root_logger.handlers:
                    root_logger.removeHandler(handler)
            with contextlib.suppress(Exception):
                handler.flush()
                handler.close()

        # Clear our handler registry
        self.handlers.clear()

        # Clean up ring buffer (if not already cleaned)
        if self.ring_buffer:
            with contextlib.suppress(Exception):
                if self.ring_buffer in root_logger.handlers:
                    root_logger.removeHandler(self.ring_buffer)
            with contextlib.suppress(Exception):
                self.ring_buffer.flush()
                self.ring_buffer.close()


def get_instance() -> LoggingConfig:
    """Get or create the singleton logging configuration instance.

    Thread-safe lazy initialization following Google's pattern.
    """
    global _instance

    if _instance is None:
        with _lock:
            if _instance is None:  # Double-check locking pattern
                _instance = LoggingConfig()

    return _instance


def setup_logging(
    level: str | int = LOG_LEVEL,
    log_file: str | Path | None = LOG_FILE,
    format_style: str = LOG_FORMAT,
    force: bool = False,
) -> LoggingConfig:
    """Configure logging for the application.

    This is the main entry point for logging configuration.
    Should be called once from application entrypoints.

    Returns:
        LoggingConfig: The configured logging instance
    """
    instance = get_instance()
    instance.setup(
        level=level,
        log_file=log_file,
        format_style=format_style,
        force=force,
    )
    return instance


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    Convenience function that ensures logging is configured.

    Args:
        name: Logger name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Ensure logging is configured with defaults if not already done
    if not get_instance().is_configured:
        setup_logging()

    return logging.getLogger(name)


# Performance-conscious logging utilities
_LOG_ONCE_SEEN: set[str] = set()


def log_once(logger: logging.Logger, level: int, msg: str, key: str) -> None:
    """Log a message only once per key.

    Useful for warnings that would otherwise spam.
    """
    if key not in _LOG_ONCE_SEEN:
        logger.log(level, msg)
        _LOG_ONCE_SEEN.add(key)


_LOG_EVERY_N_COUNTS: dict[str, int] = {}


def log_every_n(logger: logging.Logger, level: int, msg: str, n: int, key: str) -> None:
    """Log a message every n occurrences.

    Useful for progress updates without spam.
    """
    count = _LOG_EVERY_N_COUNTS.get(key, 0) + 1
    _LOG_EVERY_N_COUNTS[key] = count

    if count % n == 1 or n == 1:
        logger.log(level, f"{msg} [occurrence {count}]")


# Export public API
__all__ = [
    "LOG_EVERY_N_STEPS",
    "LOG_FILE",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "LoggingConfig",
    "PerformanceFilter",
    "RingBufferHandler",
    "get_instance",
    "get_logger",
    "log_every_n",
    "log_once",
    "setup_logging",
]
