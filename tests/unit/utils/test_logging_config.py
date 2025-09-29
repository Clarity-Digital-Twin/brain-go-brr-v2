"""Unit tests for logging configuration infrastructure.

Tests thread-safety, singleton pattern, performance filters, and configuration options.
Professional test suite following Google's testing standards.
"""

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.brain_brr.utils.logging_config import (
    LoggingConfig,
    PerformanceFilter,
    RingBufferHandler,
    get_instance,
    get_logger,
    log_every_n,
    log_once,
    setup_logging,
)


class TestLoggingConfig:
    """Test the main LoggingConfig class."""

    def test_singleton_pattern(self):
        """Test that LoggingConfig follows singleton pattern."""
        instance1 = get_instance()
        instance2 = get_instance()
        assert instance1 is instance2

    def test_thread_safe_singleton(self):
        """Test thread-safe singleton initialization."""
        instances = []

        def get_and_store():
            instances.append(get_instance())

        threads = [threading.Thread(target=get_and_store) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert all(inst is instances[0] for inst in instances)

    def test_setup_idempotency(self):
        """Test that setup can be called multiple times safely."""
        config = LoggingConfig()
        config.setup(level="INFO")
        assert config.is_configured

        # Second call should not reconfigure unless forced
        with patch.object(config, "_setup_simple_handler") as mock_handler:
            config.setup(level="DEBUG")
            mock_handler.assert_not_called()

        # Force should reconfigure
        # Note: In pytest environment, always uses simple handler, not rich
        with patch.object(config, "_setup_simple_handler") as mock_handler:
            config.setup(level="DEBUG", format_style="simple", force=True)
            mock_handler.assert_called()

    def test_level_configuration(self):
        """Test different log level configurations."""
        config = LoggingConfig()

        # Test string level
        config.setup(level="WARNING", force=True)
        assert logging.getLogger().level == logging.WARNING

        # Test int level
        config.setup(level=logging.DEBUG, force=True)
        assert logging.getLogger().level == logging.DEBUG

    @pytest.mark.parametrize("format_style", ["rich", "simple", "json"])
    def test_format_styles(self, format_style):
        """Test different output format styles."""
        config = LoggingConfig()
        config.setup(format_style=format_style, force=True)

        # Check that appropriate handler was configured
        assert "console" in config.handlers
        handler = config.handlers["console"]
        assert isinstance(handler, logging.Handler)

    def test_file_handler(self, tmp_path):
        """Test file handler configuration."""
        # Create a fresh instance to avoid conflicts
        from src.brain_brr.utils.logging_config import LoggingConfig as TestConfig

        log_file = tmp_path / "test.log"
        config = TestConfig()

        try:
            # Setup with file handler
            config.setup(log_file=str(log_file), force=True)

            # Check file handler exists
            assert "file" in config.handlers

            # Write a test message
            test_logger = logging.getLogger(f"test.file.{id(config)}")
            test_logger.info("Test message for file handler")

            # Force flush to ensure write
            if "file" in config.handlers:
                config.handlers["file"].flush()

            # Check file was created
            assert log_file.exists()

            # Verify content was written
            content = log_file.read_text()
            assert "Test message for file handler" in content
        finally:
            # Clean up handlers
            config.cleanup()
            # Remove all handlers from the test logger
            test_logger.handlers.clear()

            # Verify no bgb-owned handlers remain on root
            for h in logging.getLogger().handlers:
                assert not getattr(h, "_bgb_owned", False)

    def test_suppress_noisy_loggers(self):
        """Test suppression of third-party loggers."""
        config = LoggingConfig()
        config.setup(force=True)

        # Check known noisy loggers are suppressed
        assert logging.getLogger("matplotlib").level >= logging.WARNING
        assert logging.getLogger("torch.nn.parallel.distributed").level >= logging.WARNING

    def test_temporary_level_context_manager(self):
        """Test temporary log level changes."""
        config = get_instance()
        config.setup(level="INFO", force=True)

        original_level = logging.getLogger().level
        assert original_level == logging.INFO

        # Temporarily change to DEBUG
        with config.temporary_level("DEBUG"):
            assert logging.getLogger().level == logging.DEBUG

        # Should revert after context
        assert logging.getLogger().level == original_level

    def test_suppress_context_manager(self):
        """Test logger suppression context manager."""
        config = get_instance()
        test_logger = logging.getLogger("test.suppress")
        original_level = test_logger.level

        with config.suppress("test.suppress"):
            assert test_logger.level > logging.CRITICAL

        assert test_logger.level == original_level

    def test_cleanup_on_exit(self):
        """Test resource cleanup."""
        config = LoggingConfig()
        config.setup(force=True)

        # Mock handlers to test cleanup
        for handler in config.handlers.values():
            handler.flush = MagicMock()
            handler.close = MagicMock()

        config.cleanup()

        # All handlers should be flushed and closed
        for handler in config.handlers.values():
            handler.flush.assert_called_once()
            handler.close.assert_called_once()


class TestRingBufferHandler:
    """Test the RingBufferHandler for debugging."""

    def test_capacity_limit(self):
        """Test that ring buffer respects capacity."""
        handler = RingBufferHandler(capacity=5)

        # Add more records than capacity
        for i in range(10):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        # Should only keep last 5
        records = handler.get_records()
        assert len(records) == 5
        assert records[0].msg == "Message 5"
        assert records[-1].msg == "Message 9"

    def test_thread_safety(self):
        """Test thread-safe operations."""
        handler = RingBufferHandler(capacity=100)

        def add_records(start, count):
            for i in range(start, start + count):
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"Message {i}",
                    args=(),
                    exc_info=None,
                )
                handler.emit(record)

        # Add records from multiple threads
        threads = [threading.Thread(target=add_records, args=(i * 10, 10)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        records = handler.get_records()
        assert len(records) == 50

    def test_clear_buffer(self):
        """Test buffer clearing."""
        handler = RingBufferHandler(capacity=10)

        # Add some records
        for i in range(5):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        assert len(handler.get_records()) == 5

        # Clear buffer
        handler.clear()
        assert len(handler.get_records()) == 0

    def test_get_last_n_records(self):
        """Test retrieving last N records."""
        handler = RingBufferHandler(capacity=10)

        # Add 10 records
        for i in range(10):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Message {i}",
                args=(),
                exc_info=None,
            )
            handler.emit(record)

        # Get last 3
        last_3 = handler.get_records(3)
        assert len(last_3) == 3
        assert last_3[0].msg == "Message 7"
        assert last_3[-1].msg == "Message 9"


class TestPerformanceFilter:
    """Test the PerformanceFilter for high-frequency logs."""

    def test_step_filtering(self):
        """Test that filter gates by step count."""
        perf_filter = PerformanceFilter(every_n_steps=5)

        # Create records with step attribute
        results = []
        for i in range(20):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Step {i}",
                args=(),
                exc_info=None,
            )
            record.step = i
            results.append(perf_filter.filter(record))

        # Should pass: 0, 5, 10, 15
        expected = [i % 5 == 0 for i in range(20)]
        assert results == expected

    def test_non_step_logs_pass(self):
        """Test that logs without step attribute always pass."""
        perf_filter = PerformanceFilter(every_n_steps=10)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Non-step message",
            args=(),
            exc_info=None,
        )
        # No step attribute
        assert perf_filter.filter(record) is True

    def test_first_and_last_steps(self):
        """Test that first and last steps always pass."""
        perf_filter = PerformanceFilter(every_n_steps=100)

        # First step (0) should pass
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="First",
            args=(),
            exc_info=None,
        )
        record.step = 0
        assert perf_filter.filter(record) is True

        # Last step should pass
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Last",
            args=(),
            exc_info=None,
        )
        record.step = 99
        record.is_last = True
        assert perf_filter.filter(record) is True

    def test_per_function_tracking(self):
        """Test that step counts are tracked per function."""
        perf_filter = PerformanceFilter(every_n_steps=3)

        # Create records from different functions
        for func_name in ["train", "validate"]:
            for i in range(5):
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=f"{func_name} {i}",
                    args=(),
                    exc_info=None,
                    func=func_name,
                )
                record.step = i
                record.funcName = func_name
                perf_filter.filter(record)

        # Each function should have its own counter
        assert "test:train" in perf_filter.step_counters
        assert "test:validate" in perf_filter.step_counters


class TestUtilityFunctions:
    """Test utility logging functions."""

    def test_log_once(self):
        """Test log_once utility."""
        logger = logging.getLogger("test.once")
        logger.setLevel(logging.INFO)

        # Create a handler to capture logs
        handler = logging.handlers.MemoryHandler(capacity=100)
        logger.addHandler(handler)

        # First call should log
        log_once(logger, logging.INFO, "Test message", "key1")
        assert len(handler.buffer) == 1

        # Second call with same key should not log
        log_once(logger, logging.INFO, "Test message", "key1")
        assert len(handler.buffer) == 1

        # Different key should log
        log_once(logger, logging.INFO, "Another message", "key2")
        assert len(handler.buffer) == 2

    def test_log_every_n(self):
        """Test log_every_n utility."""
        logger = logging.getLogger("test.every_n")
        logger.setLevel(logging.INFO)

        # Create a handler to capture logs
        handler = logging.handlers.MemoryHandler(capacity=100)
        logger.addHandler(handler)

        # Log every 3 occurrences
        for _i in range(10):
            log_every_n(logger, logging.INFO, "Test", 3, "test_key")

        # Should log on: 1st, 4th, 7th, 10th
        assert len(handler.buffer) == 4

    def test_get_logger(self):
        """Test get_logger convenience function."""
        # Should auto-configure if not already done
        logger = get_logger("test.auto")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.auto"

        # Should ensure logging is configured
        config = get_instance()
        assert config.is_configured


class TestIntegration:
    """Integration tests for the logging system."""

    def test_full_setup_flow(self, tmp_path):
        """Test complete setup flow."""
        log_file = tmp_path / "integration.log"

        # Setup logging with all features
        setup_logging(level="DEBUG", log_file=log_file, format_style="simple", force=True)

        # Get a logger and log messages
        logger = get_logger("integration.test")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")

        # Check that messages were logged to file
        assert log_file.exists()
        content = log_file.read_text()
        assert "Debug message" in content
        assert "Info message" in content
        assert "Warning message" in content

    def test_environment_variable_defaults(self, monkeypatch):
        """Test that environment variables are respected."""
        monkeypatch.setenv("BGB_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("BGB_LOG_FORMAT", "json")
        monkeypatch.setenv("BGB_LOG_EVERY_N_STEPS", "100")

        # Force reimport to pick up env vars
        import importlib

        import src.brain_brr.utils.logging_config as lc

        importlib.reload(lc)

        assert lc.LOG_LEVEL == "WARNING"
        assert lc.LOG_FORMAT == "json"
        assert lc.LOG_EVERY_N_STEPS == 100
