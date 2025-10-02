"""Tests for logging pattern utilities.

Testing philosophy: Test BEHAVIOR with real loggers, minimal mocks.
Verify gating logic, formatting, and structured logging extras.
"""

from __future__ import annotations

import logging
import os
from unittest import mock

import pytest

from src.brain_brr.utils.logging_patterns import (
    log_batch_metrics,
    log_data_loading_progress,
    log_epoch_progress,
    log_nan_detection_efficient,
)


@pytest.fixture
def test_logger():
    """Create a test logger with captured records."""
    logger = logging.getLogger("test_logging_patterns")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # Capture log records
    records: list[logging.LogRecord] = []

    class ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = ListHandler()
    logger.addHandler(handler)

    # Attach records list to logger for easy access
    logger.records = records  # type: ignore[attr-defined]
    return logger


class TestBatchMetricsLogging:
    """Test log_batch_metrics gating and formatting."""

    def test_logs_when_debug_enabled(self, test_logger):
        """Should log when DEBUG level enabled."""
        test_logger.setLevel(logging.DEBUG)
        test_logger.records.clear()

        log_batch_metrics(test_logger, step=0, loss=0.5, accuracy=0.9)

        assert len(test_logger.records) == 1
        record = test_logger.records[0]
        assert record.levelno == logging.DEBUG
        assert "Step 0" in record.getMessage()

    def test_does_not_log_when_debug_disabled(self, test_logger):
        """Should not log when DEBUG disabled (INFO level)."""
        test_logger.setLevel(logging.INFO)
        test_logger.records.clear()

        log_batch_metrics(test_logger, step=10, loss=0.5, accuracy=0.9)

        # Should be gated - no log records
        assert len(test_logger.records) == 0

    def test_step_gating_with_log_every_n(self, test_logger):
        """Should only log every Nth step."""
        test_logger.setLevel(logging.DEBUG)

        # Patch LOG_EVERY_N_STEPS to 50
        with mock.patch("src.brain_brr.utils.logging_patterns.LOG_EVERY_N_STEPS", 50):
            test_logger.records.clear()

            # Step 0 always logs
            log_batch_metrics(test_logger, step=0, loss=0.5)
            assert len(test_logger.records) == 1

            test_logger.records.clear()

            # Step 25 should NOT log (not multiple of 50)
            log_batch_metrics(test_logger, step=25, loss=0.5)
            assert len(test_logger.records) == 0

            # Step 50 should log
            log_batch_metrics(test_logger, step=50, loss=0.5)
            assert len(test_logger.records) == 1

    def test_step_zero_always_logs(self, test_logger):
        """Step 0 should always log."""
        test_logger.setLevel(logging.DEBUG)

        with mock.patch("src.brain_brr.utils.logging_patterns.LOG_EVERY_N_STEPS", 100):
            test_logger.records.clear()
            log_batch_metrics(test_logger, step=0, loss=0.5)
            assert len(test_logger.records) == 1

    def test_includes_extra_metrics(self, test_logger):
        """Should include extra metrics in structured logging."""
        test_logger.setLevel(logging.DEBUG)
        test_logger.records.clear()

        log_batch_metrics(test_logger, step=0, loss=0.5, accuracy=0.9, lr=0.001, grad_norm=1.5)

        record = test_logger.records[0]
        assert "step" in record.__dict__
        assert record.step == 0
        assert "metrics" in record.__dict__
        assert record.metrics["loss"] == 0.5
        assert record.metrics["accuracy"] == 0.9
        assert record.metrics["lr"] == 0.001
        assert record.metrics["grad_norm"] == 1.5

    def test_handles_none_accuracy(self, test_logger):
        """Should handle None accuracy gracefully."""
        test_logger.setLevel(logging.DEBUG)
        test_logger.records.clear()

        log_batch_metrics(test_logger, step=0, loss=0.5, accuracy=None)

        record = test_logger.records[0]
        # Should use 0.0 for None accuracy
        assert "Acc: 0.0000" in record.getMessage()


class TestEpochProgressLogging:
    """Test log_epoch_progress formatting."""

    def test_logs_at_info_level(self, test_logger):
        """Should always log at INFO level."""
        test_logger.setLevel(logging.INFO)
        test_logger.records.clear()

        log_epoch_progress(
            test_logger, epoch=5, total_epochs=100, phase="train", loss=0.3, acc=0.95
        )

        assert len(test_logger.records) == 1
        record = test_logger.records[0]
        assert record.levelno == logging.INFO

    def test_formats_phase_uppercase(self, test_logger):
        """Should format phase as uppercase."""
        test_logger.setLevel(logging.INFO)
        test_logger.records.clear()

        log_epoch_progress(test_logger, epoch=1, total_epochs=10, phase="train", loss=0.5)

        record = test_logger.records[0]
        assert "[TRAIN]" in record.getMessage()

    def test_includes_structured_extras(self, test_logger):
        """Should include structured logging extras."""
        test_logger.setLevel(logging.INFO)
        test_logger.records.clear()

        log_epoch_progress(
            test_logger, epoch=5, total_epochs=100, phase="val", loss=0.3, acc=0.95, auroc=0.98
        )

        record = test_logger.records[0]
        assert record.epoch == 5
        assert record.phase == "val"
        assert record.metrics["loss"] == 0.3
        assert record.metrics["acc"] == 0.95
        assert record.metrics["auroc"] == 0.98


class TestNaNDetectionLogging:
    """Test log_nan_detection_efficient conditional logging."""

    def test_debug_mode_logs_detailed_info(self, test_logger):
        """When BGB_NAN_DEBUG=1, should log DEBUG info."""
        test_logger.setLevel(logging.DEBUG)

        with mock.patch.dict(os.environ, {"BGB_NAN_DEBUG": "1"}):
            test_logger.records.clear()
            log_nan_detection_efficient(test_logger, location="forward_pass", step=100)

            assert len(test_logger.records) == 1
            record = test_logger.records[0]
            assert record.levelno == logging.DEBUG
            assert "debug mode" in record.getMessage()
            assert record.nan_location == "forward_pass"
            assert record.step == 100

    def test_non_debug_mode_logs_warning(self, test_logger):
        """When BGB_NAN_DEBUG not set, should log WARNING."""
        test_logger.setLevel(logging.DEBUG)

        with mock.patch.dict(os.environ, {}, clear=True):
            test_logger.records.clear()
            log_nan_detection_efficient(test_logger, location="loss_computation", step=50)

            assert len(test_logger.records) == 1
            record = test_logger.records[0]
            assert record.levelno == logging.WARNING
            assert "NaN at loss_computation" in record.getMessage()

    def test_should_debug_flag_forces_debug_log(self, test_logger):
        """should_debug=True should force DEBUG log."""
        test_logger.setLevel(logging.DEBUG)

        with mock.patch.dict(os.environ, {}, clear=True):
            test_logger.records.clear()
            log_nan_detection_efficient(
                test_logger, location="gradient", step=200, should_debug=True
            )

            record = test_logger.records[0]
            assert record.levelno == logging.DEBUG


class TestDataLoadingProgressLogging:
    """Test log_data_loading_progress percentage gating."""

    def test_logs_at_10_percent_intervals(self, test_logger):
        """Should log every 10% of progress."""
        test_logger.setLevel(logging.INFO)

        # 100 total files
        total_files = 100

        for files_processed in range(total_files + 1):
            test_logger.records.clear()
            log_data_loading_progress(
                test_logger,
                files_processed=files_processed,
                total_files=total_files,
                current_file=f"file_{files_processed}.edf",
            )

            percent = (files_processed * 100) // total_files
            if percent % 10 == 0 and files_processed > 0:
                # Should log
                assert len(test_logger.records) == 1
                record = test_logger.records[0]
                assert f"{percent}%" in record.getMessage()
            else:
                # Should not log
                assert len(test_logger.records) == 0

    def test_does_not_log_at_zero_files(self, test_logger):
        """Should not log at 0 files processed."""
        test_logger.setLevel(logging.INFO)
        test_logger.records.clear()

        log_data_loading_progress(test_logger, files_processed=0, total_files=100, current_file="")

        assert len(test_logger.records) == 0

    def test_includes_current_file_in_message(self, test_logger):
        """Should include current file name in log."""
        test_logger.setLevel(logging.INFO)
        test_logger.records.clear()

        log_data_loading_progress(
            test_logger, files_processed=10, total_files=100, current_file="test.edf"
        )

        record = test_logger.records[0]
        assert "test.edf" in record.getMessage()

    def test_handles_edge_case_percentages(self, test_logger):
        """Should handle non-round percentages correctly."""
        test_logger.setLevel(logging.INFO)

        # 333 files total (percentages won't be exact multiples)
        total_files = 333

        # 33 files = 9.9% → should NOT log
        test_logger.records.clear()
        log_data_loading_progress(
            test_logger, files_processed=33, total_files=total_files, current_file="test.edf"
        )
        assert len(test_logger.records) == 0

        # 34 files = 10.2% → SHOULD log (10% milestone)
        test_logger.records.clear()
        log_data_loading_progress(
            test_logger, files_processed=34, total_files=total_files, current_file="test.edf"
        )
        assert len(test_logger.records) == 1


class TestLoggingPatternsIntegration:
    """Integration tests for logging patterns."""

    def test_batch_metrics_with_real_logger_hierarchy(self):
        """Test with hierarchical logger setup."""
        parent = logging.getLogger("brain_brr.train")
        child = logging.getLogger("brain_brr.train.loop")
        child.setLevel(logging.DEBUG)

        records: list[logging.LogRecord] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = ListHandler()
        parent.addHandler(handler)

        log_batch_metrics(child, step=0, loss=0.5, accuracy=0.9)

        # Should propagate to parent handler
        assert len(records) >= 1

        # Cleanup
        parent.removeHandler(handler)

    def test_all_functions_handle_unicode(self, test_logger):
        """All logging functions should handle unicode gracefully."""
        test_logger.setLevel(logging.DEBUG)

        with mock.patch.dict(os.environ, {"BGB_NAN_DEBUG": "1"}):
            # Unicode in metrics
            log_batch_metrics(test_logger, step=0, loss=0.5, metric_名前=123.45)

            # Unicode in file name
            log_data_loading_progress(test_logger, 10, 100, current_file="ファイル.edf")

            # Unicode in phase
            log_epoch_progress(test_logger, 1, 10, phase="訓練", loss=0.5)

            # Unicode in location
            log_nan_detection_efficient(test_logger, location="前方パス", step=1)

        # Should not crash - all logs processed
        assert len(test_logger.records) >= 4
