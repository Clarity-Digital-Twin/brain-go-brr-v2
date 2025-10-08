"""Tests for W&B resume logic.

Testing philosophy: Test BEHAVIOR, not implementation.
Focus on run ID file handling - minimal wandb mocking.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.brain_brr.config.schemas import Config, ExperimentConfig, WandbConfig

pytest.importorskip("wandb", reason="W&B tests require wandb package")

from src.brain_brr.train.wandb_integration import WandBLogger


@pytest.fixture
def mock_config():
    """Create config with W&B enabled."""
    config = Config()
    config.experiment = ExperimentConfig()
    config.experiment.wandb = WandbConfig(
        enabled=True, project="test-project", entity="test-entity"
    )
    config.experiment.name = "test-run"
    return config


class TestWandBRunIDPersistence:
    """Test run ID file creation and atomic writes."""

    def test_creates_run_id_file_on_init(self, mock_config):
        """Should create .wandb_run_id file on initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=False)

                run_id_path = Path(tmpdir) / ".wandb_run_id"
                assert run_id_path.exists()

    def test_run_id_file_contains_valid_uuid(self, mock_config):
        """Run ID file should contain a valid UUID hex string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=False)

                run_id_path = Path(tmpdir) / ".wandb_run_id"
                run_id = run_id_path.read_text().strip()

                assert len(run_id) == 32
                assert all(c in "0123456789abcdef" for c in run_id)

    def test_no_temp_file_remains_after_write(self, mock_config):
        """Temp .tmp file should be removed after atomic write."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=False)

                run_id_path = Path(tmpdir) / ".wandb_run_id"
                temp_path = run_id_path.with_suffix(".tmp")

                assert not temp_path.exists()


class TestWandBResumeBehavior:
    """Test resume flag behavior with existing run ID files."""

    def test_resume_true_with_existing_file_reuses_id(self, mock_config):
        """resume=True + existing file → should reuse run ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            run_id_path = Path(tmpdir) / ".wandb_run_id"
            existing_id = "abc123def456"
            run_id_path.write_text(existing_id)

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=True)

                mock_wandb.init.assert_called_once()
                call_kwargs = mock_wandb.init.call_args.kwargs
                assert call_kwargs["id"] == existing_id
                assert call_kwargs["resume"] == "allow"

    def test_resume_true_without_file_creates_new_id(self, mock_config):
        """resume=True + no file → should create new run ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=True)

                run_id_path = Path(tmpdir) / ".wandb_run_id"
                new_id = run_id_path.read_text().strip()

                mock_wandb.init.assert_called_once()
                call_kwargs = mock_wandb.init.call_args.kwargs
                assert call_kwargs["id"] == new_id
                assert len(new_id) == 32

    def test_resume_false_with_existing_file_creates_new_id(self, mock_config):
        """resume=False + existing file → should create new run ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            run_id_path = Path(tmpdir) / ".wandb_run_id"
            existing_id = "abc123def456"
            run_id_path.write_text(existing_id)

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=False)

                new_id = run_id_path.read_text().strip()

                mock_wandb.init.assert_called_once()
                call_kwargs = mock_wandb.init.call_args.kwargs
                assert call_kwargs["id"] == new_id
                assert call_kwargs["id"] != existing_id

    def test_resume_false_without_file_creates_new_id(self, mock_config):
        """resume=False + no file → should create new run ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=False)

                run_id_path = Path(tmpdir) / ".wandb_run_id"
                new_id = run_id_path.read_text().strip()

                assert len(new_id) == 32


class TestWandBResumeEdgeCases:
    """Test edge cases in resume logic."""

    def test_empty_run_id_file_creates_new_id(self, mock_config):
        """Empty run ID file should trigger new ID creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            run_id_path = Path(tmpdir) / ".wandb_run_id"
            run_id_path.write_text("")

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=True)

                new_id = run_id_path.read_text().strip()

                assert len(new_id) == 32

    def test_whitespace_only_run_id_creates_new_id(self, mock_config):
        """Whitespace-only run ID file should trigger new ID creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            run_id_path = Path(tmpdir) / ".wandb_run_id"
            run_id_path.write_text("   \n  ")

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=True)

                new_id = run_id_path.read_text().strip()

                assert len(new_id) == 32


class TestWandBGracefulFallback:
    """Test graceful fallback when W&B not available."""

    def test_disabled_when_not_in_config(self):
        """Should disable W&B if not in config."""
        config = Config()
        config.experiment = ExperimentConfig()
        config.experiment.wandb = WandbConfig(enabled=False)

        logger = WandBLogger(config, resume=False)

        assert logger.enabled is False
        assert logger.run is None

    def test_disabled_when_not_installed(self, mock_config):
        """Should disable W&B if wandb not installed."""
        with patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", False):
            logger = WandBLogger(mock_config, resume=False)

            assert logger.enabled is False
            assert logger.run is None

    def test_disabled_when_no_api_key(self, mock_config):
        """Should disable W&B if WANDB_API_KEY not set."""
        with (
            patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
            patch.dict(os.environ, {}, clear=True),
        ):
            logger = WandBLogger(mock_config, resume=False)

            assert logger.enabled is False
            assert logger.run is None


class TestWandBAtomicWrite:
    """Test atomic write behavior for run ID file."""

    def test_run_id_file_survives_concurrent_writes(self, mock_config):
        """Atomic write should prevent torn writes (simplified test)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_config.experiment.output_dir = tmpdir

            with (
                patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
                patch.dict(os.environ, {"WANDB_API_KEY": "test-key"}),
                patch("src.brain_brr.train.wandb_integration.wandb") as mock_wandb,
            ):
                mock_wandb.init.return_value = MagicMock()

                WandBLogger(mock_config, resume=False)

                run_id_path = Path(tmpdir) / ".wandb_run_id"
                run_id = run_id_path.read_text()

                assert len(run_id.strip()) == 32
                assert not run_id.startswith("\x00")
