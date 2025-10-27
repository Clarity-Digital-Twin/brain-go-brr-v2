"""Tests for .env file integration in training loop.

Critical tests to prevent WandB offline mode regression.
Tests that environment variables are loaded before training starts.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestDotenvLoading:
    """Test that .env file is loaded correctly during training initialization."""

    def test_load_dotenv_in_main_function(self):
        """Verify load_dotenv() exists in main() source code."""
        import inspect

        from src.brain_brr.train.loop import main

        source = inspect.getsource(main)

        # Verify load_dotenv is imported
        assert "from dotenv import load_dotenv" in source, "load_dotenv not imported in main()"

        # Verify load_dotenv is called
        assert "load_dotenv()" in source, "load_dotenv() not called in main()"

        # Verify it's called early (before dataset loading)
        assert source.index("load_dotenv()") < source.index("load_tusz_for_training"), (
            "load_dotenv() must be called before dataset loading"
        )

    def test_load_dotenv_called_early_in_main(self):
        """Verify load_dotenv() is called early in main() before training starts."""
        import inspect

        from src.brain_brr.train.loop import main

        source = inspect.getsource(main)

        load_dotenv_idx = source.index("load_dotenv()")

        # Verify it's called before training function
        train_idx = source.index("train(")
        assert load_dotenv_idx < train_idx, "load_dotenv() must be called before train() function"

        # Verify it's within first 1000 characters of function (early in main)
        assert load_dotenv_idx < 1000, "load_dotenv() should be called early in main() function"

    def test_env_vars_available_from_dotenv(self):
        """Test that env vars from .env are available after load_dotenv()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake .env file
            env_file = Path(tmpdir) / ".env"
            env_file.write_text("TEST_VAR=test_value\nWANDB_API_KEY=fake_key\n")

            # Clear any existing TEST_VAR
            original_test_var = os.environ.pop("TEST_VAR", None)
            original_wandb_key = os.environ.pop("WANDB_API_KEY", None)

            try:
                # Load dotenv with explicit path
                from dotenv import load_dotenv

                load_dotenv(env_file)

                # Verify env vars are loaded
                assert os.getenv("TEST_VAR") == "test_value", "TEST_VAR not loaded from .env"
                assert os.getenv("WANDB_API_KEY") == "fake_key", (
                    "WANDB_API_KEY not loaded from .env"
                )

            finally:
                # Cleanup - restore original values
                os.environ.pop("TEST_VAR", None)
                os.environ.pop("WANDB_API_KEY", None)
                if original_test_var:
                    os.environ["TEST_VAR"] = original_test_var
                if original_wandb_key:
                    os.environ["WANDB_API_KEY"] = original_wandb_key


class TestDotenvSecurityChecks:
    """Test that .env file is properly protected from git commits."""

    def test_dotenv_file_is_gitignored(self):
        """Verify .env is in .gitignore to prevent secret leaks."""
        gitignore_path = Path(__file__).parent.parent.parent.parent / ".gitignore"
        assert gitignore_path.exists(), ".gitignore file not found"

        gitignore_content = gitignore_path.read_text()

        # Check for .env in gitignore
        assert ".env" in gitignore_content, (
            ".env file is NOT in .gitignore - CRITICAL SECURITY ISSUE"
        )

    def test_wandb_directory_is_gitignored(self):
        """Verify wandb/ directory is in .gitignore to prevent committing local runs."""
        gitignore_path = Path(__file__).parent.parent.parent.parent / ".gitignore"
        assert gitignore_path.exists(), ".gitignore file not found"

        gitignore_content = gitignore_path.read_text()

        # Check for wandb/ in gitignore
        assert "wandb/" in gitignore_content, (
            "wandb/ directory is NOT in .gitignore - can leak WandB data"
        )


class TestWandBOfflineModePrevention:
    """Tests to prevent WandB offline mode regression."""

    def test_wandb_fails_gracefully_without_api_key(self):
        """WandB should fail gracefully when WANDB_API_KEY is missing."""
        from src.brain_brr.config.schemas import Config, ExperimentConfig, WandbConfig

        config = Config()
        config.experiment = ExperimentConfig()
        config.experiment.output_dir = tempfile.mkdtemp()
        config.experiment.wandb = WandbConfig(
            enabled=True, project="test-project", entity="test-entity"
        )
        config.experiment.name = "test-run"

        # Ensure no API key in environment
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
        ):
            from src.brain_brr.train.wandb_integration import WandBLogger

            logger = WandBLogger(config, resume=False)

            # Should not enable WandB without API key
            assert logger.enabled is False, "WandB should be disabled without API key"

    def test_wandb_works_with_api_key_from_env(self):
        """WandB should initialize successfully when WANDB_API_KEY is in environment."""
        from src.brain_brr.config.schemas import Config, ExperimentConfig, WandbConfig

        config = Config()
        config.experiment = ExperimentConfig()
        config.experiment.output_dir = tempfile.mkdtemp()
        config.experiment.wandb = WandbConfig(
            enabled=True, project="test-project", entity="test-entity"
        )
        config.experiment.name = "test-run"

        # Create a mock wandb module with all necessary attributes
        mock_wandb = MagicMock()
        mock_wandb.init.return_value = MagicMock()

        with (
            patch.dict(os.environ, {"WANDB_API_KEY": "test-key-12345"}),
            patch("src.brain_brr.train.wandb_integration.WANDB_AVAILABLE", True),
            patch("src.brain_brr.train.wandb_integration.wandb", mock_wandb, create=True),
        ):
            from src.brain_brr.train.wandb_integration import WandBLogger

            logger = WandBLogger(config, resume=False)

            # Should enable WandB with API key
            assert logger.enabled is True, "WandB should be enabled with API key present"
            assert mock_wandb.init.called, "wandb.init() should be called when API key is present"
