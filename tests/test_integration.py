"""End-to-end integration tests for the complete ML pipeline.

This module tests complete workflows from configuration loading through
experiment execution, ensuring all components work together correctly.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml
from typer.testing import CliRunner

from src.cli import app
from src.experiment.pipeline import Experiment, run_experiment_from_config
from src.experiment.schemas import ExperimentConfig, load_config


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for integration tests."""
    temp_dir = Path(tempfile.mkdtemp())

    # Create directory structure
    (temp_dir / "configs").mkdir()
    (temp_dir / "data" / "raw").mkdir(parents=True)
    (temp_dir / "cache").mkdir()
    (temp_dir / "results").mkdir()

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config(temp_workspace):
    """Create a sample configuration file for testing."""
    config_dict = {
        "data": {
            "dataset": "mnist",
            "batch_size": 8,
            "validation_split": 0.2,
            "download": False,
        },
        "model": {"name": "linear", "input_size": 784, "num_classes": 10},
        "training": {"epochs": 2, "learning_rate": 0.01, "optimizer": "sgd"},
        "evaluation": {"metrics": ["accuracy"], "save_predictions": False},
        "experiment": {
            "name": "integration_test",
            "random_seed": 42,
            "device": "cpu",
            "cache_dir": str(temp_workspace / "cache"),
            "save_model": False,
        },
        "logging": {"log_every_n_steps": 1},
    }

    config_file = temp_workspace / "configs" / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)

    return config_file


class TestConfigurationIntegration:
    """Test configuration loading and validation integration."""

    def test_load_and_validate_sample_configs(self):
        """Test loading and validating all sample configurations."""
        # Test that our predefined configs are valid
        config_files = [
            "configs/local.yaml",
            "configs/production.yaml",
            "configs/example_linear.yaml",
        ]

        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                try:
                    config = load_config(config_path)
                    assert isinstance(config, ExperimentConfig)

                    # Basic validation
                    assert config.experiment.name
                    assert config.model.name in ["linear", "mlp", "cnn"]
                    assert config.training.epochs > 0

                except Exception as e:
                    pytest.fail(f"Failed to load {config_file}: {e}")

    def test_config_validation_errors(self, temp_workspace):
        """Test that invalid configurations are properly rejected."""
        invalid_configs = [
            # Missing required fields
            {
                "data": {"dataset": "mnist"},
                "model": {"name": "linear"},
                # Missing training, evaluation, experiment, logging
            },
            # Invalid model configuration
            {
                "data": {"dataset": "mnist"},
                "model": {"name": "invalid_model"},
                "training": {"epochs": 1},
                "evaluation": {"metrics": ["accuracy"]},
                "experiment": {"name": "test"},
                "logging": {"log_every_n_steps": 1},
            },
            # Invalid parameter ranges
            {
                "data": {"dataset": "mnist", "batch_size": 0},  # Invalid batch size
                "model": {"name": "linear"},
                "training": {"epochs": 1},
                "evaluation": {"metrics": ["accuracy"]},
                "experiment": {"name": "test"},
                "logging": {"log_every_n_steps": 1},
            },
        ]

        for i, invalid_config in enumerate(invalid_configs):
            config_file = temp_workspace / f"invalid_{i}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(invalid_config, f)

            with pytest.raises(Exception):  # Should raise ValidationError or similar
                load_config(config_file)


@patch("src.experiment.data.datasets.MNIST")
class TestDataPipelineIntegration:
    """Test complete data pipeline integration."""

    def test_data_preparation_pipeline(self, mock_mnist, sample_config):
        """Test complete data preparation pipeline."""
        # Mock MNIST dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        # Make the mock dataset subscriptable
        def mock_getitem(self, idx):
            return (torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item())

        mock_dataset.__getitem__ = mock_getitem
        mock_mnist.return_value = mock_dataset

        config = load_config(sample_config)

        with patch("src.experiment.data.random_split") as mock_split:
            # Mock train/val split
            mock_train = Mock()
            mock_train.__len__ = Mock(return_value=80)
            mock_train.__getitem__ = lambda self, idx: mock_getitem(self, idx)
            mock_val = Mock()
            mock_val.__len__ = Mock(return_value=20)
            mock_val.__getitem__ = lambda self, idx: mock_getitem(self, idx)
            mock_split.return_value = (mock_train, mock_val)

            from src.experiment.data import prepare_data

            dataloaders, stats = prepare_data(
                config=config.data, augmentation_config=None, data_dir=Path("test_data")
            )

            # Verify pipeline outputs
            assert isinstance(dataloaders, dict)
            assert "train" in dataloaders
            assert "val" in dataloaders
            assert "test" in dataloaders

            assert isinstance(stats, dict)
            assert "train" in stats
            assert "test" in stats


class TestModelIntegration:
    """Test model creation and training integration."""

    def test_model_creation_from_configs(self):
        """Test creating models from different configurations."""
        from src.experiment.models import create_model
        from src.experiment.schemas import CNNModel, LinearModel, MLPModel

        configs = [
            LinearModel(input_size=784, num_classes=10),
            MLPModel(hidden_size=128, num_layers=2),
            CNNModel(channels=[16, 32], dropout=0.1),
        ]

        for config in configs:
            model = create_model(config)

            # Test model can handle forward pass
            if config.name == "cnn":
                x = torch.randn(2, 1, 28, 28)
            else:
                x = torch.randn(2, 784)

            output = model(x)
            assert output.shape == (2, 10)
            assert not torch.isnan(output).any()

    def test_optimizer_scheduler_integration(self):
        """Test optimizer and scheduler integration."""
        from src.experiment.models import (
            create_model,
            create_optimizer,
            create_scheduler,
        )
        from src.experiment.schemas import LinearModel, SchedulerConfig, TrainingConfig

        model = create_model(LinearModel())

        # Test different optimizers
        training_configs = [
            TrainingConfig(optimizer="adam", learning_rate=0.001),
            TrainingConfig(optimizer="sgd", learning_rate=0.01, momentum=0.9),
            TrainingConfig(
                optimizer="adam",
                learning_rate=0.001,
                scheduler=SchedulerConfig(name="reduce_lr_on_plateau", patience=3),
            ),
        ]

        for config in training_configs:
            optimizer = create_optimizer(model, config)
            scheduler = create_scheduler(optimizer, config)

            # Test training step
            x = torch.randn(4, 784)
            y = torch.randint(0, 10, (4,))

            output = model(x)
            loss = torch.nn.CrossEntropyLoss()(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                if hasattr(scheduler, "step"):
                    if "ReduceLR" in scheduler.__class__.__name__:
                        scheduler.step(loss.item())
                    else:
                        scheduler.step()

            # Should complete without error
            assert torch.isfinite(loss).all()


@patch("src.experiment.data.datasets.MNIST")
class TestExperimentIntegration:
    """Test complete experiment workflow integration."""

    def test_minimal_experiment_run(self, mock_mnist, sample_config, temp_workspace):
        """Test running a minimal experiment end-to-end."""
        import torch

        # Mock MNIST dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=16)  # Small dataset for test
        mock_dataset.__getitem__ = Mock(
            return_value=(torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item())
        )
        mock_mnist.return_value = mock_dataset

        config = load_config(sample_config)

        with patch("src.experiment.data.random_split") as mock_split:
            # Mock small train/val split
            def mock_getitem_small(idx):
                return (torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item())

            mock_train = Mock()
            mock_train.__len__ = Mock(return_value=12)
            mock_train.__getitem__ = lambda self, idx: mock_getitem_small(idx)
            mock_val = Mock()
            mock_val.__len__ = Mock(return_value=4)
            mock_val.__getitem__ = lambda self, idx: mock_getitem_small(idx)
            mock_split.return_value = (mock_train, mock_val)

            # Create experiment
            experiment = Experiment.from_config(config)

            # Mock some heavy operations for faster test
            with patch.object(experiment, "prepare_data") as mock_prepare_data:
                # Create minimal mock dataloaders
                mock_dataloader = Mock()
                mock_dataloader.__iter__ = Mock(
                    return_value=iter(
                        [(torch.randn(2, 1, 28, 28), torch.randint(0, 10, (2,)))]
                    )
                )
                mock_dataloader.__len__ = Mock(return_value=1)

                mock_dataloaders = {
                    "train": mock_dataloader,
                    "val": mock_dataloader,
                    "test": mock_dataloader,
                }
                mock_prepare_data.return_value = mock_dataloaders

                # Run experiment
                results = experiment.run()

                # Verify results structure
                assert isinstance(results, dict)
                assert "experiment_time" in results
                assert "results_path" in results
                assert results["experiment_time"] > 0

    def test_experiment_with_different_models(self, mock_mnist, temp_workspace):
        """Test experiments with different model types."""
        import torch

        # Mock MNIST dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=8)
        mock_dataset.__getitem__ = Mock(
            return_value=(torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item())
        )
        mock_mnist.return_value = mock_dataset

        model_configs = [
            {"name": "linear", "input_size": 784},
            {"name": "mlp", "hidden_size": 64, "num_layers": 1},
            {"name": "cnn", "channels": [8, 16]},
        ]

        for model_config in model_configs:
            config_dict = {
                "data": {"dataset": "mnist", "batch_size": 4, "download": False},
                "model": model_config,
                "training": {"epochs": 1, "learning_rate": 0.1},
                "evaluation": {"metrics": ["accuracy"]},
                "experiment": {
                    "name": f"test_{model_config['name']}",
                    "device": "cpu",
                    "cache_dir": str(temp_workspace / "cache"),
                    "save_model": False,
                },
                "logging": {"log_every_n_steps": 1},
            }

            config = ExperimentConfig(**config_dict)

            with patch("src.experiment.data.random_split") as mock_split:
                mock_split.return_value = (Mock(), Mock())

                experiment = Experiment.from_config(config)

                # Should create without error
                assert isinstance(experiment, Experiment)
                assert experiment.config.model.name == model_config["name"]


class TestCLIIntegration:
    """Test CLI integration with the experiment pipeline."""

    def test_cli_validate_command(self, sample_config):
        """Test CLI validate command."""
        runner = CliRunner()

        # Test successful validation
        result = runner.invoke(app, ["validate", str(sample_config)])

        # Should exit successfully
        assert result.exit_code == 0
        assert "Configuration is valid" in result.stdout

    def test_cli_validate_invalid_config(self, temp_workspace):
        """Test CLI validation with invalid config."""
        # Create invalid config
        invalid_config = {
            "data": {"dataset": "mnist"},
            "model": {"name": "invalid_model"},  # Missing required fields
        }

        config_file = temp_workspace / "invalid.yaml"
        with open(config_file, "w") as f:
            yaml.dump(invalid_config, f)

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(config_file)])

        # Should fail validation
        assert result.exit_code == 1
        assert "validation failed" in result.stdout

    def test_cli_list_configs(self, temp_workspace):
        """Test CLI list-configs command."""
        # Create some config files
        for i in range(3):
            config_dict = {
                "data": {"dataset": "mnist"},
                "model": {"name": "linear"},
                "training": {"epochs": 1},
                "evaluation": {"metrics": ["accuracy"]},
                "experiment": {"name": f"test_{i}"},
                "logging": {"log_every_n_steps": 1},
            }

            config_file = temp_workspace / "configs" / f"test_{i}.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_dict, f)

        runner = CliRunner()
        result = runner.invoke(
            app, ["list-configs", "--dir", str(temp_workspace / "configs")]
        )

        assert result.exit_code == 0
        assert "Available Configurations" in result.stdout
        assert "test_0.yaml" in result.stdout

    @patch("src.experiment.data.datasets.MNIST")
    def test_cli_dry_run(self, mock_mnist, sample_config):
        """Test CLI dry run functionality."""
        runner = CliRunner()

        result = runner.invoke(app, ["run", str(sample_config), "--dry-run"])

        # Should complete successfully without running experiment
        assert result.exit_code == 0
        assert "Configuration loaded" in result.stdout
        assert "Dry run completed" in result.stdout


@pytest.mark.slow
class TestPerformanceIntegration:
    """Performance integration tests (marked as slow)."""

    @patch("src.experiment.data.datasets.MNIST")
    def test_caching_performance(self, mock_mnist, sample_config):
        """Test that caching improves performance on repeated runs."""
        # This would test actual caching performance
        # Implementation depends on timing measurements
        pass

    def test_memory_usage_integration(self):
        """Test memory usage throughout complete pipeline."""
        # This would test memory usage patterns
        # Implementation depends on memory profiling
        pass


@pytest.mark.integration
class TestRealDataIntegration:
    """Integration tests with real data (when available)."""

    @pytest.mark.skipif(not Path("data/raw").exists(), reason="No real data available")
    def test_with_real_mnist(self):
        """Test with real MNIST data if available."""
        # This would test with actual MNIST data
        # Only runs if data directory exists
        pass

    def test_config_compatibility(self):
        """Test backward compatibility of configurations."""
        # This would test that old configs still work
        # Important for maintaining compatibility
        pass


class TestErrorHandling:
    """Test error handling in integration scenarios."""

    @patch("src.experiment.pipeline.prepare_data")
    def test_experiment_graceful_failure(
        self, mock_prepare_data, sample_config, temp_workspace
    ):
        """Test that experiments fail gracefully with proper error messages."""
        config = load_config(sample_config)

        # Force an error in data preparation
        mock_prepare_data.side_effect = Exception("Simulated data error")

        experiment = Experiment.from_config(config)

        with pytest.raises(Exception, match="Simulated data error"):
            experiment.run()

        # Should have saved error information
        # (This would be verified by checking the error artifact)

    def test_invalid_device_handling(self, temp_workspace):
        """Test handling of invalid device specifications."""
        config_dict = {
            "data": {"dataset": "mnist", "download": False},
            "model": {"name": "linear"},
            "training": {"epochs": 1},
            "evaluation": {"metrics": ["accuracy"]},
            "experiment": {
                "name": "device_test",
                "device": "invalid_device",  # Invalid device
                "cache_dir": str(temp_workspace / "cache"),
            },
            "logging": {"log_every_n_steps": 1},
        }

        # Should handle invalid device gracefully
        # (Implementation would depend on how device validation is handled)
