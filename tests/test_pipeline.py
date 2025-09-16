"""Tests for the experiment pipeline.

This module tests the main Experiment class and its orchestration
of the complete ML pipeline using ExCa for caching and reproducibility.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from src.experiment.infra import ExperimentInfra
from src.experiment.pipeline import EarlyStopping, Experiment
from src.experiment.schemas import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ExperimentMetadata,
    LinearModel,
    LoggingConfig,
    TrainingConfig,
)


class TestEarlyStopping:
    """Test EarlyStopping utility class."""

    def test_early_stopping_initialization(self):
        """Test EarlyStopping initialization."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01)

        assert early_stop.patience == 3
        assert early_stop.min_delta == 0.01
        assert early_stop.counter == 0
        assert early_stop.best_score is None
        assert early_stop.early_stop is False

    def test_early_stopping_improvement(self):
        """Test early stopping with continuous improvement."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01)

        # Continuous improvement should not trigger early stopping
        assert early_stop(0.5) is False
        assert early_stop(0.6) is False
        assert early_stop(0.7) is False

        assert early_stop.counter == 0
        assert early_stop.best_score == 0.7

    def test_early_stopping_no_improvement(self):
        """Test early stopping with no improvement."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01)

        # Initial improvement
        assert early_stop(0.5) is False

        # No significant improvement (within min_delta)
        assert early_stop(0.505) is False  # counter = 1
        assert early_stop(0.504) is False  # counter = 2
        assert early_stop(0.503) is True  # counter = 3, triggers early stopping

        assert early_stop.early_stop is True

    def test_early_stopping_reset_counter(self):
        """Test that counter resets on improvement."""
        early_stop = EarlyStopping(patience=3, min_delta=0.01)

        assert early_stop(0.5) is False
        assert early_stop(0.505) is False  # No improvement, counter = 1
        assert early_stop(0.52) is False  # Improvement, counter should reset
        assert early_stop(0.52) is False  # No improvement, counter = 1

        assert early_stop.counter == 1  # Should have reset after improvement


class TestExperiment:
    """Test main Experiment class."""

    def create_minimal_config(self) -> ExperimentConfig:
        """Create minimal valid experiment configuration for testing."""
        config_dict = {
            "data": {"dataset": "mnist", "batch_size": 16, "download": False},
            "model": {"name": "linear", "input_size": 784, "num_classes": 10},
            "training": {"epochs": 2, "learning_rate": 0.01},
            "evaluation": {"metrics": ["accuracy"]},
            "experiment": {
                "name": "test_experiment",
                "random_seed": 42,
                "device": "cpu",
                "cache_dir": "test_cache",
            },
            "logging": {"log_every_n_steps": 1},
        }
        return ExperimentConfig(**config_dict)

    def test_experiment_initialization(self):
        """Test experiment initialization."""
        config = self.create_minimal_config()

        with patch("src.experiment.pipeline.create_infra") as mock_create_infra:
            mock_infra = Mock()
            mock_create_infra.return_value = mock_infra

            experiment = Experiment(config)

            assert experiment.config == config
            assert experiment.infra == mock_infra
            assert experiment.device.type == "cpu"
            assert experiment.model is None
            assert experiment.optimizer is None
            assert experiment.scheduler is None

    def test_device_setup_auto(self):
        """Test automatic device selection."""
        config = self.create_minimal_config()
        config.experiment.device = "auto"

        with patch("src.experiment.pipeline.create_infra"):
            with patch("torch.cuda.is_available", return_value=False):
                with patch("torch.backends.mps.is_available", return_value=False):
                    experiment = Experiment(config)
                    assert experiment.device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    def test_device_setup_cuda_available(self, mock_cuda):
        """Test device selection when CUDA is available."""
        config = self.create_minimal_config()
        config.experiment.device = "auto"

        with patch("src.experiment.pipeline.create_infra"):
            experiment = Experiment(config)
            assert experiment.device.type == "cuda"

    def test_random_seed_setting(self):
        """Test that random seeds are set correctly."""
        config = self.create_minimal_config()
        config.experiment.random_seed = 123

        with patch("src.experiment.pipeline.create_infra"):
            with patch("torch.manual_seed") as mock_torch_seed:
                with patch("numpy.random.seed") as mock_np_seed:
                    Experiment(config)

                    mock_torch_seed.assert_called_with(123)
                    mock_np_seed.assert_called_with(123)

    def test_get_expected_input_shape(self):
        """Test expected input shape for different models."""
        from src.experiment.schemas import CNNModel, MLPModel

        with patch("src.experiment.pipeline.create_infra"):
            # Linear model
            config = self.create_minimal_config()
            config.model = LinearModel()
            experiment = Experiment(config)
            assert experiment._get_expected_input_shape() == (784,)

            # MLP model
            config = self.create_minimal_config()
            config.model = MLPModel(hidden_size=128)
            experiment = Experiment(config)
            assert experiment._get_expected_input_shape() == (784,)

            # CNN model
            config = self.create_minimal_config()
            config.model = CNNModel(channels=[32, 64])
            experiment = Experiment(config)
            assert experiment._get_expected_input_shape() == (1, 28, 28)

    @patch("src.experiment.pipeline.create_model")
    def test_create_model(self, mock_create_model):
        """Test model creation."""
        mock_model = Mock()
        mock_model.to.return_value = mock_model
        mock_create_model.return_value = mock_model

        config = self.create_minimal_config()

        with patch("src.experiment.pipeline.create_infra"):
            with patch(
                "src.experiment.pipeline.get_model_summary",
                return_value="Model summary",
            ):
                experiment = Experiment(config)
                result = experiment.create_model()

                assert result == mock_model
                assert experiment.model == mock_model
                mock_model.to.assert_called_with(experiment.device)

    def test_create_optimizer_and_scheduler(self):
        """Test optimizer and scheduler creation."""
        config = self.create_minimal_config()

        with patch("src.experiment.pipeline.create_infra"):
            with patch("src.experiment.pipeline.create_optimizer") as mock_create_opt:
                with patch(
                    "src.experiment.pipeline.create_scheduler"
                ) as mock_create_sched:
                    mock_optimizer = Mock()
                    mock_scheduler = Mock()
                    mock_create_opt.return_value = mock_optimizer
                    mock_create_sched.return_value = mock_scheduler

                    experiment = Experiment(config)
                    experiment.model = Mock()  # Set model first
                    experiment.create_optimizer_and_scheduler()

                    assert experiment.optimizer == mock_optimizer
                    assert experiment.scheduler == mock_scheduler

    def test_create_optimizer_without_model_raises_error(self):
        """Test that creating optimizer without model raises error."""
        config = self.create_minimal_config()

        with patch("src.experiment.pipeline.create_infra"):
            experiment = Experiment(config)

            with pytest.raises(
                ValueError, match="Model must be created before optimizer"
            ):
                experiment.create_optimizer_and_scheduler()


class TestTrainingLoop:
    """Test training loop functionality."""

    def create_mock_dataloader(self, num_batches=2, batch_size=4):
        """Create a mock dataloader for testing."""
        batches = []
        for _ in range(num_batches):
            data = torch.randn(batch_size, 1, 28, 28)
            target = torch.randint(0, 10, (batch_size,))
            batches.append((data, target))

        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(batches))
        mock_dataloader.__len__ = Mock(return_value=num_batches)
        return mock_dataloader

    def test_train_epoch(self):
        """Test training for one epoch."""
        config = ExperimentConfig(
            **{
                "data": {"dataset": "mnist", "download": False},
                "model": {"name": "linear"},
                "training": {"epochs": 1},
                "evaluation": {"metrics": ["accuracy"]},
                "experiment": {"name": "test", "device": "cpu"},
                "logging": {"log_every_n_steps": 1},
            }
        )

        with patch("src.experiment.pipeline.create_infra"):
            experiment = Experiment(config)

            # Create mock model and optimizer
            mock_model = Mock()
            mock_model.train = Mock()
            mock_output = torch.randn(4, 10)
            mock_model.return_value = mock_output

            mock_optimizer = Mock()

            experiment.model = mock_model
            experiment.optimizer = mock_optimizer

            # Create mock dataloader
            dataloader = self.create_mock_dataloader()

            with patch("torch.nn.CrossEntropyLoss") as mock_loss:
                mock_loss_fn = Mock()
                mock_loss_value = Mock()
                mock_loss_value.item.return_value = 0.5
                mock_loss_value.backward = Mock()
                mock_loss_fn.return_value = mock_loss_value
                mock_loss.return_value = mock_loss_fn

                metrics = experiment.train_epoch(dataloader)

                # Verify training mode was set
                mock_model.train.assert_called_once()

                # Verify optimizer steps
                assert mock_optimizer.zero_grad.call_count == 2  # 2 batches
                assert mock_optimizer.step.call_count == 2

                # Verify metrics returned
                assert "loss" in metrics
                assert "accuracy" in metrics
                assert isinstance(metrics["loss"], float)
                assert isinstance(metrics["accuracy"], float)

    @patch("torch.no_grad")
    def test_evaluate(self, mock_no_grad):
        """Test model evaluation."""
        config = ExperimentConfig(
            **{
                "data": {"dataset": "mnist", "download": False},
                "model": {"name": "linear"},
                "training": {"epochs": 1},
                "evaluation": {"metrics": ["accuracy", "f1_macro"]},
                "experiment": {"name": "test", "device": "cpu"},
                "logging": {"log_every_n_steps": 1},
            }
        )

        with patch("src.experiment.pipeline.create_infra"):
            experiment = Experiment(config)

            # Mock model
            mock_model = Mock()
            mock_model.eval = Mock()
            mock_output = torch.randn(4, 10)
            mock_model.return_value = mock_output
            experiment.model = mock_model

            # Create mock dataloader
            dataloader = self.create_mock_dataloader()

            with patch("torch.nn.CrossEntropyLoss") as mock_loss:
                with patch("src.experiment.pipeline.accuracy_score", return_value=0.75):
                    with patch("src.experiment.pipeline.f1_score", return_value=0.7):
                        mock_loss_fn = Mock()
                        mock_loss_value = Mock()
                        mock_loss_value.item.return_value = 0.3
                        mock_loss_fn.return_value = mock_loss_value
                        mock_loss.return_value = mock_loss_fn

                        metrics = experiment.evaluate(dataloader, "test")

                        # Verify eval mode was set
                        mock_model.eval.assert_called_once()

                        # Verify metrics
                        assert "loss" in metrics
                        assert "accuracy" in metrics
                        assert "f1_macro" in metrics
                        assert metrics["accuracy"] == 0.75
                        assert metrics["f1_macro"] == 0.7


@patch("src.experiment.pipeline.prepare_data")
@patch("src.experiment.pipeline.create_model")
@patch("src.experiment.pipeline.create_optimizer")
@patch("src.experiment.pipeline.create_scheduler")
class TestExperimentRun:
    """Test complete experiment run."""

    def create_test_config(self):
        """Create test configuration."""
        return ExperimentConfig(
            **{
                "data": {"dataset": "mnist", "batch_size": 4, "download": False},
                "model": {"name": "linear"},
                "training": {"epochs": 1},
                "evaluation": {"metrics": ["accuracy"]},
                "experiment": {
                    "name": "test_run",
                    "device": "cpu",
                    "save_model": False,
                },
                "logging": {"log_every_n_steps": 1},
            }
        )

    def test_experiment_run_success(
        self, mock_scheduler, mock_optimizer_fn, mock_model_fn, mock_prepare_data
    ):
        """Test successful experiment run."""
        config = self.create_test_config()

        # Setup mocks
        mock_dataloaders = {"train": Mock(), "test": Mock()}
        mock_stats = {"train": {"num_samples": 100}}
        mock_prepare_data.return_value = (mock_dataloaders, mock_stats)

        mock_model = Mock()
        mock_model.to.return_value = mock_model
        # Mock parameters() method for model summary
        mock_param = Mock()
        mock_param.numel.return_value = 1000
        mock_param.requires_grad = True
        mock_model.parameters.return_value = [mock_param]
        # Mock model info method
        mock_model.get_model_info.return_value = {"model_class": "MockModel"}
        mock_model_fn.return_value = mock_model

        mock_optimizer = Mock()
        mock_optimizer_fn.return_value = mock_optimizer
        mock_scheduler.return_value = None

        with patch("src.experiment.pipeline.create_infra") as mock_create_infra:
            # Setup infra mock
            mock_infra = Mock()
            mock_infra.save_artifact.return_value = Path("test/result.pkl")
            mock_infra.save_metrics.return_value = Path("test/metrics.json")
            mock_infra.record_run.return_value = Path("test/run.json")

            # Make cached_stage not cache but pass through, preserving kwargs
            def mock_cached_stage(stage_name):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        kwargs.pop("_cache_context", None)
                        return func(*args, **kwargs)

                    return wrapper

                return decorator

            mock_infra.cached_stage = mock_cached_stage
            mock_create_infra.return_value = mock_infra

            experiment = Experiment(config)

            # Mock training and evaluation
            with patch.object(experiment, "train") as mock_train:
                with patch.object(experiment, "evaluate") as mock_evaluate:
                    with patch(
                        "src.experiment.pipeline.validate_data_shape", return_value=True
                    ):
                        mock_train.return_value = {
                            "train_loss": [0.5],
                            "train_accuracy": [0.8],
                        }
                        mock_evaluate.return_value = {"accuracy": 0.85, "loss": 0.3}

                        results = experiment.run()

                    # Verify pipeline was executed
                    mock_prepare_data.assert_called_once()
                    mock_model_fn.assert_called_once()
                    mock_train.assert_called_once()
                    mock_evaluate.assert_called()

                    # Verify results structure
                    assert "experiment_time" in results
                    assert "results_path" in results
                    assert "run_record" in results
                    assert isinstance(results["experiment_time"], float)

    def test_experiment_run_with_error(
        self, mock_scheduler, mock_optimizer_fn, mock_model_fn, mock_prepare_data
    ):
        """Test experiment run with error handling."""
        config = self.create_test_config()

        # Make prepare_data raise an error
        mock_prepare_data.side_effect = Exception("Data loading failed")

        with patch("src.experiment.pipeline.create_infra") as mock_create_infra:
            mock_infra = Mock()
            mock_infra.save_artifact.return_value = Path("test/error.pkl")

            # Make cached_stage not cache but pass through, preserving kwargs
            def mock_cached_stage(stage_name):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        kwargs.pop("_cache_context", None)
                        return func(*args, **kwargs)

                    return wrapper

                return decorator

            mock_infra.cached_stage = mock_cached_stage
            mock_create_infra.return_value = mock_infra

            experiment = Experiment(config)

            with pytest.raises(Exception, match="Data loading failed"):
                experiment.run()

            # Verify error was saved
            mock_infra.save_artifact.assert_called()
            # Check that error info was saved
            call_args = mock_infra.save_artifact.call_args
            assert call_args[0][1] == "experiment_error"  # artifact name
            assert "error" in call_args[0][0]  # error info in artifact


@pytest.mark.integration
class TestExperimentIntegration:
    """Integration tests for complete experiment workflows."""

    def test_from_config_factory(self):
        """Test creating experiment from configuration."""
        config_dict = {
            "data": {"dataset": "mnist", "download": False},
            "model": {"name": "linear"},
            "training": {"epochs": 1},
            "evaluation": {"metrics": ["accuracy"]},
            "experiment": {"name": "factory_test", "device": "cpu"},
            "logging": {"log_every_n_steps": 1},
        }
        config = ExperimentConfig(**config_dict)

        with patch("src.experiment.pipeline.create_infra"):
            experiment = Experiment.from_config(config)

            assert isinstance(experiment, Experiment)
            assert experiment.config == config

    def test_cached_data_preparation(self):
        """Test that data preparation uses caching."""
        config = ExperimentConfig(
            **{
                "data": {"dataset": "mnist", "download": False},
                "model": {"name": "linear"},
                "training": {"epochs": 1},
                "evaluation": {"metrics": ["accuracy"]},
                "experiment": {"name": "cache_test", "device": "cpu"},
                "logging": {"log_every_n_steps": 1},
            }
        )

        with patch("src.experiment.pipeline.create_infra") as mock_create_infra:
            mock_infra = Mock()

            # Mock the cached_stage decorator, preserving args/kwargs
            def mock_cached_stage(stage_name):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        kwargs.pop("_cache_context", None)
                        return func(*args, **kwargs)

                    return wrapper

                return decorator

            mock_infra.cached_stage = mock_cached_stage
            mock_create_infra.return_value = mock_infra

            experiment = Experiment(config)

            with patch("src.experiment.pipeline.prepare_data") as mock_prepare_data:
                with patch(
                    "src.experiment.pipeline.validate_data_shape", return_value=True
                ):
                    mock_prepare_data.return_value = (
                        {"train": Mock()},
                        {"stats": "test"},
                    )

                    dataloaders = experiment.prepare_data()

                    # Verify prepare_data was called
                    mock_prepare_data.assert_called_once()
                    # Verify infra.save_artifact was called for stats
                    mock_infra.save_artifact.assert_called_with(
                        {"stats": "test"}, "dataset_statistics"
                    )

    def test_experiment_with_different_model_types(self):
        """Test experiment creation with different model configurations."""
        base_config = {
            "data": {"dataset": "mnist", "download": False},
            "training": {"epochs": 1},
            "evaluation": {"metrics": ["accuracy"]},
            "experiment": {"name": "model_test", "device": "cpu"},
            "logging": {"log_every_n_steps": 1},
        }

        model_configs = [
            {"name": "linear", "input_size": 784},
            {"name": "mlp", "hidden_size": 128, "num_layers": 2},
            {"name": "cnn", "channels": [32, 64]},
        ]

        with patch("src.experiment.pipeline.create_infra"):
            for model_config in model_configs:
                config_dict = {**base_config, "model": model_config}
                config = ExperimentConfig(**config_dict)

                experiment = Experiment(config)

                # Should create without error
                assert isinstance(experiment, Experiment)
                assert experiment.config.model.name == model_config["name"]


@pytest.mark.slow
class TestExperimentPerformance:
    """Performance tests for experiments (marked as slow)."""

    def test_experiment_memory_usage(self):
        """Test experiment memory footprint."""
        # This would test memory usage during experiment runs
        # Implementation depends on memory profiling tools
        pass

    def test_experiment_caching_performance(self):
        """Test that caching provides performance benefits."""
        # This would test that repeated runs are faster due to caching
        # Implementation depends on timing measurements
        pass
