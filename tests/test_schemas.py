"""Tests for Pydantic schema validation.

This module tests all configuration schemas to ensure proper validation,
error handling, and default behavior.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.experiment.schemas import (
    CNNModel,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ExperimentMetadata,
    LinearModel,
    LoggingConfig,
    MLPModel,
    TrainingConfig,
    load_config,
    save_config,
)


class TestDataConfig:
    """Test DataConfig validation."""

    def test_default_values(self):
        """Test that default values are applied correctly."""
        config = DataConfig()

        assert config.dataset == "mnist"
        assert config.batch_size == 64
        assert config.validation_split == 0.1
        assert config.num_workers == 0
        assert config.shuffle == True
        assert config.pin_memory == False
        assert config.download == True

    def test_valid_batch_size(self):
        """Test valid batch size values."""
        config = DataConfig(batch_size=32)
        assert config.batch_size == 32

        config = DataConfig(batch_size=256)
        assert config.batch_size == 256

    def test_invalid_batch_size(self):
        """Test invalid batch size values."""
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            DataConfig(batch_size=0)

        with pytest.raises(ValidationError, match="batch_size must be positive"):
            DataConfig(batch_size=-1)

        with pytest.raises(ValidationError, match="batch_size too large"):
            DataConfig(batch_size=2048)

    def test_validation_split_bounds(self):
        """Test validation split boundary values."""
        # Valid values
        DataConfig(validation_split=0.0)
        DataConfig(validation_split=0.2)
        DataConfig(validation_split=0.5)

        # Invalid values
        with pytest.raises(ValidationError):
            DataConfig(validation_split=-0.1)

        with pytest.raises(ValidationError):
            DataConfig(validation_split=0.6)


class TestModelConfigs:
    """Test model configuration schemas."""

    def test_linear_model_defaults(self):
        """Test LinearModel default values."""
        config = LinearModel()

        assert config.name == "linear"
        assert config.input_size == 784
        assert config.num_classes == 10
        assert config.bias == True

    def test_linear_model_input_size_validation(self):
        """Test LinearModel input size validation."""
        # Valid input size
        LinearModel(input_size=784)

        # Invalid input size
        with pytest.raises(ValidationError, match="input_size must be 784 for MNIST"):
            LinearModel(input_size=1000)

    def test_mlp_model_defaults(self):
        """Test MLPModel default values."""
        config = MLPModel()

        assert config.name == "mlp"
        assert config.input_size == 784
        assert config.hidden_size == 128
        assert config.num_layers == 2
        assert config.num_classes == 10
        assert config.dropout == 0.1
        assert config.activation == "relu"

    def test_mlp_model_hidden_size_validation(self):
        """Test MLPModel hidden size validation."""
        # Valid sizes
        MLPModel(hidden_size=64)
        MLPModel(hidden_size=512)

        # Too small
        with pytest.raises(ValidationError, match="hidden_size too small"):
            MLPModel(hidden_size=4)

        # Too large (warning, but allowed)
        with pytest.raises(ValidationError, match="hidden_size very large"):
            MLPModel(hidden_size=4096)

    def test_mlp_model_layer_bounds(self):
        """Test MLPModel layer count bounds."""
        MLPModel(num_layers=1)  # Minimum
        MLPModel(num_layers=5)  # Middle
        MLPModel(num_layers=10)  # Maximum

        # Invalid bounds
        with pytest.raises(ValidationError):
            MLPModel(num_layers=0)

        with pytest.raises(ValidationError):
            MLPModel(num_layers=15)

    def test_cnn_model_defaults(self):
        """Test CNNModel default values."""
        config = CNNModel()

        assert config.name == "cnn"
        assert config.input_channels == 1
        assert config.num_classes == 10
        assert config.channels == [32, 64]
        assert config.kernel_size == 3
        assert config.dropout == 0.2
        assert config.batch_norm == True

    def test_cnn_model_channels_validation(self):
        """Test CNNModel channels validation."""
        # Valid channels
        CNNModel(channels=[16, 32, 64])
        CNNModel(channels=[32])

        # Invalid channels - negative
        with pytest.raises(ValidationError, match="All channel sizes must be positive"):
            CNNModel(channels=[32, -16])

        # Invalid channels - too large
        with pytest.raises(
            ValidationError, match="Channel sizes >512 may be too large"
        ):
            CNNModel(channels=[1024])

    def test_kernel_size_bounds(self):
        """Test CNNModel kernel size bounds."""
        CNNModel(kernel_size=3)  # Valid
        CNNModel(kernel_size=5)  # Valid
        CNNModel(kernel_size=7)  # Valid

        with pytest.raises(ValidationError):
            CNNModel(kernel_size=1)  # Too small

        with pytest.raises(ValidationError):
            CNNModel(kernel_size=9)  # Too large


class TestTrainingConfig:
    """Test training configuration validation."""

    def test_defaults(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig()

        assert config.epochs == 10
        assert config.learning_rate == 0.001
        assert config.optimizer == "adam"
        assert config.weight_decay == 0.0
        assert config.momentum == 0.9
        assert config.scheduler is None
        assert config.early_stopping is None

    def test_epoch_bounds(self):
        """Test epoch validation."""
        TrainingConfig(epochs=1)  # Minimum
        TrainingConfig(epochs=100)  # Normal
        TrainingConfig(epochs=1000)  # Maximum

        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)

        with pytest.raises(ValidationError):
            TrainingConfig(epochs=1001)

    def test_learning_rate_validation(self):
        """Test learning rate validation."""
        TrainingConfig(learning_rate=0.0001)  # Valid
        TrainingConfig(learning_rate=0.01)  # Valid

        # Too high
        with pytest.raises(ValidationError, match="learning_rate >0.1 is very high"):
            TrainingConfig(learning_rate=0.5)

        # Invalid bounds
        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=0.0)

        with pytest.raises(ValidationError):
            TrainingConfig(learning_rate=1.5)


class TestExperimentConfig:
    """Test complete experiment configuration."""

    def test_minimal_config(self):
        """Test minimal valid configuration."""
        config_dict = {
            "data": {"dataset": "mnist"},
            "model": {"name": "linear"},
            "training": {"epochs": 5},
            "evaluation": {"metrics": ["accuracy"]},
            "experiment": {"name": "test_experiment"},
            "logging": {"log_every_n_steps": 10},
        }

        config = ExperimentConfig(**config_dict)
        assert config.experiment.name == "test_experiment"
        assert config.model.name == "linear"
        assert config.training.epochs == 5

    def test_full_config_with_all_options(self):
        """Test configuration with all optional fields."""
        config_dict = {
            "data": {"dataset": "mnist", "batch_size": 128, "validation_split": 0.2},
            "model": {
                "name": "mlp",
                "hidden_size": 256,
                "num_layers": 3,
                "dropout": 0.3,
            },
            "training": {
                "epochs": 50,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "scheduler": {
                    "name": "reduce_lr_on_plateau",
                    "factor": 0.5,
                    "patience": 5,
                },
                "early_stopping": {"patience": 10, "min_delta": 0.001},
            },
            "evaluation": {
                "metrics": ["accuracy", "f1_macro"],
                "save_predictions": True,
            },
            "experiment": {
                "name": "full_test",
                "description": "Full configuration test",
                "random_seed": 123,
                "device": "cpu",
            },
            "logging": {"log_every_n_steps": 50, "log_gradients": True},
            "data_augmentation": {"enabled": True, "rotation_degrees": 15.0},
        }

        config = ExperimentConfig(**config_dict)
        assert config.experiment.name == "full_test"
        assert config.model.hidden_size == 256
        assert config.training.scheduler.patience == 5
        assert config.data_augmentation.enabled == True

    def test_model_discriminator(self):
        """Test model discriminated union."""
        # Test different model types
        for model_config in [
            {"name": "linear", "input_size": 784},
            {"name": "mlp", "hidden_size": 128},
            {"name": "cnn", "channels": [32, 64]},
        ]:
            config_dict = {
                "data": {"dataset": "mnist"},
                "model": model_config,
                "training": {"epochs": 1},
                "evaluation": {"metrics": ["accuracy"]},
                "experiment": {"name": "test"},
                "logging": {"log_every_n_steps": 10},
            }

            config = ExperimentConfig(**config_dict)
            assert config.model.name == model_config["name"]

    def test_cross_field_validation(self):
        """Test cross-field validation rules."""
        # Test inconsistent configuration
        config_dict = {
            "data": {"dataset": "mnist"},
            "model": {"name": "linear", "input_size": 1000},  # Wrong size
            "training": {"epochs": 1},
            "evaluation": {"metrics": ["accuracy"]},
            "experiment": {"name": "test"},
            "logging": {"log_every_n_steps": 10},
        }

        with pytest.raises(ValidationError):
            ExperimentConfig(**config_dict)


class TestConfigIO:
    """Test configuration loading and saving."""

    def test_load_config_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_dict = {
            "data": {"dataset": "mnist", "batch_size": 32},
            "model": {"name": "linear"},
            "training": {"epochs": 5},
            "evaluation": {"metrics": ["accuracy"]},
            "experiment": {"name": "yaml_test"},
            "logging": {"log_every_n_steps": 10},
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        loaded_config = load_config(config_file)
        assert loaded_config.experiment.name == "yaml_test"
        assert loaded_config.data.batch_size == 32

    def test_load_nonexistent_config(self):
        """Test loading non-existent configuration file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent.yaml"))

    def test_save_config(self, tmp_path):
        """Test saving configuration to YAML file."""
        config_dict = {
            "data": {"dataset": "mnist"},
            "model": {"name": "linear"},
            "training": {"epochs": 5},
            "evaluation": {"metrics": ["accuracy"]},
            "experiment": {"name": "save_test"},
            "logging": {"log_every_n_steps": 10},
        }

        config = ExperimentConfig(**config_dict)
        output_file = tmp_path / "saved_config.yaml"

        save_config(config, output_file)

        # Verify file exists and can be loaded
        assert output_file.exists()
        loaded_config = load_config(output_file)
        assert loaded_config.experiment.name == "save_test"


class TestExperimentMetadata:
    """Test experiment metadata validation."""

    def test_valid_experiment_names(self):
        """Test valid experiment name formats."""
        valid_names = [
            "simple_name",
            "name-with-hyphens",
            "name123",
            "MixedCase",
            "under_score_123",
        ]

        for name in valid_names:
            config = ExperimentMetadata(name=name)
            assert config.name == name

    def test_invalid_experiment_names(self):
        """Test invalid experiment name formats."""
        invalid_names = [
            "name with spaces",
            "name@with@symbols",
            "name.with.dots",
            "name/with/slashes",
        ]

        for name in invalid_names:
            with pytest.raises(
                ValidationError, match="name must contain only alphanumeric"
            ):
                ExperimentMetadata(name=name)

    def test_name_length_limits(self):
        """Test experiment name length validation."""
        # Valid lengths
        ExperimentMetadata(name="a")  # Minimum length
        ExperimentMetadata(name="a" * 100)  # Maximum length

        # Invalid lengths
        with pytest.raises(ValidationError):
            ExperimentMetadata(name="")  # Too short

        with pytest.raises(ValidationError):
            ExperimentMetadata(name="a" * 101)  # Too long


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for complete configuration workflows."""

    def test_all_example_configs_valid(self):
        """Test that all example configurations are valid."""
        config_dir = Path("configs")
        if not config_dir.exists():
            pytest.skip("No configs directory found")

        yaml_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))

        for config_file in yaml_files:
            try:
                config = load_config(config_file)
                assert isinstance(config, ExperimentConfig)
                # Basic validation that required fields exist
                assert config.experiment.name
                assert config.model.name in ["linear", "mlp", "cnn"]
            except Exception as e:
                pytest.fail(f"Configuration {config_file} is invalid: {e}")

    def test_config_roundtrip(self, tmp_path):
        """Test configuration save/load roundtrip."""
        original_dict = {
            "data": {"dataset": "mnist", "batch_size": 64},
            "model": {"name": "mlp", "hidden_size": 256},
            "training": {"epochs": 10, "learning_rate": 0.001},
            "evaluation": {"metrics": ["accuracy", "f1_macro"]},
            "experiment": {"name": "roundtrip_test", "random_seed": 42},
            "logging": {"log_every_n_steps": 20},
        }

        # Create config and save
        original_config = ExperimentConfig(**original_dict)
        config_file = tmp_path / "roundtrip.yaml"
        save_config(original_config, config_file)

        # Load and compare
        loaded_config = load_config(config_file)

        # Key fields should match
        assert loaded_config.experiment.name == original_config.experiment.name
        assert loaded_config.model.name == original_config.model.name
        assert loaded_config.model.hidden_size == original_config.model.hidden_size
        assert loaded_config.training.epochs == original_config.training.epochs
        assert (
            loaded_config.experiment.random_seed
            == original_config.experiment.random_seed
        )
