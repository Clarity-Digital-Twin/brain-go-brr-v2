"""Pydantic schemas for experiment configuration validation.

This module defines strongly-typed, validated configuration objects for all aspects
of the machine learning pipeline. Using discriminated unions and field validators
ensures that invalid configurations are caught early with helpful error messages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pydantic as pd
from pydantic import Field

# ---- Data Configuration ----


class DataConfig(pd.BaseModel):
    """Configuration for dataset loading and preprocessing.


    Validates MNIST-specific parameters and ensures consistent data handling
    across experiments.
    """

    dataset: Literal["mnist"] = "mnist"
    batch_size: int = Field(64, description="Batch size for training and evaluation")
    validation_split: float = Field(
        0.1, ge=0.0, le=0.5, description="Fraction of training data for validation"
    )
    num_workers: int = Field(0, ge=0, description="Number of data loader workers")
    shuffle: bool = Field(True, description="Whether to shuffle training data")
    pin_memory: bool = Field(False, description="Whether to pin memory in data loader")
    download: bool = Field(True, description="Whether to download dataset if not found")

    @pd.field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("batch_size must be positive")
        if v > 1024:
            raise ValueError(
                "batch_size too large (>1024), consider reducing for memory"
            )
        return v


# ---- Model Configurations (Discriminated Union) ----


class LinearModel(pd.BaseModel):
    """Linear/Logistic regression model configuration."""

    name: Literal["linear"] = "linear"
    input_size: int = Field(
        784, gt=0, description="Input feature dimension (28x28=784 for MNIST)"
    )
    num_classes: int = Field(10, gt=0, description="Number of output classes")
    bias: bool = Field(True, description="Whether to include bias term")

    @pd.field_validator("input_size")
    @classmethod
    def validate_input_size(cls, v: int) -> int:
        if v != 784:
            raise ValueError("input_size must be 784 for MNIST (28x28 flattened)")
        return v


class MLPModel(pd.BaseModel):
    """Multi-layer perceptron model configuration."""

    name: Literal["mlp"] = "mlp"
    input_size: int = Field(784, gt=0, description="Input feature dimension")
    hidden_size: int = Field(128, gt=0, description="Hidden layer size")
    num_layers: int = Field(2, ge=1, le=10, description="Number of hidden layers")
    num_classes: int = Field(10, gt=0, description="Number of output classes")
    dropout: float = Field(0.1, ge=0.0, le=0.9, description="Dropout probability")
    activation: Literal["relu", "tanh", "sigmoid", "gelu"] = Field(
        "relu", description="Activation function"
    )

    @pd.field_validator("hidden_size")
    @classmethod
    def validate_hidden_size(cls, v: int) -> int:
        if v < 8:
            raise ValueError("hidden_size too small (<8)")
        if v > 2048:
            raise ValueError("hidden_size very large (>2048), consider reducing")
        return v


class CNNModel(pd.BaseModel):
    """Convolutional neural network model configuration."""

    name: Literal["cnn"] = "cnn"
    input_channels: int = Field(
        1, gt=0, description="Number of input channels (1 for grayscale MNIST)"
    )
    num_classes: int = Field(10, gt=0, description="Number of output classes")
    channels: List[int] = Field(
        [32, 64],
        min_length=1,
        max_length=5,
        description="Channel sizes for conv layers",
    )
    kernel_size: int = Field(3, ge=3, le=7, description="Convolutional kernel size")
    dropout: float = Field(0.2, ge=0.0, le=0.9, description="Dropout probability")
    batch_norm: bool = Field(True, description="Whether to use batch normalization")

    @pd.field_validator("channels")
    @classmethod
    def validate_channels(cls, v: List[int]) -> List[int]:
        if not all(c > 0 for c in v):
            raise ValueError("All channel sizes must be positive")
        if any(c > 512 for c in v):
            raise ValueError("Channel sizes >512 may be too large")
        return v


# Model discriminated union
ModelConfig = Union[LinearModel, MLPModel, CNNModel]


# ---- Training Configuration ----


class SchedulerConfig(pd.BaseModel):
    """Learning rate scheduler configuration."""

    name: Literal["reduce_lr_on_plateau", "step", "cosine"] = "reduce_lr_on_plateau"
    factor: float = Field(
        0.5, gt=0.0, lt=1.0, description="Factor by which to reduce LR"
    )
    patience: int = Field(
        5, gt=0, description="Epochs with no improvement after which LR is reduced"
    )
    min_lr: float = Field(1e-6, gt=0.0, description="Minimum learning rate")


class EarlyStoppingConfig(pd.BaseModel):
    """Early stopping configuration."""

    patience: int = Field(
        5, gt=0, description="Epochs with no improvement after which training stops"
    )
    min_delta: float = Field(
        0.001, ge=0.0, description="Minimum change to qualify as improvement"
    )


class TrainingConfig(pd.BaseModel):
    """Training loop configuration."""

    epochs: int = Field(10, ge=1, le=1000, description="Number of training epochs")
    learning_rate: float = Field(
        0.001, gt=0.0, le=1.0, description="Initial learning rate"
    )
    optimizer: Literal["adam", "sgd", "adamw", "rmsprop"] = Field(
        "adam", description="Optimizer type"
    )
    weight_decay: float = Field(
        0.0, ge=0.0, le=0.1, description="L2 regularization strength"
    )
    momentum: float = Field(
        0.9, ge=0.0, lt=1.0, description="Momentum for SGD optimizer"
    )
    scheduler: Optional[SchedulerConfig] = Field(
        None, description="Learning rate scheduler"
    )
    early_stopping: Optional[EarlyStoppingConfig] = Field(
        None, description="Early stopping configuration"
    )

    @pd.field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, v: float) -> float:
        if v > 0.1:
            raise ValueError(
                "learning_rate >0.1 is very high and may cause instability"
            )
        return v


# ---- Evaluation Configuration ----


class EvaluationConfig(pd.BaseModel):
    """Evaluation and metrics configuration."""

    metrics: List[
        Literal["accuracy", "f1_macro", "f1_micro", "precision", "recall"]
    ] = Field(["accuracy"], min_length=1, description="Metrics to compute")
    save_predictions: bool = Field(
        False, description="Whether to save model predictions"
    )
    save_confusion_matrix: bool = Field(
        True, description="Whether to save confusion matrix"
    )
    save_per_class_metrics: bool = Field(
        False, description="Whether to save per-class metrics"
    )

    @pd.field_validator("metrics")
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        if "accuracy" not in v:
            v.append("accuracy")  # Always include accuracy
        return list(set(v))  # Remove duplicates


# ---- Data Augmentation Configuration ----


class DataAugmentationConfig(pd.BaseModel):
    """Data augmentation configuration."""

    enabled: bool = Field(False, description="Whether to apply data augmentation")
    rotation_degrees: float = Field(
        10.0, ge=0.0, le=45.0, description="Max rotation in degrees"
    )
    translation: float = Field(
        0.1, ge=0.0, le=0.5, description="Max translation as fraction"
    )
    scale: List[float] = Field(
        [0.9, 1.1], min_length=2, max_length=2, description="Scale range [min, max]"
    )

    @pd.field_validator("scale")
    @classmethod
    def validate_scale(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("scale must have exactly 2 values [min, max]")
        if v[0] >= v[1]:
            raise ValueError("scale[0] must be < scale[1]")
        if v[0] <= 0:
            raise ValueError("scale values must be positive")
        return v


# ---- Regularization Configuration ----


class RegularizationConfig(pd.BaseModel):
    """Regularization techniques configuration."""

    label_smoothing: float = Field(
        0.0, ge=0.0, le=0.5, description="Label smoothing factor"
    )
    mixup_alpha: float = Field(0.0, ge=0.0, le=2.0, description="Mixup alpha parameter")


# ---- Resource Management ----


class ResourceConfig(pd.BaseModel):
    """Resource management for cluster/HPC environments."""

    max_memory_gb: int = Field(
        8, gt=0, le=128, description="Maximum memory usage in GB"
    )
    distributed: bool = Field(False, description="Whether to use distributed training")
    mixed_precision: bool = Field(
        False, description="Whether to use mixed precision training"
    )


# ---- Experiment Configuration ----


class ExperimentConfig(pd.BaseModel):
    """Main experiment configuration that ties everything together.

    This is the top-level config object that gets loaded from YAML files.
    It validates the entire experiment specification and provides helpful
    error messages for invalid configurations.
    """

    # Core configurations
    data: DataConfig
    model: ModelConfig = Field(..., discriminator="name")
    training: TrainingConfig
    evaluation: EvaluationConfig

    # Experiment metadata
    experiment: ExperimentMetadata
    logging: LoggingConfig

    # Optional advanced configurations
    data_augmentation: Optional[DataAugmentationConfig] = None
    regularization: Optional[RegularizationConfig] = None
    resources: Optional[ResourceConfig] = None

    @pd.model_validator(mode="after")
    def validate_experiment_consistency(self) -> "ExperimentConfig":
        """Cross-field validation to ensure configuration consistency."""

        # Validate model-data compatibility
        if isinstance(self.model, LinearModel) and self.model.input_size != 784:
            raise ValueError("LinearModel input_size must be 784 for MNIST")

        # Validate training-evaluation consistency
        if self.training.epochs < 2 and self.evaluation.save_per_class_metrics:
            raise ValueError(
                "Need at least 2 epochs to compute meaningful per-class metrics"
            )

        # Validate scheduler-optimizer compatibility
        if (
            self.training.scheduler
            and self.training.scheduler.name == "reduce_lr_on_plateau"
            and self.training.optimizer == "sgd"
            and self.training.momentum == 0.0
        ):
            raise ValueError(
                "SGD with reduce_lr_on_plateau typically needs momentum > 0"
            )

        return self


class ExperimentMetadata(pd.BaseModel):
    """Experiment metadata and configuration."""

    name: str = Field(..., min_length=1, max_length=100, description="Experiment name")
    description: str = Field("", max_length=500, description="Experiment description")
    random_seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    device: Literal["auto", "cpu", "cuda", "mps"] = Field(
        "auto", description="Device to use for training"
    )
    cache_dir: Path = Field(Path("cache"), description="Directory for ExCa cache")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Logging level"
    )
    save_model: bool = Field(True, description="Whether to save trained model")
    save_best_only: bool = Field(
        True, description="Whether to save only the best model"
    )

    @pd.field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "name must contain only alphanumeric characters, hyphens, and underscores"
            )
        return v

    @pd.field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, v: Path) -> Path:
        return Path(v)


class LoggingConfig(pd.BaseModel):
    """Logging configuration."""

    log_every_n_steps: int = Field(10, ge=1, description="Log metrics every N steps")
    log_gradients: bool = Field(False, description="Whether to log gradient norms")
    log_weights: bool = Field(False, description="Whether to log weight distributions")


# ---- Utility Functions ----


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load and validate experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated ExperimentConfig object

    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If config file doesn't exist
    """
    import yaml

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    return ExperimentConfig(**config_dict)


def save_config(config: ExperimentConfig, output_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        config: ExperimentConfig object to save
        output_path: Path to save YAML file
    """
    import yaml

    config_dict = config.model_dump(mode="python")

    # Convert Path objects to strings for YAML serialization
    def convert_paths(obj):
        if isinstance(obj, dict):
            return {k: convert_paths(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths(item) for item in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    config_dict = convert_paths(config_dict)

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Export key classes for convenience
__all__ = [
    "ExperimentConfig",
    "DataConfig",
    "ModelConfig",
    "LinearModel",
    "MLPModel",
    "CNNModel",
    "TrainingConfig",
    "EvaluationConfig",
    "ExperimentMetadata",
    "load_config",
    "save_config",
]
