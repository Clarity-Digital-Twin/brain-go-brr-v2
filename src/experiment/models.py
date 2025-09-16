"""Machine learning models for MNIST classification.

This module implements three types of models:
- LinearModel: Simple logistic regression
- MLPModel: Multi-layer perceptron with configurable architecture
- CNNModel: Convolutional neural network

All models follow consistent interfaces and integrate with the Pydantic
configuration system for reproducible experiments.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from .schemas import CNNModel, LinearModel, MLPModel, ModelConfig

logger = logging.getLogger(__name__)


# ---- Base Model Interface ----


class BaseModel(nn.Module):
    """Base class for all models with common functionality."""

    def __init__(self):
        super().__init__()
        self._input_shape = None
        self._num_parameters = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses."""
        raise NotImplementedError

    def get_num_parameters(self) -> int:
        """Count total number of trainable parameters."""
        if self._num_parameters is None:
            self._num_parameters = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
        return self._num_parameters

    def get_model_info(self) -> Dict[str, any]:
        """Get model information for logging/debugging."""
        return {
            "model_class": self.__class__.__name__,
            "num_parameters": self.get_num_parameters(),
            "input_shape": self._input_shape,
        }


# ---- Linear Model ----


class LinearClassifier(BaseModel):
    """Simple linear/logistic regression model.

    Flattens input images and applies a single linear transformation
    followed by softmax for classification.
    """

    def __init__(self, config: LinearModel):
        super().__init__()
        self.config = config
        self._input_shape = (config.input_size,)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(
            in_features=config.input_size,
            out_features=config.num_classes,
            bias=config.bias,
        )

        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        if config.bias:
            nn.init.zeros_(self.linear.bias)

        logger.info(
            f"Created LinearClassifier: {config.input_size} -> {config.num_classes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, input_size)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        if x.dim() > 2:  # If image format (B, C, H, W)
            x = self.flatten(x)

        return self.linear(x)


# ---- Multi-Layer Perceptron ----


class MLPClassifier(BaseModel):
    """Multi-layer perceptron with configurable architecture.

    Supports multiple hidden layers, dropout, and various activation functions.
    """

    def __init__(self, config: MLPModel):
        super().__init__()
        self.config = config
        self._input_shape = (config.input_size,)

        self.flatten = nn.Flatten()

        # Build layers dynamically
        layers = []
        in_features = config.input_size

        # Hidden layers
        for i in range(config.num_layers):
            layers.extend(
                [
                    nn.Linear(in_features, config.hidden_size),
                    self._get_activation(config.activation),
                    nn.Dropout(config.dropout),
                ]
            )
            in_features = config.hidden_size

        # Output layer (no activation - raw logits)
        layers.append(nn.Linear(in_features, config.num_classes))

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Created MLPClassifier: {config.input_size} -> {config.hidden_size}x{config.num_layers} -> {config.num_classes}"
        )

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function from string name."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "gelu": nn.GELU(),
        }
        return activations[activation]

    def _init_weights(self):
        """Initialize model weights using appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.activation == "relu":
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, input_size)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        if x.dim() > 2:  # If image format
            x = self.flatten(x)

        return self.layers(x)


# ---- Convolutional Neural Network ----


class CNNClassifier(BaseModel):
    """Convolutional neural network for image classification.

    Features:
    - Multiple convolutional layers with configurable channels
    - Batch normalization (optional)
    - Dropout for regularization
    - Global average pooling before classification
    """

    def __init__(self, config: CNNModel):
        super().__init__()
        self.config = config
        self._input_shape = (config.input_channels, 28, 28)  # MNIST specific

        # Build convolutional layers
        conv_layers = []
        in_channels = config.input_channels

        for out_channels in config.channels:
            layer_group = [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=config.kernel_size,
                    padding=config.kernel_size // 2,  # Same padding
                    bias=not config.batch_norm,
                )
            ]

            if config.batch_norm:
                layer_group.append(nn.BatchNorm2d(out_channels))

            layer_group.extend(
                [
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(config.dropout),
                ]
            )

            conv_layers.extend(layer_group)
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Global average pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(config.channels[-1], config.num_classes)

        # Initialize weights
        self._init_weights()

        logger.info(
            f"Created CNNClassifier: channels={config.channels}, kernel_size={config.kernel_size}"
        )

    def _init_weights(self):
        """Initialize CNN weights using appropriate schemes."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Ensure correct input shape
        if x.dim() == 3:  # Add batch dimension if missing
            x = x.unsqueeze(0)

        # Convolutional feature extraction
        features = self.conv_layers(x)

        # Global pooling and classification
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)

        return self.classifier(flattened)

    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract feature maps from intermediate layers.

        Useful for visualization and analysis.

        Args:
            x: Input tensor

        Returns:
            List of feature map tensors from each conv layer
        """
        feature_maps = []

        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if isinstance(layer, nn.ReLU):  # Save after activation
                feature_maps.append(x.detach().clone())

        return feature_maps


# ---- Model Factory Functions ----


def create_model(config: ModelConfig) -> BaseModel:
    """Factory function to create models from configuration.

    Args:
        config: Model configuration (discriminated union)

    Returns:
        Instantiated model

    Raises:
        ValueError: If model type is not supported
    """
    if isinstance(config, LinearModel):
        return LinearClassifier(config)
    elif isinstance(config, MLPModel):
        return MLPClassifier(config)
    elif isinstance(config, CNNModel):
        return CNNClassifier(config)
    else:
        raise ValueError(f"Unsupported model type: {type(config)}")


def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Count parameters in a model with breakdown.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def get_model_summary(model: BaseModel, input_shape: Tuple[int, ...]) -> str:
    """Generate a model summary string.

    Args:
        model: Model to summarize
        input_shape: Input shape (without batch dimension)

    Returns:
        Formatted summary string
    """
    param_counts = count_model_parameters(model)
    model_info = model.get_model_info()

    summary = f"""
Model Summary:
--------------
Class: {model_info['model_class']}
Input Shape: {input_shape}
Parameters:
  - Total: {param_counts['total_parameters']:,}
  - Trainable: {param_counts['trainable_parameters']:,}
  - Non-trainable: {param_counts['non_trainable_parameters']:,}

Architecture:
{model}
"""
    return summary


def validate_model_config(config: ModelConfig, input_shape: Tuple[int, ...]) -> bool:
    """Validate model configuration against input data shape.

    Args:
        config: Model configuration
        input_shape: Expected input shape

    Returns:
        True if configuration is valid
    """
    try:
        if isinstance(config, LinearModel):
            expected_input_size = 1
            for dim in input_shape:
                expected_input_size *= dim
            if config.input_size != expected_input_size:
                logger.error(
                    f"LinearModel input_size mismatch: config={config.input_size}, expected={expected_input_size}"
                )
                return False

        elif isinstance(config, MLPModel):
            expected_input_size = 1
            for dim in input_shape:
                expected_input_size *= dim
            if config.input_size != expected_input_size:
                logger.error(
                    f"MLPModel input_size mismatch: config={config.input_size}, expected={expected_input_size}"
                )
                return False

        elif isinstance(config, CNNModel):
            if len(input_shape) != 3:  # (C, H, W)
                logger.error(
                    f"CNNModel expects 3D input (C, H, W), got {len(input_shape)}D"
                )
                return False
            if config.input_channels != input_shape[0]:
                logger.error(
                    f"CNNModel channel mismatch: config={config.input_channels}, input={input_shape[0]}"
                )
                return False

        else:
            # Invalid config type
            logger.error(f"Invalid model config type: {type(config)}")
            return False

        logger.info(
            f"Model configuration validation passed for {type(config).__name__}"
        )
        return True

    except Exception as e:
        logger.error(f"Model configuration validation failed: {e}")
        return False


# ---- Training Utilities ----


def create_optimizer(model: nn.Module, config) -> Optimizer:
    """Create optimizer from training configuration.

    Args:
        model: Model to optimize
        config: Training configuration with optimizer settings

    Returns:
        Configured optimizer
    """
    if config.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def create_scheduler(optimizer: Optimizer, config):
    """Create learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule
        config: Training configuration with scheduler settings

    Returns:
        Configured scheduler or None
    """
    if not config.scheduler:
        return None

    scheduler_config = config.scheduler

    if scheduler_config.name == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.factor,
            patience=scheduler_config.patience,
            min_lr=scheduler_config.min_lr,
        )
    elif scheduler_config.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.patience,  # Reuse patience as step_size
            gamma=scheduler_config.factor,
        )
    elif scheduler_config.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,  # Default, should be set based on total epochs
            eta_min=scheduler_config.min_lr,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_config.name}")


__all__ = [
    "BaseModel",
    "LinearClassifier",
    "MLPClassifier",
    "CNNClassifier",
    "create_model",
    "count_model_parameters",
    "get_model_summary",
    "validate_model_config",
    "create_optimizer",
    "create_scheduler",
]
