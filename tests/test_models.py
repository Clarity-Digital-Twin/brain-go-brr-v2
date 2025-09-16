"""Tests for machine learning models.

This module tests all model implementations including architecture,
forward passes, parameter counting, and factory functions.
"""

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from src.experiment.models import (
    BaseModel,
    CNNClassifier,
    LinearClassifier,
    MLPClassifier,
    count_model_parameters,
    create_model,
    create_optimizer,
    create_scheduler,
    get_model_summary,
    validate_model_config,
)
from src.experiment.schemas import (
    CNNModel,
    LinearModel,
    MLPModel,
    SchedulerConfig,
    TrainingConfig,
)


class TestBaseModel:
    """Test BaseModel functionality."""

    def test_base_model_interface(self):
        """Test that BaseModel provides expected interface."""

        class TestModel(BaseModel):
            def forward(self, x):
                return x

        model = TestModel()

        # Should have required methods
        assert hasattr(model, "get_num_parameters")
        assert hasattr(model, "get_model_info")
        assert callable(model.forward)

    def test_parameter_counting(self):
        """Test parameter counting in BaseModel."""

        class SimpleModel(BaseModel):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        param_count = model.get_num_parameters()

        # Linear layer: (10 * 5) + 5 = 55 parameters
        assert param_count == 55

    def test_model_info(self):
        """Test model info dictionary."""

        class InfoModel(BaseModel):
            def __init__(self):
                super().__init__()
                self._input_shape = (784,)
                self.linear = nn.Linear(784, 10)

            def forward(self, x):
                return self.linear(x)

        model = InfoModel()
        info = model.get_model_info()

        assert "model_class" in info
        assert "num_parameters" in info
        assert "input_shape" in info
        assert info["model_class"] == "InfoModel"
        assert info["input_shape"] == (784,)


class TestLinearClassifier:
    """Test LinearClassifier model."""

    def test_linear_model_creation(self):
        """Test LinearClassifier creation from config."""
        config = LinearModel(input_size=784, num_classes=10, bias=True)
        model = LinearClassifier(config)

        assert isinstance(model, LinearClassifier)
        assert isinstance(model.linear, nn.Linear)
        assert model.linear.in_features == 784
        assert model.linear.out_features == 10
        assert model.linear.bias is not None

    def test_linear_model_no_bias(self):
        """Test LinearClassifier without bias."""
        config = LinearModel(input_size=784, num_classes=10, bias=False)
        model = LinearClassifier(config)

        assert model.linear.bias is None

    def test_linear_forward_pass_flattened(self):
        """Test forward pass with already flattened input."""
        config = LinearModel(input_size=784, num_classes=10)
        model = LinearClassifier(config)

        batch_size = 32
        x = torch.randn(batch_size, 784)

        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_linear_forward_pass_image(self):
        """Test forward pass with image input (should be flattened)."""
        config = LinearModel(input_size=784, num_classes=10)
        model = LinearClassifier(config)

        batch_size = 16
        x = torch.randn(batch_size, 1, 28, 28)  # Image format

        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()

    def test_linear_parameter_count(self):
        """Test parameter counting for linear model."""
        config = LinearModel(input_size=784, num_classes=10, bias=True)
        model = LinearClassifier(config)

        expected_params = 784 * 10 + 10  # weights + bias
        assert model.get_num_parameters() == expected_params


class TestMLPClassifier:
    """Test MLPClassifier model."""

    def test_mlp_creation_default(self):
        """Test MLP creation with default parameters."""
        config = MLPModel()
        model = MLPClassifier(config)

        assert isinstance(model, MLPClassifier)
        assert isinstance(model.layers, nn.Sequential)

        # Should have (linear + activation + dropout) * num_layers + final linear
        # flatten is separate from model.layers
        expected_layers = (3 * config.num_layers) + 1  # hidden layers + output
        assert len(model.layers) == expected_layers

    def test_mlp_custom_architecture(self):
        """Test MLP with custom architecture."""
        config = MLPModel(
            input_size=784,
            hidden_size=256,
            num_layers=3,
            num_classes=10,
            dropout=0.3,
            activation="relu",
        )
        model = MLPClassifier(config)

        # Check that model uses correct parameters
        assert model.config.hidden_size == 256
        assert model.config.num_layers == 3
        assert model.config.dropout == 0.3

    def test_mlp_activation_functions(self):
        """Test different activation functions."""
        activations = ["relu", "tanh", "sigmoid", "gelu"]

        for activation in activations:
            config = MLPModel(activation=activation)
            model = MLPClassifier(config)

            # Should create without error
            assert isinstance(model, MLPClassifier)

    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        config = MLPModel(input_size=784, hidden_size=128, num_layers=2)
        model = MLPClassifier(config)

        batch_size = 8
        x = torch.randn(batch_size, 1, 28, 28)

        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_mlp_parameter_initialization(self):
        """Test that MLP parameters are properly initialized."""
        config = MLPModel()
        model = MLPClassifier(config)

        # Check that weights are initialized (not zero)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                assert not torch.allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                if module.bias is not None:
                    assert torch.allclose(module.bias, torch.zeros_like(module.bias))


class TestCNNClassifier:
    """Test CNNClassifier model."""

    def test_cnn_creation_default(self):
        """Test CNN creation with default parameters."""
        config = CNNModel()
        model = CNNClassifier(config)

        assert isinstance(model, CNNClassifier)
        assert isinstance(model.conv_layers, nn.Sequential)
        assert isinstance(model.global_pool, nn.AdaptiveAvgPool2d)
        assert isinstance(model.classifier, nn.Linear)

    def test_cnn_custom_channels(self):
        """Test CNN with custom channel configuration."""
        config = CNNModel(
            input_channels=1, channels=[16, 32, 64, 128], kernel_size=5, dropout=0.3
        )
        model = CNNClassifier(config)

        assert model.config.channels == [16, 32, 64, 128]
        assert model.config.kernel_size == 5

        # Final classifier should use last channel size
        assert model.classifier.in_features == 128

    def test_cnn_batch_normalization(self):
        """Test CNN with and without batch normalization."""
        # With batch norm
        config_bn = CNNModel(batch_norm=True)
        model_bn = CNNClassifier(config_bn)

        # Without batch norm
        config_no_bn = CNNModel(batch_norm=False)
        model_no_bn = CNNClassifier(config_no_bn)

        # Both should create successfully
        assert isinstance(model_bn, CNNClassifier)
        assert isinstance(model_no_bn, CNNClassifier)

    def test_cnn_forward_pass(self):
        """Test CNN forward pass with correct input shape."""
        config = CNNModel(input_channels=1, channels=[32, 64])
        model = CNNClassifier(config)

        batch_size = 4
        x = torch.randn(batch_size, 1, 28, 28)

        output = model(x)

        assert output.shape == (batch_size, 10)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_cnn_single_image_input(self):
        """Test CNN with single image (no batch dimension)."""
        config = CNNModel()
        model = CNNClassifier(config)

        x = torch.randn(1, 28, 28)  # Single image

        output = model(x)

        assert output.shape == (1, 10)  # Should add batch dimension

    def test_cnn_feature_maps(self):
        """Test CNN feature map extraction."""
        config = CNNModel(channels=[16, 32])
        model = CNNClassifier(config)

        x = torch.randn(2, 1, 28, 28)

        feature_maps = model.get_feature_maps(x)

        assert isinstance(feature_maps, list)
        assert len(feature_maps) > 0
        # Each feature map should have batch dimension
        for fmap in feature_maps:
            assert fmap.size(0) == 2


class TestModelFactory:
    """Test model factory functions."""

    def test_create_linear_model(self):
        """Test creating linear model from config."""
        config = LinearModel(input_size=784, num_classes=10)
        model = create_model(config)

        assert isinstance(model, LinearClassifier)
        assert model.config == config

    def test_create_mlp_model(self):
        """Test creating MLP model from config."""
        config = MLPModel(hidden_size=256, num_layers=3)
        model = create_model(config)

        assert isinstance(model, MLPClassifier)
        assert model.config == config

    def test_create_cnn_model(self):
        """Test creating CNN model from config."""
        config = CNNModel(channels=[32, 64, 128])
        model = create_model(config)

        assert isinstance(model, CNNClassifier)
        assert model.config == config

    def test_create_model_invalid_config(self):
        """Test error handling for invalid model config."""
        invalid_config = "not_a_model_config"

        with pytest.raises(ValueError, match="Unsupported model type"):
            create_model(invalid_config)


class TestModelUtils:
    """Test model utility functions."""

    def test_count_model_parameters(self):
        """Test parameter counting utility."""
        model = nn.Sequential(
            nn.Linear(100, 50),  # 100*50 + 50 = 5050
            nn.ReLU(),
            nn.Linear(50, 10),  # 50*10 + 10 = 510
        )

        param_counts = count_model_parameters(model)

        assert param_counts["total_parameters"] == 5560
        assert param_counts["trainable_parameters"] == 5560
        assert param_counts["non_trainable_parameters"] == 0

    def test_count_parameters_with_frozen(self):
        """Test parameter counting with frozen parameters."""
        model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 2))

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        param_counts = count_model_parameters(model)

        assert param_counts["total_parameters"] == 67  # (10*5+5) + (5*2+2)
        assert param_counts["trainable_parameters"] == 12  # Only second layer
        assert param_counts["non_trainable_parameters"] == 55  # First layer

    def test_get_model_summary(self):
        """Test model summary generation."""
        config = LinearModel()
        model = LinearClassifier(config)

        summary = get_model_summary(model, (784,))

        assert isinstance(summary, str)
        assert "Model Summary" in summary
        assert "LinearClassifier" in summary
        assert "784" in summary  # Input shape

    def test_validate_model_config_linear(self):
        """Test model config validation for linear model."""
        config = LinearModel(input_size=784)

        # Valid input shape
        assert validate_model_config(config, (784,)) is True

        # Invalid input shape
        assert validate_model_config(config, (1000,)) is False

    def test_validate_model_config_cnn(self):
        """Test model config validation for CNN model."""
        config = CNNModel(input_channels=1)

        # Valid input shape (C, H, W)
        assert validate_model_config(config, (1, 28, 28)) is True

        # Invalid channel count
        assert validate_model_config(config, (3, 28, 28)) is False

        # Invalid dimensionality
        assert validate_model_config(config, (784,)) is False

    def test_validate_model_config_exception_handling(self):
        """Test validation with invalid config."""
        # This should handle exceptions gracefully
        result = validate_model_config("invalid_config", (784,))
        assert result is False


class TestOptimizerCreation:
    """Test optimizer creation utilities."""

    def test_create_adam_optimizer(self):
        """Test Adam optimizer creation."""
        model = nn.Linear(10, 2)
        config = TrainingConfig(
            optimizer="adam", learning_rate=0.001, weight_decay=0.01
        )

        optimizer = create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]["lr"] == 0.001
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_create_sgd_optimizer(self):
        """Test SGD optimizer creation."""
        model = nn.Linear(10, 2)
        config = TrainingConfig(
            optimizer="sgd", learning_rate=0.01, momentum=0.9, weight_decay=0.0001
        )

        optimizer = create_optimizer(model, config)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]["lr"] == 0.01
        assert optimizer.param_groups[0]["momentum"] == 0.9
        assert optimizer.param_groups[0]["weight_decay"] == 0.0001

    def test_create_unsupported_optimizer(self):
        """Test error handling for unsupported optimizer."""
        from unittest.mock import Mock

        model = nn.Linear(10, 2)
        # Create mock config with unsupported optimizer
        mock_config = Mock()
        mock_config.optimizer = "unsupported_optimizer"
        mock_config.learning_rate = 0.001
        mock_config.weight_decay = 0.0

        with pytest.raises(ValueError, match="Unsupported optimizer"):
            create_optimizer(model, mock_config)


class TestSchedulerCreation:
    """Test learning rate scheduler creation."""

    def test_create_reduce_lr_scheduler(self):
        """Test ReduceLROnPlateau scheduler creation."""
        optimizer = torch.optim.Adam(nn.Linear(10, 2).parameters())
        scheduler_config = SchedulerConfig(
            name="reduce_lr_on_plateau", factor=0.5, patience=5, min_lr=1e-6
        )
        config = TrainingConfig(scheduler=scheduler_config)

        scheduler = create_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_create_step_scheduler(self):
        """Test StepLR scheduler creation."""
        optimizer = torch.optim.Adam(nn.Linear(10, 2).parameters())
        scheduler_config = SchedulerConfig(name="step", factor=0.1, patience=10)
        config = TrainingConfig(scheduler=scheduler_config)

        scheduler = create_scheduler(optimizer, config)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_create_scheduler_none(self):
        """Test scheduler creation when no scheduler specified."""
        optimizer = torch.optim.Adam(nn.Linear(10, 2).parameters())
        config = TrainingConfig(scheduler=None)

        scheduler = create_scheduler(optimizer, config)

        assert scheduler is None

    def test_create_unsupported_scheduler(self):
        """Test error handling for unsupported scheduler."""
        from unittest.mock import Mock

        optimizer = torch.optim.Adam(nn.Linear(10, 2).parameters())
        # Create mock config with unsupported scheduler
        mock_config = Mock()
        mock_scheduler_config = Mock()
        mock_scheduler_config.name = "unsupported_scheduler"
        mock_config.scheduler = mock_scheduler_config

        with pytest.raises(ValueError, match="Unsupported scheduler"):
            create_scheduler(optimizer, mock_config)


@pytest.mark.integration
class TestModelIntegration:
    """Integration tests for models with real data flow."""

    def test_linear_model_training_step(self):
        """Test linear model can perform a training step."""
        config = LinearModel()
        model = LinearClassifier(config)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # Simulate training data
        batch_size = 8
        x = torch.randn(batch_size, 1, 28, 28)
        y = torch.randint(0, 10, (batch_size,))

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check that gradients were computed
        assert all(param.grad is not None for param in model.parameters())
        assert torch.isfinite(loss).all()

    def test_model_eval_mode(self):
        """Test models work correctly in eval mode."""
        configs = [
            LinearModel(),
            MLPModel(dropout=0.5),  # High dropout to test difference
            CNNModel(dropout=0.3),
        ]

        for config in configs:
            model = create_model(config)
            x = torch.randn(4, 1, 28, 28)

            # Training mode
            model.train()
            out_train = model(x)

            # Eval mode
            model.eval()
            out_eval = model(x)

            # Outputs should have same shape
            assert out_train.shape == out_eval.shape

            # For models with dropout, outputs might be different
            # But shape should be consistent
            assert out_eval.shape == (4, 10)


@pytest.mark.slow
class TestModelPerformance:
    """Performance tests for models (marked as slow)."""

    def test_model_memory_usage(self):
        """Test model memory footprint."""
        # This would test memory usage of different models
        # Implementation depends on memory profiling tools
        pass

    def test_model_inference_speed(self):
        """Test model inference speed."""
        # This would benchmark inference speed
        # Implementation depends on timing requirements
        pass
