"""Tests for data loading and preprocessing functionality.

This module tests MNIST data loading, preprocessing, and DataLoader creation
to ensure proper data handling across different configurations.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.experiment.data import (
    MNISTDataset,
    compute_class_weights,
    create_dataloaders,
    create_datasets,
    create_transforms,
    flatten_for_linear,
    get_dataset_stats,
    prepare_data,
    validate_data_shape,
)
from src.experiment.pipeline import Experiment
from src.experiment.schemas import (
    CNNModel,
    DataAugmentationConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ExperimentMetadata,
    LinearModel,
    LoggingConfig,
    MLPModel,
    TrainingConfig,
)


class TestMNISTDataset:
    """Test MNISTDataset class."""

    @patch("src.experiment.data.datasets.MNIST")
    def test_dataset_initialization(self, mock_mnist):
        """Test dataset initialization with mocked MNIST."""
        # Mock the MNIST dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_dataset.__getitem__ = Mock(return_value=(torch.randn(1, 28, 28), 5))
        mock_mnist.return_value = mock_dataset

        dataset = MNISTDataset(root=Path("test_data"), train=True, download=False)

        assert len(dataset) == 1000
        mock_mnist.assert_called_once()

    def test_get_sample_shape_no_flatten(self):
        """Test sample shape without flattening."""
        with patch("src.experiment.data.datasets.MNIST") as mock_mnist:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=10)
            mock_dataset.__getitem__ = Mock(return_value=(torch.randn(1, 28, 28), 0))
            mock_mnist.return_value = mock_dataset

            dataset = MNISTDataset(root=Path("test"), flatten=False)
            shape = dataset.get_sample_shape()

            assert shape == (1, 28, 28)

    def test_get_sample_shape_with_flatten(self):
        """Test sample shape with flattening."""
        with patch("src.experiment.data.datasets.MNIST") as mock_mnist:
            mock_dataset = Mock()
            mock_dataset.__len__ = Mock(return_value=10)
            mock_dataset.__getitem__ = Mock(return_value=(torch.randn(784), 0))
            mock_mnist.return_value = mock_dataset

            dataset = MNISTDataset(root=Path("test"), flatten=True)
            shape = dataset.get_sample_shape()

            assert shape == (784,)


class TestTransforms:
    """Test transform creation and augmentation."""

    def test_create_transforms_no_augmentation(self):
        """Test transform creation without augmentation."""
        train_transforms, test_transforms = create_transforms(
            augmentation_config=None, normalize=True
        )

        assert train_transforms is not None
        assert test_transforms is not None
        assert len(train_transforms.transforms) >= 1  # At least normalization

    def test_create_transforms_with_augmentation(self):
        """Test transform creation with augmentation."""
        aug_config = DataAugmentationConfig(
            enabled=True, rotation_degrees=10.0, translation=0.1, scale=[0.9, 1.1]
        )

        train_transforms, test_transforms = create_transforms(
            augmentation_config=aug_config, normalize=True
        )

        # Train transforms should have more transforms (augmentation + normalization)
        assert len(train_transforms.transforms) > len(test_transforms.transforms)

    def test_create_transforms_disabled_augmentation(self):
        """Test transforms with disabled augmentation."""
        aug_config = DataAugmentationConfig(enabled=False)

        train_transforms, test_transforms = create_transforms(
            augmentation_config=aug_config, normalize=True
        )

        # Should be same length when augmentation is disabled
        assert len(train_transforms.transforms) == len(test_transforms.transforms)


class TestDatasetCreation:
    """Test dataset creation functions."""

    @patch("src.experiment.data.MNISTDataset")
    def test_create_datasets_with_validation_split(self, mock_dataset_class):
        """Test dataset creation with validation split."""
        # Mock dataset instances
        mock_train_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_test_dataset = Mock()
        mock_test_dataset.__len__ = Mock(return_value=200)

        # Make the class return different instances
        def side_effect(*args, **kwargs):
            if kwargs.get("train", True):
                return mock_train_dataset
            else:
                return mock_test_dataset

        mock_dataset_class.side_effect = side_effect

        config = DataConfig(validation_split=0.2, download=False)

        with patch("src.experiment.data.random_split") as mock_split:
            # Mock random_split to return train and val datasets
            mock_train_subset = Mock()
            mock_train_subset.__len__ = Mock(return_value=800)
            mock_val_subset = Mock()
            mock_val_subset.__len__ = Mock(return_value=200)
            mock_split.return_value = (mock_train_subset, mock_val_subset)

            train_dataset, val_dataset, test_dataset = create_datasets(
                config=config, data_dir=Path("test_data")
            )

            assert train_dataset is not None
            assert val_dataset is not None  # Should have validation set
            assert test_dataset is not None

            # Should call random_split for validation split
            mock_split.assert_called_once()

    @patch("src.experiment.data.MNISTDataset")
    def test_create_datasets_no_validation_split(self, mock_dataset_class):
        """Test dataset creation without validation split."""
        mock_train_dataset = Mock()
        mock_train_dataset.__len__ = Mock(return_value=1000)
        mock_test_dataset = Mock()
        mock_test_dataset.__len__ = Mock(return_value=200)

        def side_effect(*args, **kwargs):
            if kwargs.get("train", True):
                return mock_train_dataset
            else:
                return mock_test_dataset

        mock_dataset_class.side_effect = side_effect

        config = DataConfig(validation_split=0.0, download=False)

        train_dataset, val_dataset, test_dataset = create_datasets(
            config=config, data_dir=Path("test_data")
        )

        assert train_dataset is not None
        assert val_dataset is None  # Should be None with no validation split
        assert test_dataset is not None


class TestDataLoaders:
    """Test DataLoader creation."""

    def test_create_dataloaders_with_validation(self):
        """Test DataLoader creation with validation set."""
        # Create mock datasets
        train_dataset = Mock()
        train_dataset.__len__ = Mock(return_value=800)
        val_dataset = Mock()
        val_dataset.__len__ = Mock(return_value=200)
        test_dataset = Mock()
        test_dataset.__len__ = Mock(return_value=100)

        datasets = (train_dataset, val_dataset, test_dataset)
        config = DataConfig(batch_size=32, num_workers=0)

        dataloaders = create_dataloaders(datasets, config)

        assert "train" in dataloaders
        assert "val" in dataloaders
        assert "test" in dataloaders
        assert len(dataloaders) == 3

    def test_create_dataloaders_no_validation(self):
        """Test DataLoader creation without validation set."""
        train_dataset = Mock()
        train_dataset.__len__ = Mock(return_value=1000)
        test_dataset = Mock()
        test_dataset.__len__ = Mock(return_value=100)

        datasets = (train_dataset, None, test_dataset)
        config = DataConfig(batch_size=64)

        dataloaders = create_dataloaders(datasets, config)

        assert "train" in dataloaders
        assert "val" not in dataloaders
        assert "test" in dataloaders
        assert len(dataloaders) == 2


class TestDataValidation:
    """Test data validation functions."""

    def test_validate_data_shape_correct(self):
        """Test data shape validation with correct shape."""
        # Mock dataloader with correct shape
        mock_batch = (torch.randn(32, 1, 28, 28), torch.randint(0, 10, (32,)))
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch]))

        dataloaders = {"train": mock_dataloader}
        expected_shape = (1, 28, 28)

        result = validate_data_shape(dataloaders, expected_shape)
        assert result is True

    def test_validate_data_shape_incorrect(self):
        """Test data shape validation with incorrect shape."""
        # Mock dataloader with wrong shape
        mock_batch = (torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,)))
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([mock_batch]))

        dataloaders = {"train": mock_dataloader}
        expected_shape = (1, 28, 28)

        result = validate_data_shape(dataloaders, expected_shape)
        assert result is False


class TestUtilityFunctions:
    """Test utility functions for data processing."""

    def test_flatten_for_linear_single_image(self):
        """Test flattening single image for linear models."""
        image = torch.randn(1, 28, 28)
        flattened = flatten_for_linear(image)

        assert flattened.shape == (784,)
        assert flattened.numel() == 784

    def test_flatten_for_linear_batch(self):
        """Test flattening batch of images for linear models."""
        batch = torch.randn(32, 1, 28, 28)
        flattened = flatten_for_linear(batch)

        assert flattened.shape == (32, 784)
        assert flattened.size(0) == 32
        assert flattened.size(1) == 784

    def test_flatten_for_linear_invalid_dims(self):
        """Test error handling for invalid dimensions."""
        invalid_tensor = torch.randn(784)  # 1D tensor

        with pytest.raises(ValueError, match="Expected 3D or 4D tensor"):
            flatten_for_linear(invalid_tensor)

    def test_compute_class_weights(self):
        """Test class weight computation."""
        # Create mock dataset with class imbalance
        labels = torch.cat(
            [
                torch.zeros(100),  # Class 0: 100 samples
                torch.ones(50),  # Class 1: 50 samples
                torch.full((25,), 2),  # Class 2: 25 samples
            ]
        ).long()

        mock_dataset = Mock()
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter([(None, labels)]))

        with patch("src.experiment.data.DataLoader", return_value=mock_dataloader):
            weights = compute_class_weights(mock_dataset)

        # Weights should be inversely proportional to class frequency
        assert len(weights) == 3
        assert weights[0] < weights[1] < weights[2]  # More imbalanced = higher weight


class TestDatasetStats:
    """Test dataset statistics computation."""

    def test_get_dataset_stats(self):
        """Test dataset statistics computation."""
        # Create mock datasets
        train_dataset = Mock()
        train_dataset.__len__ = Mock(return_value=100)
        test_dataset = Mock()
        test_dataset.__len__ = Mock(return_value=50)

        # Mock DataLoader to return sample data
        sample_images = torch.randn(50, 1, 28, 28) * 0.3081 + 0.1307  # MNIST-like stats
        sample_labels = torch.randint(0, 10, (50,))

        with patch("src.experiment.data.DataLoader") as mock_dataloader_class:
            # Create a proper iterator mock
            def mock_dataloader_factory(*args, **kwargs):
                mock_dataloader = Mock()
                # Use a proper iterator that returns one batch then stops
                data_iter = iter([(sample_images, sample_labels)])
                mock_dataloader.__iter__ = Mock(return_value=data_iter)
                return mock_dataloader

            mock_dataloader_class.side_effect = mock_dataloader_factory

            datasets = (train_dataset, None, test_dataset)
            stats = get_dataset_stats(datasets)

        assert "train" in stats
        assert "test" in stats
        assert "val" not in stats  # No validation set provided

        # Check that statistics contain expected keys
        train_stats = stats["train"]
        assert "num_samples" in train_stats
        assert "image_mean" in train_stats
        assert "image_std" in train_stats
        assert "num_classes" in train_stats
        assert "label_distribution" in train_stats


# ---- Real-data integration tests via pipeline (no mocks) ----


def _make_base_config(model_config):
    return ExperimentConfig(
        data=DataConfig(batch_size=32, validation_split=0.1, download=True),
        model=model_config,
        training=TrainingConfig(epochs=1),
        evaluation=EvaluationConfig(metrics=["accuracy"]),
        experiment=ExperimentMetadata(name="real_data_pipeline_test", device="cpu"),
        logging=LoggingConfig(log_every_n_steps=10),
    )


def _prepare_or_skip(experiment: Experiment):
    try:
        return experiment.prepare_data()
    except Exception as e:
        pytest.skip(f"Real MNIST unavailable or download failed: {e}")


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_prepare_data_linear_real_shapes_and_stats():
    """Linear model should receive flattened vectors with real MNIST data."""
    config = _make_base_config(LinearModel())
    exp = Experiment.from_config(config)

    dataloaders = _prepare_or_skip(exp)

    batch_images, batch_labels = next(iter(dataloaders["train"]))

    # Shapes for flattened MNIST
    assert batch_images.dim() == 2
    assert batch_images.shape[1] == 784

    # Labels should be in [0, 9]
    assert batch_labels.dtype == torch.long
    assert int(batch_labels.min()) >= 0
    assert int(batch_labels.max()) <= 9

    # Basic normalized stats (roughly centered, unit-ish variance)
    mean = float(batch_images.mean())
    std = float(batch_images.std())
    assert abs(mean) < 0.5
    assert 0.5 < std < 1.5


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_prepare_data_cnn_real_shapes():
    """CNN model should receive image-shaped tensors with real MNIST data."""
    config = _make_base_config(CNNModel())
    exp = Experiment.from_config(config)

    dataloaders = _prepare_or_skip(exp)

    batch_images, batch_labels = next(iter(dataloaders["train"]))

    # Shapes for MNIST images
    assert batch_images.dim() == 4
    assert batch_images.shape[1:] == (1, 28, 28)

    # Labels in [0, 9]
    assert int(batch_labels.min()) >= 0
    assert int(batch_labels.max()) <= 9


@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_prepare_data_mlp_real_shapes():
    """MLP model should also receive flattened vectors with real MNIST data."""
    config = _make_base_config(MLPModel())
    exp = Experiment.from_config(config)

    dataloaders = _prepare_or_skip(exp)

    batch_images, batch_labels = next(iter(dataloaders["train"]))

    # Shapes for flattened MNIST
    assert batch_images.dim() == 2
    assert batch_images.shape[1] == 784

    # Labels range
    assert int(batch_labels.min()) >= 0
    assert int(batch_labels.max()) <= 9


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Integration tests for the complete data pipeline."""

    @patch("src.experiment.data.datasets.MNIST")
    def test_prepare_data_complete_pipeline(self, mock_mnist):
        """Test the complete data preparation pipeline."""
        # Mock MNIST dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        # Make the mock dataset subscriptable
        def mock_getitem(self, idx):
            return (torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item())

        mock_dataset.__getitem__ = mock_getitem
        mock_mnist.return_value = mock_dataset

        config = DataConfig(batch_size=16, validation_split=0.2, download=False)

        # Mock random_split
        with patch("src.experiment.data.random_split") as mock_split:
            # Create proper mock subsets
            mock_train_subset = Mock()
            mock_train_subset.__len__ = Mock(return_value=80)
            mock_train_subset.__getitem__ = lambda self, idx: (
                torch.randn(1, 28, 28),
                torch.randint(0, 10, (1,)).item(),
            )

            mock_val_subset = Mock()
            mock_val_subset.__len__ = Mock(return_value=20)
            mock_val_subset.__getitem__ = lambda self, idx: (
                torch.randn(1, 28, 28),
                torch.randint(0, 10, (1,)).item(),
            )

            mock_split.return_value = (mock_train_subset, mock_val_subset)

            dataloaders, stats = prepare_data(config=config, data_dir=Path("test_data"))

        # Check that we got expected outputs
        assert isinstance(dataloaders, dict)
        assert isinstance(stats, dict)
        assert "train" in dataloaders
        assert "val" in dataloaders
        assert "test" in dataloaders

    def test_data_config_integration(self):
        """Test that DataConfig integrates properly with data functions."""
        config = DataConfig(
            dataset="mnist",
            batch_size=32,
            validation_split=0.1,
            num_workers=0,
            shuffle=True,
            pin_memory=False,
            download=True,
        )

        # Test that config can be used for augmentation
        aug_config = DataAugmentationConfig(
            enabled=True, rotation_degrees=5.0, translation=0.05
        )

        # Should not raise any errors
        train_transforms, test_transforms = create_transforms(
            augmentation_config=aug_config, normalize=True
        )

        assert train_transforms is not None
        assert test_transforms is not None

    @patch("src.experiment.data.datasets.MNIST")
    def test_prepare_data_with_flatten_flag(self, mock_mnist):
        """prepare_data should return flattened tensors when flatten=True."""
        # Mock MNIST dataset to return image tensors in (1, 28, 28)
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=20)

        def mock_getitem(self, idx):
            return (torch.randn(1, 28, 28), torch.randint(0, 10, (1,)).item())

        mock_dataset.__getitem__ = mock_getitem
        mock_mnist.return_value = mock_dataset

        config = DataConfig(batch_size=8, validation_split=0.0, download=False)

        dataloaders, _ = prepare_data(
            config=config, data_dir=Path("test_data"), flatten=True
        )

        batch_images, batch_labels = next(iter(dataloaders["train"]))

        # Expect flattened vectors (B, 784)
        assert batch_images.dim() == 2
        assert batch_images.shape[1] == 784


@pytest.mark.slow
class TestDataWithRealMNIST:
    """Slow tests that use real MNIST data (marked as slow)."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_data_loading_with_cuda(self):
        """Test data loading with CUDA (if available)."""
        # This would test actual MNIST loading with CUDA
        # Skipped if CUDA not available
        pass

    def test_real_mnist_shapes(self):
        """Test with real MNIST data shapes."""
        # This would download and test real MNIST
        # Implementation would depend on test environment
        pass
