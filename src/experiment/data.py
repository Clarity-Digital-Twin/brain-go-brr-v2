"""MNIST data loading and preprocessing utilities.

This module handles all aspects of MNIST dataset management including:
- Dataset downloading and loading
- Preprocessing and normalization
- Train/validation splits
- Data augmentation
- PyTorch DataLoader creation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from .schemas import DataAugmentationConfig, DataConfig

logger = logging.getLogger(__name__)


class MNISTDataset(Dataset):
    """Custom MNIST dataset with optional preprocessing.

    Wraps torchvision MNIST dataset with additional preprocessing capabilities
    and shape validation for different model types.
    """

    def __init__(
        self,
        root: Path,
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        flatten: bool = False,
        download: bool = True,
    ):
        """Initialize MNIST dataset.

        Args:
            root: Root directory for dataset storage
            train: Whether to load training or test set
            transform: Optional torchvision transforms
            flatten: Whether to flatten images to 1D (for linear models)
            download: Whether to download dataset if not found
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        self.dataset = datasets.MNIST(
            root=str(self.root),
            train=train,
            download=download,
            transform=transforms.ToTensor(),  # Always convert to tensor first
        )

        self.transform = transform
        self.flatten = flatten

        logger.info(
            f"Loaded MNIST {'train' if train else 'test'} set: {len(self.dataset)} samples"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image, label) tensors
        """
        image, label = self.dataset[idx]

        # Apply additional transforms if provided
        if self.transform:
            image = self.transform(image)

        # Flatten for linear models if requested
        if self.flatten:
            image = image.view(-1)

        return image, torch.tensor(label, dtype=torch.long)

    def get_sample_shape(self) -> Tuple[int, ...]:
        """Get the shape of a single sample (excluding batch dimension).

        Returns:
            Shape tuple of a single sample
        """
        sample_image, _ = self[0]
        return tuple(sample_image.shape)


def create_transforms(
    augmentation_config: Optional[DataAugmentationConfig] = None, normalize: bool = True
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train and test transforms.

    Args:
        augmentation_config: Optional data augmentation configuration
        normalize: Whether to normalize images to [0, 1] range

    Returns:
        Tuple of (train_transforms, test_transforms)
    """
    # Base transforms (always applied)
    base_transforms = []
    if normalize:
        base_transforms.append(
            transforms.Normalize((0.1307,), (0.3081,))
        )  # MNIST stats

    # Test transforms (no augmentation)
    test_transforms = transforms.Compose(base_transforms)

    # Train transforms (with optional augmentation)
    train_transform_list = []

    if augmentation_config and augmentation_config.enabled:
        # Add augmentation transforms
        if augmentation_config.rotation_degrees > 0:
            train_transform_list.append(
                transforms.RandomRotation(degrees=augmentation_config.rotation_degrees)
            )

        if augmentation_config.translation > 0:
            train_transform_list.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(
                        augmentation_config.translation,
                        augmentation_config.translation,
                    ),
                )
            )

        if augmentation_config.scale != [1.0, 1.0]:
            train_transform_list.append(
                transforms.RandomAffine(
                    degrees=0, scale=tuple(augmentation_config.scale)
                )
            )

    # Add base transforms
    train_transform_list.extend(base_transforms)
    train_transforms = transforms.Compose(train_transform_list)

    logger.info(
        f"Created transforms - Train: {len(train_transform_list)}, Test: {len(base_transforms)}"
    )

    return train_transforms, test_transforms


def create_datasets(
    config: DataConfig,
    augmentation_config: Optional[DataAugmentationConfig] = None,
    data_dir: Path = Path("data/raw"),
    *,
    flatten: bool = False,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create train, validation, and test datasets.

    Args:
        config: Data configuration
        augmentation_config: Optional augmentation configuration
        data_dir: Directory to store raw data

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create transforms
    train_transforms, test_transforms = create_transforms(
        augmentation_config=augmentation_config, normalize=True
    )

    # Load full training set
    full_train_dataset = MNISTDataset(
        root=data_dir,
        train=True,
        transform=train_transforms,
        flatten=flatten,
        download=config.download,
    )

    # Load test set
    test_dataset = MNISTDataset(
        root=data_dir,
        train=False,
        transform=test_transforms,
        flatten=flatten,
        download=config.download,
    )

    # Split training set into train/validation
    if config.validation_split > 0:
        total_train = len(full_train_dataset)
        val_size = int(total_train * config.validation_split)
        train_size = total_train - val_size

        train_dataset, val_dataset = random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),  # Reproducible split
        )

        logger.info(f"Split training data: {train_size} train, {val_size} validation")
    else:
        train_dataset = full_train_dataset
        val_dataset = None
        logger.info("No validation split - using full training set")

    logger.info(
        f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    datasets: Tuple[Dataset, Optional[Dataset], Dataset], config: DataConfig
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for all dataset splits.

    Args:
        datasets: Tuple of (train, val, test) datasets
        config: Data configuration

    Returns:
        Dictionary mapping split names to DataLoaders
    """
    train_dataset, val_dataset, test_dataset = datasets

    dataloaders = {}

    # Training dataloader (with shuffling)
    dataloaders["train"] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    # Validation dataloader (no shuffling)
    if val_dataset:
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        )

    # Test dataloader (no shuffling)
    dataloaders["test"] = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    logger.info(f"Created {len(dataloaders)} dataloaders")
    return dataloaders


def get_dataset_stats(
    datasets: Tuple[Dataset, Optional[Dataset], Dataset],
) -> Dict[str, Dict[str, float]]:
    """Compute basic statistics for each dataset split.

    Args:
        datasets: Tuple of (train, val, test) datasets

    Returns:
        Dictionary with statistics for each split
    """
    train_dataset, val_dataset, test_dataset = datasets
    stats = {}

    def compute_stats(dataset: Dataset, name: str) -> Dict[str, float]:
        """Compute statistics for a single dataset."""
        if len(dataset) == 0:
            return {}

        # Sample a few batches to compute statistics
        loader = DataLoader(dataset, batch_size=min(1000, len(dataset)), shuffle=False)
        images, labels = next(iter(loader))

        return {
            "num_samples": len(dataset),
            "image_mean": float(images.mean()),
            "image_std": float(images.std()),
            "image_min": float(images.min()),
            "image_max": float(images.max()),
            "num_classes": len(torch.unique(labels)),
            "label_distribution": dict(
                zip(torch.unique(labels).tolist(), torch.bincount(labels).tolist())
            ),
        }

    stats["train"] = compute_stats(train_dataset, "train")
    if val_dataset:
        stats["val"] = compute_stats(val_dataset, "val")
    stats["test"] = compute_stats(test_dataset, "test")

    return stats


def prepare_data(
    config: DataConfig,
    augmentation_config: Optional[DataAugmentationConfig] = None,
    data_dir: Path = Path("data/raw"),
    *,
    flatten: bool = False,
) -> Tuple[Dict[str, DataLoader], Dict[str, Dict[str, float]]]:
    """Complete data preparation pipeline.

    This is the main function that orchestrates the entire data loading process.

    Args:
        config: Data configuration
        augmentation_config: Optional augmentation configuration
        data_dir: Directory for raw data storage

    Returns:
        Tuple of (dataloaders_dict, dataset_stats)
    """
    logger.info("Starting data preparation pipeline")

    # Create datasets
    datasets = create_datasets(
        config=config,
        augmentation_config=augmentation_config,
        data_dir=data_dir,
        flatten=flatten,
    )

    # Create dataloaders
    dataloaders = create_dataloaders(datasets, config)

    # Compute statistics
    stats = get_dataset_stats(datasets)

    logger.info("Data preparation completed successfully")

    return dataloaders, stats


def validate_data_shape(
    dataloaders: Dict[str, DataLoader], expected_shape: Tuple[int, ...]
) -> bool:
    """Validate that data has expected shape for the model.

    Args:
        dataloaders: Dictionary of dataloaders
        expected_shape: Expected shape (excluding batch dimension)

    Returns:
        True if shapes match, False otherwise
    """
    try:
        # Get a sample from training data
        train_loader = dataloaders["train"]
        batch_images, batch_labels = next(iter(train_loader))

        # Check image shape (excluding batch dimension)
        actual_shape = tuple(batch_images.shape[1:])

        if actual_shape != expected_shape:
            logger.error(
                f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
            )
            return False

        logger.info(f"Data shape validation passed: {actual_shape}")
        return True

    except Exception as e:
        logger.error(f"Data shape validation failed: {e}")
        return False


# Utility functions for common data operations


def flatten_for_linear(image: torch.Tensor) -> torch.Tensor:
    """Flatten image tensor for linear models.

    Args:
        image: Image tensor of shape (C, H, W) or (B, C, H, W)

    Returns:
        Flattened tensor of shape (C*H*W,) or (B, C*H*W)
    """
    if image.dim() == 3:  # Single image (C, H, W)
        return image.view(-1)
    elif image.dim() == 4:  # Batch (B, C, H, W)
        return image.view(image.size(0), -1)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {image.dim()}D")


def compute_class_weights(dataset: Dataset) -> torch.Tensor:
    """Compute class weights for imbalanced datasets.

    Args:
        dataset: Dataset to analyze

    Returns:
        Tensor of class weights for loss functions
    """
    # Count labels in dataset
    labels = []
    loader = DataLoader(dataset, batch_size=1000, shuffle=False)

    for _, batch_labels in loader:
        labels.extend(batch_labels.tolist())

    labels = torch.tensor(labels)
    class_counts = torch.bincount(labels)

    # Compute inverse frequency weights
    total_samples = len(labels)
    num_classes = len(class_counts)
    class_weights = total_samples / (num_classes * class_counts.float())

    logger.info(f"Computed class weights: {class_weights}")
    return class_weights


__all__ = [
    "MNISTDataset",
    "create_transforms",
    "create_datasets",
    "create_dataloaders",
    "prepare_data",
    "validate_data_shape",
    "flatten_for_linear",
    "compute_class_weights",
    "get_dataset_stats",
]
