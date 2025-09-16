"""Main experiment pipeline with ExCa orchestration.

This module contains the core Experiment class that orchestrates the entire
machine learning pipeline, from data loading to model training and evaluation.
Uses ExCa for caching expensive operations and ensuring reproducibility.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import prepare_data, validate_data_shape
from .infra import ExperimentInfra, create_infra
from .models import create_model, create_optimizer, create_scheduler, get_model_summary
from .schemas import ExperimentConfig

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        """Check if training should stop early.

        Args:
            val_score: Current validation score (higher is better)

        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop


class Experiment:
    """Main experiment orchestrator using ExCa for caching and reproducibility.

    This class handles the complete ML pipeline:
    1. Data preparation (cached)
    2. Model creation and training
    3. Evaluation and metrics computation
    4. Result saving and artifact management
    """

    def __init__(
        self, config: ExperimentConfig, infra: Optional[ExperimentInfra] = None
    ):
        """Initialize experiment.

        Args:
            config: Validated experiment configuration
            infra: Optional pre-configured infrastructure (creates new if None)
        """
        self.config = config
        self.infra = infra or create_infra(
            cache_dir=config.experiment.cache_dir,
            experiment_name=config.experiment.name,
        )

        # Set up device
        self.device = self._setup_device()

        # Set random seeds for reproducibility
        self._set_random_seeds()

        # Initialize training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.training_history = []

        logger.info(f"Initialized experiment: {config.experiment.name}")
        logger.info(f"Using device: {self.device}")

    def _setup_device(self) -> torch.device:
        """Set up compute device based on configuration."""
        if self.config.experiment.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(self.config.experiment.device)

        return device

    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        seed = self.config.experiment.random_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "Experiment":
        """Factory method to create experiment from configuration."""
        return cls(config)

    def prepare_data(self) -> Dict[str, DataLoader]:
        """Prepare data with caching.

        Returns:
            Dictionary of dataloaders for train/val/test splits
        """

        @self.infra.cached_stage("data_preparation")
        def _prepare_data():
            logger.info("Preparing data...")

            # Determine if inputs should be flattened based on model type
            model_expects_flat = self._get_expected_input_shape() == (784,)

            # Prepare data using configuration
            dataloaders, stats = prepare_data(
                config=self.config.data,
                augmentation_config=self.config.data_augmentation,
                data_dir=Path("data/raw"),
                flatten=model_expects_flat,
            )

            # Validate data shapes match model expectations
            expected_shape = self._get_expected_input_shape()
            if not validate_data_shape(dataloaders, expected_shape):
                raise ValueError("Data shape validation failed")

            # Save dataset statistics
            self.infra.save_artifact(stats, "dataset_statistics")

            logger.info("Data preparation completed")
            return dataloaders

        # Include config in cache key context (without changing function signature)
        return _prepare_data(_cache_context=self.config.model_dump())

    def _get_expected_input_shape(self) -> Tuple[int, ...]:
        """Get expected input shape based on model type."""
        if self.config.model.name == "linear":
            return (784,)  # Flattened MNIST
        elif self.config.model.name == "mlp":
            return (784,)  # Flattened MNIST
        elif self.config.model.name == "cnn":
            return (1, 28, 28)  # MNIST image format
        else:
            raise ValueError(f"Unknown model type: {self.config.model.name}")

    def create_model(self) -> nn.Module:
        """Create and initialize model.

        Returns:
            Initialized PyTorch model
        """
        logger.info(f"Creating {self.config.model.name} model...")

        model = create_model(self.config.model)
        model = model.to(self.device)

        # Log model summary
        input_shape = self._get_expected_input_shape()
        summary = get_model_summary(model, input_shape)
        logger.info(summary)

        self.model = model
        return model

    def create_optimizer_and_scheduler(self):
        """Create optimizer and learning rate scheduler."""
        if self.model is None:
            raise ValueError("Model must be created before optimizer")

        self.optimizer = create_optimizer(self.model, self.config.training)
        self.scheduler = create_scheduler(self.optimizer, self.config.training)

        logger.info(f"Created {self.config.training.optimizer} optimizer")
        if self.scheduler and self.config.training.scheduler:
            logger.info(f"Created {self.config.training.scheduler.name} scheduler")

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train model for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(dataloader, desc="Training", leave=False)

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = nn.CrossEntropyLoss()(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Update progress bar
            if batch_idx % self.config.logging.log_every_n_steps == 0:
                current_acc = 100.0 * correct / total
                progress_bar.set_postfix(
                    {"Loss": f"{loss.item():.4f}", "Acc": f"{current_acc:.2f}%"}
                )

        metrics = {
            "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
            "accuracy": correct / total if total > 0 else 0.0,
        }

        return metrics

    def evaluate(
        self, dataloader: DataLoader, split_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate model on given dataset.

        Args:
            dataloader: Data loader for evaluation
            split_name: Name of the data split (for logging)

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in tqdm(
                dataloader, desc=f"Evaluating {split_name}", leave=False
            ):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                total_loss += loss.item()

                pred = output.argmax(dim=1)
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Compute metrics
        metrics = {"loss": total_loss / len(dataloader)}

        # Add requested metrics
        for metric in self.config.evaluation.metrics:
            if metric == "accuracy":
                metrics["accuracy"] = accuracy_score(all_targets, all_predictions)
            elif metric == "f1_macro":
                metrics["f1_macro"] = f1_score(
                    all_targets, all_predictions, average="macro"
                )
            elif metric == "f1_micro":
                metrics["f1_micro"] = f1_score(
                    all_targets, all_predictions, average="micro"
                )
            elif metric == "precision":
                metrics["precision"] = precision_score(
                    all_targets, all_predictions, average="macro"
                )
            elif metric == "recall":
                metrics["recall"] = recall_score(
                    all_targets, all_predictions, average="macro"
                )

        # Save predictions if requested
        if self.config.evaluation.save_predictions:
            predictions_data = {
                "predictions": all_predictions,
                "targets": all_targets,
                "split": split_name,
            }
            self.infra.save_artifact(predictions_data, f"predictions_{split_name}")

        return metrics

    def train(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, List[float]]:
        """Complete training loop with validation.

        Args:
            dataloaders: Dictionary containing train/val/test dataloaders

        Returns:
            Dictionary of training history
        """
        train_loader = dataloaders["train"]
        val_loader = dataloaders.get("val")

        # Initialize early stopping
        early_stopping = None
        if self.config.training.early_stopping:
            early_stopping = EarlyStopping(
                patience=self.config.training.early_stopping.patience,
                min_delta=self.config.training.early_stopping.min_delta,
            )

        # Training history
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_val_accuracy = 0.0
        best_model_state = None

        logger.info(f"Starting training for {self.config.training.epochs} epochs...")

        for epoch in range(self.config.training.epochs):
            epoch_start_time = time.time()

            # Training
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])

            # Validation
            if val_loader:
                val_metrics = self.evaluate(val_loader, "validation")
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])

                # Save best model
                if val_metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = val_metrics["accuracy"]
                    best_model_state = self.model.state_dict().copy()

                # Learning rate scheduling
                if self.scheduler:
                    if hasattr(self.scheduler, "step"):
                        if isinstance(
                            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            self.scheduler.step(val_metrics["loss"])
                        else:
                            self.scheduler.step()

                # Early stopping check
                if early_stopping and early_stopping(val_metrics["accuracy"]):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

                # Logging
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.training.epochs}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )
            else:
                # No validation set
                epoch_time = time.time() - epoch_start_time
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.training.epochs}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.4f}, "
                    f"Time: {epoch_time:.2f}s"
                )

                # Update best model based on training accuracy
                if train_metrics["accuracy"] > best_val_accuracy:
                    best_val_accuracy = train_metrics["accuracy"]
                    best_model_state = self.model.state_dict().copy()

        # Restore best model
        if best_model_state and self.config.experiment.save_best_only:
            self.model.load_state_dict(best_model_state)
            logger.info(
                f"Restored best model with validation accuracy: {best_val_accuracy:.4f}"
            )

        self.training_history = history
        return history

    def run(self) -> Dict[str, Any]:
        """Run complete experiment pipeline.

        Returns:
            Dictionary with experiment results
        """
        start_time = time.time()
        logger.info(f"Starting experiment: {self.config.experiment.name}")

        try:
            # 1. Prepare data
            dataloaders = self.prepare_data()

            # 2. Create model
            model = self.create_model()

            # 3. Create optimizer and scheduler
            self.create_optimizer_and_scheduler()

            # 4. Train model
            training_history = self.train(dataloaders)

            # 5. Final evaluation
            results = {}
            for split_name, dataloader in dataloaders.items():
                if split_name in ["train", "test"] or (
                    split_name == "val" and dataloader is not None
                ):
                    metrics = self.evaluate(dataloader, split_name)
                    results[f"{split_name}_metrics"] = metrics
                    logger.info(f"{split_name.capitalize()} metrics: {metrics}")

            # 6. Save artifacts
            total_time = time.time() - start_time

            # Save model if requested
            if self.config.experiment.save_model:
                model_path = self.infra.save_model(
                    model=self.model,
                    name=f"model_{self.config.experiment.name}",
                    metrics=results.get("test_metrics", {}),
                )
                results["model_path"] = str(model_path)

            # Save training history
            self.infra.save_artifact(training_history, "training_history")

            # Save final results
            final_results = {
                "config": self.config.model_dump(),
                "results": results,
                "training_history": training_history,
                "experiment_time": total_time,
                "device": str(self.device),
            }

            results_path = self.infra.save_artifact(final_results, "experiment_results")
            self.infra.save_metrics(results.get("test_metrics", {}), "final")

            # Record complete run
            run_path = self.infra.record_run(
                config=self.config.model_dump(), results=final_results
            )

            logger.info(f"Experiment completed in {total_time:.2f}s")
            logger.info(f"Results saved to: {results_path}")

            return {
                **results,
                "experiment_time": total_time,
                "results_path": str(results_path),
                "run_record": str(run_path),
            }

        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            # Save error information
            error_info = {
                "error": str(e),
                "config": self.config.model_dump(),
                "timestamp": time.time(),
            }
            self.infra.save_artifact(error_info, "experiment_error")
            raise


# ---- Utility Functions ----


def run_experiment_from_config(config_path: Path) -> Dict[str, Any]:
    """Run experiment from configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Experiment results dictionary
    """
    from .schemas import load_config

    config = load_config(config_path)
    experiment = Experiment.from_config(config)
    return experiment.run()


def compare_experiments(experiment_paths: List[Path]) -> Dict[str, Any]:
    """Compare results from multiple experiments.

    Args:
        experiment_paths: List of paths to experiment result files

    Returns:
        Comparison summary
    """
    comparison = {"experiments": {}, "summary": {}}

    for exp_path in experiment_paths:
        # Load experiment results
        # This would need to be implemented based on how results are stored
        pass

    return comparison


__all__ = [
    "Experiment",
    "EarlyStopping",
    "run_experiment_from_config",
    "compare_experiments",
]
