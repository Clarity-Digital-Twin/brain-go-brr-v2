"""Early stopping handler for training.

Monitors validation metrics and stops training when no improvement is seen.
"""

from __future__ import annotations

from src.brain_brr.config.schemas import EarlyStoppingConfig


class EarlyStopping:
    """Early stopping handler.

    Encapsulates early stopping logic.
    """

    def __init__(self, config: EarlyStoppingConfig) -> None:
        self.patience = config.patience
        self.min_epochs = config.min_epochs
        self.metric = config.metric
        self.mode = config.mode
        self.best_score = float("-inf") if self.mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int = 0) -> bool:
        """Check if should stop.

        Args:
            score: Current metric value
            epoch: Current epoch (0-indexed)

        Returns:
            True if should stop
        """
        # Don't allow early stopping before min_epochs
        if epoch < self.min_epochs:
            # Still track best score for later
            improved = score > self.best_score if self.mode == "max" else score < self.best_score
            if improved:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0
            return False

        improved = score > self.best_score if self.mode == "max" else score < self.best_score

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience

    def state_dict(self) -> dict:
        """Get early stopping state for checkpointing.

        Returns:
            Dictionary containing early stopping state
        """
        return {
            "best_score": self.best_score,
            "counter": self.counter,
            "best_epoch": self.best_epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load early stopping state from checkpoint.

        Args:
            state: Dictionary containing early stopping state
        """
        self.best_score = state["best_score"]
        self.counter = state["counter"]
        self.best_epoch = state["best_epoch"]
