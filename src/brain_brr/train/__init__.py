"""Training utilities and pipeline.

This module will contain:
- loop.py: Training/validation loop
- losses.py: Loss functions
- optim.py: Optimizers and schedulers
- sampler.py: Balanced sampling
- early_stopping.py: Early stopping logic
- checkpoints.py: Model save/load
"""

# During migration, re-export from experiment
try:
    from src.experiment.pipeline import (
        create_optimizer,
        create_scheduler,
        load_checkpoint,
        save_checkpoint,
        train,
        train_epoch,
        validate_epoch,
    )

    __all__ = [
        "create_optimizer",
        "create_scheduler",
        "load_checkpoint",
        "save_checkpoint",
        "train",
        "train_epoch",
        "validate_epoch",
    ]
except ImportError:
    # Clean-slate imports will go here after migration
    pass
