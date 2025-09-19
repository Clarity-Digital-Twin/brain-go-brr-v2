"""Training and validation pipeline.

DEPRECATED: This module has been moved to src.brain_brr.train
Please update your imports to use the new location:
- from src.brain_brr.train import train, train_epoch, validate_epoch, etc.
"""

# Import everything from new location for compatibility (imports first per E402)
import warnings

from src.brain_brr.train import *  # noqa: F403
from src.brain_brr.train.loop import EarlyStopping, create_balanced_sampler

warnings.warn(
    "Importing from 'src.experiment.pipeline' is deprecated. "
    "Please use 'from src.brain_brr.train import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)
