"""Configuration schemas for experiment management.

DEPRECATED: This module has been moved to src.brain_brr.config.schemas
Please update your imports to use the new location:
- from src.brain_brr.config.schemas import ExperimentConfig, ModelConfig, etc.
"""

# Import everything from new location for compatibility (imports first per E402)
from src.brain_brr.config.schemas import *  # noqa: F403

import warnings

warnings.warn(
    "Importing from 'src.experiment.schemas' is deprecated. "
    "Please use 'from src.brain_brr.config.schemas import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)