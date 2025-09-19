"""Neural architecture components for seizure detection.

DEPRECATED: This module has been split and moved to src.brain_brr.models
Please update your imports to use the new locations:
- from src.brain_brr.models import SeizureDetector
- from src.brain_brr.models.unet import UNetEncoder, UNetDecoder
- from src.brain_brr.models.rescnn import ResCNNBlock, ResCNNStack
- from src.brain_brr.models.mamba import BiMamba2Layer, BiMamba2
"""

import warnings

warnings.warn(
    "Importing from 'src.experiment.models' is deprecated. "
    "Please use 'from src.brain_brr.models import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import everything from new location for compatibility
from src.brain_brr.models import *  # noqa: F403
