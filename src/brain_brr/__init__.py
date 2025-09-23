"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

First architecture to combine:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling
- Projection + upsampling for output restoration
Specifically optimized for TUSZ seizure detection.
"""

__version__ = "2.3.0"

# Clean imports from new package structure
from .constants import *  # noqa: F403
from .models import *  # noqa: F403
