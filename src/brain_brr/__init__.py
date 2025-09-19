"""Brain-Go-Brr: Bi-Mamba-2 + U-Net + ResCNN for TUSZ seizure detection.

First architecture to combine:
- Bidirectional Mamba-2 for O(N) sequence modeling
- U-Net for multi-scale feature extraction
- ResCNN for temporal convolution
Specifically optimized for TUSZ seizure detection.
"""

__version__ = "2.0.0-alpha"

# Clean imports from new package structure
from .constants import *  # noqa: F403
from .models import *  # noqa: F403
