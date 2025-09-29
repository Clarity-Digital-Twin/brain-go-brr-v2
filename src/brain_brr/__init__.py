"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

First architecture to combine:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling
- Projection + upsampling for output restoration
Specifically optimized for TUSZ seizure detection.
"""

__version__ = "3.2.0"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
