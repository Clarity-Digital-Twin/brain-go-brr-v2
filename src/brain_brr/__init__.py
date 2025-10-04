"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.6.1: Gradient logging enhancement (ML 2025 best practices)
- Median-first logging (P50 emphasized over mean)
- IQR (Interquartile Range) for robust spread measurement
- Removed mean (outlier-sensitive) in favor of percentiles
- Fixed contradictory documentation examples
- Modal image forced rebuild for latest gradient logging code
"""

__version__ = "3.6.1"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
