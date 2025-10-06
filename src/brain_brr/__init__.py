"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.8.2: Zero Warnings - Professional Fixes
- Eliminates read-only tensor warnings at source (np.array copy pattern in all 3 datasets)
- Fixes GradScaler + LRScheduler interaction (proper scale tracking, 2 locations)
- Training logs now 100% clean with accurate LR schedule
- Follows official PyTorch AMP best practices
"""

__version__ = "3.8.2"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
