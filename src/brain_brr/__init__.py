"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.8.3: Manifest Naming Cleanup - Zero Technical Debt
- Eliminates NPZ-style naming from manifests (now uses *_data.npy directly)
- Removed all 11 .replace("_windows", "") string manipulation workarounds
- Simplified cache_utils.py and datasets.py for better maintainability
- Regenerated manifests: train (303,990 windows), dev (148,224 windows)
- Zero P0/P1/P2/P3 technical debt remaining
"""

__version__ = "3.8.3"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
