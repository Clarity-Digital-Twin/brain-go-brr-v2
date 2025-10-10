"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.11.0: StatefulDataLoader & Mid-Epoch Resume Fix
- Exact mid-epoch checkpoint resume with PyTorch StatefulDataLoader
- Eliminates 1-2 hours wasted compute per Modal restart ($150+ savings)
- Pydantic v2 warning fix using Annotated pattern
- Complete backward compatibility with old checkpoints
"""

__version__ = "3.11.0"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
