"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.5.0: Clean code refactoring following Uncle Bob principles
- Modular builders for component construction
- Pipeline decomposition for improved maintainability
- Service layer extraction for CLI orchestration
- 78% test coverage with comprehensive validation
"""

__version__ = "3.5.0"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
