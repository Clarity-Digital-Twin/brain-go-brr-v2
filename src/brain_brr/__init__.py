"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.9.2: CI/CD Stability & Documentation Cleanup
- Fixed O(n) event extraction validation (CI timeout eliminated)
- Professional FLA test skip logic (module-level flag check)
- Deterministic memory testing (array property validation)
- Dedicated FLA CI job for continuous code path validation
"""

__version__ = "3.9.2"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
