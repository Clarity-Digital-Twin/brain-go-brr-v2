"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.7.0: Zero Debt Modal Baseline - FINAL release before A100 training
- Complete P2/P3 technical debt elimination (0 blockers)
- Constants cleanup: 90→84 total, 58 used (69% utilization), 26 documented reserves
- Type safety: 21→17 ignores (all third-party/dynamic, documented)
- Production robustness: All assertions → proper exceptions
- Documentation: 100% accuracy, configs/code perfectly aligned
- Test suite: 40 tests, 82.88% coverage, all passing
"""

__version__ = "3.7.0"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
