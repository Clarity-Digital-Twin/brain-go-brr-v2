"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.8.0: Read-Only Mmap Baseline (Zero Debt)
- Finalized NPY mmap cache workflow with read-only datasets (fail-fast on cache miss)
- Modal automation: cache validation, stray NPZ cleanup, and health checks baked in
- Shared mmap loader eliminates 120 duplicate lines across datasets
- Type safety tightened (Console/WandB typing, lint clean)
- Documentation refreshed end-to-end for the mmap pipeline and Modal operations
- Test suite: 104 unit/integration tests + clinical suite, 83.8% coverage, all passing
"""

__version__ = "3.8.0"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
