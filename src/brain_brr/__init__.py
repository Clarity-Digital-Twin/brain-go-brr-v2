"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.9.0: Production Training Baseline – Bulletproof Resume
- Atomic checkpoint saves (temp + fsync + rename) with AMP scaler & RNG capture
- Wall-clock timeout guard exits ~23 h with timeout_exit.pt (Modal-friendly resume)
- Metric key normalization + W&B run persistence for seamless dashboards
- Documentation refreshed for Modal operations / checkpoint strategy
"""

__version__ = "3.9.0"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
