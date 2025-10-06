"""Brain-Go-Brr: TCN + Bi-Mamba-2 for TUSZ seizure detection.

V3 dual-stream architecture combining:
- TCN (Temporal Convolutional Network) for efficient feature extraction
- Bidirectional Mamba-2 for O(N) sequence modeling with state-space models
- GNN (Graph Neural Network) with dynamic Laplacian positional encoding
- Multi-head gated fusion for node/edge stream combination

v3.8.1: Complete Tensor Safety (All Datasets)
- Completes P0-2 tensor safety by adding .clone() to EEGWindowDataset (missed in v3.8.0)
- Removes paper-over broad warning suppression from train_step.py
- Verifies scheduler step order is correct (no fix needed, cosmetic warning only)
- All THREE datasets now safe from read-only tensor undefined behavior
- TECHNICAL_DEBT.md updated with truth (verified correct, not "fixed")
"""

__version__ = "3.8.1"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
