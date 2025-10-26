"""Brain-Go-Brr: Dual SSM Stacks for TUSZ Seizure Detection.

V3 dual-stream architecture with TWO production stacks:
- BiMamba2 (Baseline): TCN + Bidirectional Mamba-2 + GNN + Dynamic LPE
- FLA (Research): TCN + BiGatedDeltaNet + GNN + Dynamic LPE

Both achieve O(N) complexity with state-space models and graph neural networks.

v4.2.0: Checkpoint Resume Bug Fixes (PATCH RELEASE)
- CRITICAL FIX: Early stopping state now persisted in all checkpoints
- FIX: CLI --resume flag preserves YAML defaults (no longer clobbers config)
- FIX: RNG seeding moved before DataLoader creation (proper determinism)
- ENHANCEMENT: Backward compatibility for old checkpoints (seeds from best_metric)
- Training now resumable with full state preservation (model, optimizer, scheduler, RNG, early stopping)
"""

__version__ = "4.2.0"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
