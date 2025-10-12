"""Brain-Go-Brr: Dual SSM Stacks for TUSZ Seizure Detection.

V3 dual-stream architecture with TWO production stacks:
- BiMamba2 (Baseline): TCN + Bidirectional Mamba-2 + GNN + Dynamic LPE
- FLA (Research): TCN + BiGatedDeltaNet + GNN + Dynamic LPE

Both achieve O(N) complexity with state-space models and graph neural networks.

v4.0.0: FLA Production + WSL2 Fix (MAJOR RELEASE)
- FLA (BiGatedDeltaNet) now production-ready, training on local RTX 4090
- WSL2 SIGBUS fix: cache must be on native ext4 filesystem (not Windows drives)
- Dual production stacks training simultaneously for empirical A/B comparison
- Research milestone: Both BiMamba2 and FLA validated on full TUSZ dataset
"""

__version__ = "4.0.0"

# NO HEAVY IMPORTS AT PACKAGE LEVEL
# Models should be imported explicitly when needed:
#   from src.brain_brr.models.detector import SeizureDetector
# This avoids importing torch/mamba when just accessing utilities
