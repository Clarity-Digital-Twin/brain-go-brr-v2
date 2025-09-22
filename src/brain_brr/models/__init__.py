"""Neural network models for seizure detection.

Split into focused modules:
- layers.py: Basic building blocks (ConvBlock)
- tcn.py: TCNEncoder and ProjectionHead
- mamba.py: BiMamba2Layer and BiMamba2
- detector.py: SeizureDetector (full model)
"""

from .detector import SeizureDetector
from .layers import ConvBlock
from .mamba import MAMBA_AVAILABLE, BiMamba2, BiMamba2Layer
from .tcn import ProjectionHead, TCNEncoder

__all__ = [
    "MAMBA_AVAILABLE",
    "BiMamba2",
    "BiMamba2Layer",
    "ConvBlock",
    "ProjectionHead",
    "SeizureDetector",
    "TCNEncoder",
]
