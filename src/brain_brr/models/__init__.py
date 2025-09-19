"""Neural network models for seizure detection.

Split into focused modules:
- layers.py: Basic building blocks (ConvBlock)
- unet.py: UNetEncoder and UNetDecoder
- rescnn.py: ResCNNBlock and ResCNNStack
- mamba.py: BiMamba2Layer and BiMamba2
- detector.py: SeizureDetector (full model)
"""

# Import from split modules (which proxy to full_backup for now)
from .detector import SeizureDetector
from .layers import ConvBlock
from .mamba import MAMBA_AVAILABLE, BiMamba2, BiMamba2Layer
from .rescnn import ResCNNBlock, ResCNNStack
from .unet import UNetDecoder, UNetEncoder

__all__ = [
    "MAMBA_AVAILABLE",
    "BiMamba2",
    "BiMamba2Layer",
    "ConvBlock",
    "ResCNNBlock",
    "ResCNNStack",
    "SeizureDetector",
    "UNetDecoder",
    "UNetEncoder",
]