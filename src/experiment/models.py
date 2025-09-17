"""Neural architecture components for Bi-Mamba-2 + U-Net + ResCNN seizure detection.

Phase 2 implementation will land incrementally via TDD:
- UNetEncoder: 4-stage encoder with skip connections (implemented in Phase 2.1)
- ResCNNStack: Multi-scale residual CNN blocks (Phase 2.2)
- BiMamba2: Bidirectional Mamba-2 layers (Phase 2.3)
- UNetDecoder: 4-stage decoder with skip fusion (Phase 2.4)
- SeizureDetector: Full assembled model (Phase 2.5)
"""

import warnings
from typing import cast

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic convolutional building block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through conv -> batch norm -> ReLU.

        Args:
            x: Input tensor of shape (B, C_in, L)

        Returns:
            Activated output of shape (B, C_out, L)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UNetEncoder(nn.Module):
    """U-Net encoder with progressive downsampling and skip connections.

    Architecture:
        - Initial projection: 19 -> 64 channels
        - 4 encoder stages with channel doubling: [64, 128, 256, 512]
        - Each stage: double conv block + downsample
        - Total downsampling: x16 (15360 -> 960)
        - Skip connections saved before downsampling
    """

    def __init__(self, in_channels: int = 19, base_channels: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth

        # Channel progression: [64, 128, 256, 512]
        channels = [base_channels * (2**i) for i in range(depth)]

        # Initial projection from 19 -> 64 channels
        # Use kernel_size=7 as specified in docs
        self.input_conv = ConvBlock(in_channels, channels[0], kernel_size=7, padding=3)

        # Build encoder stages
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()

        for i in range(depth):
            # Channel progression: 64->64, 64->128, 128->256, 256->512
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]

            # Double convolution block with kernel_size=5 (matches schemas)
            self.encoder_blocks.append(
                nn.Sequential(
                    ConvBlock(in_ch, out_ch, kernel_size=5, padding=2),
                    ConvBlock(out_ch, out_ch, kernel_size=5, padding=2),
                )
            )

            # Downsample maintains channel count
            self.downsample.append(nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass through encoder.

        Args:
            x: Input EEG tensor of shape (B, 19, 15360)

        Returns:
            Tuple of:
                - encoded: Final encoding (B, 512, 960)
                - skips: List of 4 skip tensors for decoder:
                    [(B, 64, 15360), (B, 128, 7680), (B, 256, 3840), (B, 512, 1920)]
        """
        skips = []

        # Defensive check: input length must be divisible by 2**depth to downsample cleanly
        length = int(x.shape[-1])
        factor = 2**self.depth
        if length % factor != 0:
            raise ValueError(
                f"UNetEncoder expects input length divisible by {factor}; got L={length}. "
                "Ensure window size aligns with downsampling (e.g., 15360 for depth=4)."
            )

        # Initial projection
        x = self.input_conv(x)  # (B, 64, 15360)

        # Process through encoder stages
        for i in range(self.depth):
            # Process through encoder block
            x = self.encoder_blocks[i](x)

            # Save skip AFTER block, BEFORE downsample (standard U-Net pattern)
            skips.append(x)

            # Downsample for next stage
            x = self.downsample[i](x)

        # Final state: x is (B, 512, 960)
        # Skips are [(64,15360), (128,7680), (256,3840), (512,1920)]
        return x, skips

    def get_dimension_info(self) -> dict:
        """Get information about encoder dimensions for debugging.

        Returns:
            Dictionary with stage dimensions and channel counts
        """
        info = {
            "input_channels": 19,
            "base_channels": 64,
            "depth": self.depth,
            "channel_progression": [64 * (2**i) for i in range(self.depth)],
            "spatial_progression": [15360 // (2**i) for i in range(self.depth + 1)],
            "skip_shapes": [
                (64, 15360),
                (128, 7680),
                (256, 3840),
                (512, 1920),
            ],
            "output_shape": (512, 960),
        }
        return info


class ResCNNBlock(nn.Module):
    """Residual CNN block with multi-scale kernels for bottleneck processing.

    Splits 512 channels across 3 kernel sizes for multi-scale feature extraction.
    Uses residual connections to maintain gradient flow.
    """

    def __init__(
        self,
        channels: int = 512,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.channels = channels
        self.kernel_sizes = kernel_sizes

        # Multi-scale convolution branches
        # Split channels evenly across branches (170, 170, 172 for 512)
        branch_channels = channels // len(kernel_sizes)
        remainder = channels % len(kernel_sizes)

        self.branches = nn.ModuleList()
        channel_splits = []  # Track for validation

        for i, k in enumerate(kernel_sizes):
            # Add remainder channels to last branch
            out_ch = branch_channels + (remainder if i == len(kernel_sizes) - 1 else 0)
            channel_splits.append(out_ch)

            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(channels, out_ch, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        # Validate branch split invariance (not stripped under -O): raise clear error
        if sum(channel_splits) != channels:
            raise ValueError(
                f"ResCNN branch split {channel_splits} does not sum to input channels={channels}"
            )

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.Dropout1d(dropout),  # Channel-wise dropout for 1D signals
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through multi-scale branches with residual connection.

        Args:
            x: Input tensor (B, 512, 960) from encoder

        Returns:
            Output tensor (B, 512, 960) with residual connection
        """
        # Process through multi-scale branches
        branches_out = []
        for branch in self.branches:
            branches_out.append(branch(x))

        # Concatenate multi-scale features
        multi_scale = torch.cat(branches_out, dim=1)

        # Fusion and residual connection
        fused = self.fusion(multi_scale)
        output = self.relu(fused + x)  # Residual connection

        # Torch layers lack precise typing; cast to satisfy type checkers
        from typing import cast

        return cast(torch.Tensor, output)


class ResCNNStack(nn.Module):
    """Stack of ResCNN blocks for deep multi-scale feature extraction."""

    def __init__(
        self,
        channels: int = 512,
        num_blocks: int = 3,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.channels = channels
        self.num_blocks = num_blocks

        # Stack of ResCNN blocks
        self.blocks = nn.ModuleList(
            [ResCNNBlock(channels, kernel_sizes, dropout) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process through stacked ResCNN blocks.

        Args:
            x: Encoded features (B, 512, 960)

        Returns:
            Enhanced features (B, 512, 960)
        """
        for block in self.blocks:
            x = block(x)

        return x

    def get_receptive_field(self) -> int:
        """Calculate total receptive field of the stack.

        With kernel 7 and 3 blocks: 7 + 6 + 6 = 19 samples
        """
        # Access kernel_sizes attribute directly from first block
        from typing import cast

        first_block = cast(ResCNNBlock, self.blocks[0])
        max_kernel: int = max(first_block.kernel_sizes)
        return max_kernel + (self.num_blocks - 1) * (max_kernel - 1)


# ============================================================================
# Phase 2.3: Bidirectional Mamba-2 Components
# ============================================================================

# Conditional import for GPU/CPU compatibility
try:
    from mamba_ssm import Mamba2  # type: ignore[import-not-found]

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba-ssm not available; using Conv1d fallback for BiMamba2. "
        "Fallback is for shape validation only and is not functionally equivalent.",
        stacklevel=1,
    )


class BiMamba2Layer(nn.Module):
    """Single bidirectional Mamba-2 layer.

    Args:
        d_model: Feature dimension (matches encoder bottleneck)
        d_state: SSM state dimension
        d_conv: Conv kernel size (default 5 to match schemas/configs)
        expand: Expansion factor in Mamba component
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 5,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        if MAMBA_AVAILABLE:
            # Real Mamba-2 for GPU
            self.forward_mamba = Mamba2(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.backward_mamba = Mamba2(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
        else:
            # WARNING: Conv1d fallback for CPU testing only
            # This is NOT functionally equivalent to Mamba-2 SSM!
            # - Mamba uses complex state-space transitions with selective gates
            # - This fallback is a simple convolution for shape validation only
            # DO NOT use CPU tests to validate model convergence or accuracy
            warnings.warn(
                "Using Conv1d fallback for BiMamba2Layer — NOT equivalent to Mamba-2.",
                stacklevel=1,
            )
            self.forward_mamba = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
            self.backward_mamba = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)

        # Fusion and normalization
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input bidirectionally.

        Args:
            x: Input (B, L, D) where L=960, D=512

        Returns:
            Bidirectional output (B, L, D)
        """
        residual = x

        # Forward direction
        if MAMBA_AVAILABLE:
            x_forward = self.forward_mamba(x)
        else:
            # Conv1d expects (B, C, L)
            x_forward = self.forward_mamba(x.transpose(1, 2)).transpose(1, 2)

        # Backward direction (flip sequence)
        x_backward = x.flip(dims=[1])
        if MAMBA_AVAILABLE:
            x_backward = self.backward_mamba(x_backward)
        else:
            x_backward = self.backward_mamba(x_backward.transpose(1, 2)).transpose(1, 2)

        # Flip backward to align
        x_backward = x_backward.flip(dims=[1])

        # Concatenate bidirectional features
        x_combined = torch.cat([x_forward, x_backward], dim=-1)  # (B, L, 2D)

        # Project back to d_model
        x_output = self.output_proj(x_combined)  # (B, L, D)

        # Add residual and normalize
        output = self.layer_norm(residual + self.dropout(x_output))

        return cast(torch.Tensor, output)


class BiMamba2(nn.Module):
    """Stack of bidirectional Mamba-2 layers for O(N) temporal modeling.

    Processes sequences with linear complexity, avoiding the O(N²) cost
    of transformers on long EEG sequences.

    Args:
        d_model: Model dimension (512 for encoder bottleneck)
        d_state: SSM state dimension (16 default)
        d_conv: Temporal conv kernel (5 default)
        num_layers: Number of bidirectional layers (6 default)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 5,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Stack of bidirectional layers
        self.layers = nn.ModuleList(
            [
                BiMamba2Layer(d_model=d_model, d_state=d_state, d_conv=d_conv, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process features through bidirectional Mamba-2 stack.

        Args:
            x: Input features (B, C, L) where C=512, L=960

        Returns:
            Temporal output (B, C, L)
        """
        # Transpose for sequence processing: (B, L, C)
        x = x.transpose(1, 2)

        # Process through bidirectional layers
        for layer in self.layers:
            x = layer(x)

        # Transpose back: (B, C, L)
        return x.transpose(1, 2)

    def get_complexity(self) -> str:
        """Return complexity analysis."""
        if MAMBA_AVAILABLE:
            return "O(N) with Mamba-2 SSM"
        else:
            return "O(N) with Conv1d fallback"


# ============================================================================
# Phase 2.4: U-Net Decoder Components
# ============================================================================


class UNetDecoder(nn.Module):
    """U-Net decoder with skip connections and progressive upsampling.

    Architecture:
        - 4 decoder stages with channel halving: [512, 256, 128, 64]
        - Each stage: upsample + skip concat + double conv
        - Total upsampling: x16 (960 -> 15360)
        - Output projection: 64 -> 19 channels
    """

    def __init__(self, out_channels: int = 19, base_channels: int = 64, depth: int = 4):
        super().__init__()
        self.depth = depth

        # Channel progression (reversed): [512, 256, 128, 64]
        channels = [base_channels * (2 ** (depth - 1 - i)) for i in range(depth)]

        # Build decoder stages
        self.upsample = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        for i in range(depth):
            # Current stage channels
            in_ch = channels[i]
            out_ch = channels[i + 1] if i < depth - 1 else base_channels

            # Transposed convolution for x2 upsampling
            self.upsample.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2))

            # After concatenation with skip, we have out_ch + skip_ch channels
            # Skip channels match encoder output at each stage (before downsampling)
            # Stage 0 uses skip[3] (512 channels), Stage 1 uses skip[2] (256), etc.
            skip_idx = depth - 1 - i
            skip_ch = base_channels * (2**skip_idx)

            # Double convolution after skip concatenation
            self.decoder_blocks.append(
                nn.Sequential(
                    ConvBlock(out_ch + skip_ch, out_ch, kernel_size=5, padding=2),
                    ConvBlock(out_ch, out_ch, kernel_size=5, padding=2),
                )
            )

        # Output projection
        self.output_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass through decoder with skip connections.

        Args:
            x: Bottleneck features (B, 512, 960) from BiMamba2
            skips: Skip connections from encoder [4 tensors]
                   Order: [stage1, stage2, stage3, stage4] from encoder
                   Shapes: [(B,64,15360), (B,128,7680), (B,256,3840), (B,512,1920)]
                   Used in reverse: skip[3], skip[2], skip[1], skip[0]

        Returns:
            Decoded output (B, 19, 15360)
        """
        # Process decoder stages with skips (deepest → shallowest)
        for i in range(self.depth):
            # Upsample by 2x
            x = self.upsample[i](x)

            # Concatenate with corresponding skip (reverse order)
            # Decoder stage 0 uses skip[3] (deepest), stage 1 uses skip[2], etc.
            skip_idx = self.depth - 1 - i
            x = torch.cat([x, skips[skip_idx]], dim=1)

            # Process through decoder block
            x = self.decoder_blocks[i](x)

        # Final output projection to target channels
        return cast(torch.Tensor, self.output_conv(x))

    def check_skip_compatibility(self, skips: list[torch.Tensor]) -> bool:
        """Verify skip dimensions match expected shapes.

        Args:
            skips: List of skip tensors from encoder

        Returns:
            True if all skip dimensions are compatible
        """
        # Expected shapes for depth=4, base_channels=64
        expected_shapes = [
            (None, 64, 15360),  # Stage 1
            (None, 128, 7680),  # Stage 2
            (None, 256, 3840),  # Stage 3
            (None, 512, 1920),  # Stage 4
        ]

        if len(skips) != self.depth:
            return False

        for skip, expected in zip(skips, expected_shapes[: self.depth], strict=False):
            # Check channel and spatial dimensions (ignore batch)
            if skip.shape[1:] != expected[1:]:
                return False

        return True

    def get_dimension_info(self) -> dict:
        """Get information about decoder dimensions for debugging.

        Returns:
            Dictionary with stage dimensions and channel counts
        """
        info = {
            "output_channels": 19,
            "base_channels": 64,
            "depth": self.depth,
            "channel_progression": [512, 256, 128, 64],
            "spatial_progression": [960, 1920, 3840, 7680, 15360],
            "skip_usage_order": "skips[3] → skips[2] → skips[1] → skips[0]",
            "expected_skip_shapes": [
                (64, 15360),
                (128, 7680),
                (256, 3840),
                (512, 1920),
            ],
        }
        return info
