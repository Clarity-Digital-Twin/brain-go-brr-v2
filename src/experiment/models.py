"""Neural architecture components for Bi-Mamba-2 + U-Net + ResCNN seizure detection.

Phase 2 implementation will land incrementally via TDD:
- UNetEncoder: 4-stage encoder with skip connections (implemented in Phase 2.1)
- ResCNNStack: Multi-scale residual CNN blocks (Phase 2.2)
- BiMamba2: Bidirectional Mamba-2 layers (Phase 2.3)
- UNetDecoder: 4-stage decoder with skip fusion (Phase 2.4)
- SeizureDetector: Full assembled model (Phase 2.5)
"""

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

        # Runtime validation: ensure branches sum to input channels
        assert sum(channel_splits) == channels, (
            f"Branch channels {channel_splits} don't sum to {channels}"
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

        return output


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
        max_kernel = max(self.blocks[0].kernel_sizes)
        return max_kernel + (self.num_blocks - 1) * (max_kernel - 1)
