"""U-Net encoder and decoder components for seizure detection.

Architecture:
    - U-Net provides hierarchical feature extraction with multi-scale skip connections
    - Encoder: Progressive downsampling [64→128→256→512] captures different scales
    - Decoder: Progressive upsampling with skip fusion preserves fine details
    - Total 16x downsampling balances local detail vs global context
"""

from typing import cast

import torch
import torch.nn as nn

from .layers import ConvBlock


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
            "skip_usage_order": "skips[3] -> skips[2] -> skips[1] -> skips[0]",
            "expected_skip_shapes": [
                (64, 15360),
                (128, 7680),
                (256, 3840),
                (512, 1920),
            ],
        }
        return info


__all__ = ["UNetDecoder", "UNetEncoder"]