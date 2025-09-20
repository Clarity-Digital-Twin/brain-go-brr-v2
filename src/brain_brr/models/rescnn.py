"""Residual CNN modules for multi-scale feature extraction.

Multi-scale temporal modeling for seizure dynamics:
    - Fast (10-80 Hz): Spike detection via multi-kernel [3,5,7]
    - Captures nonlinear chaotic seizure evolution that SSMs miss
    - Residual connections maintain gradient flow through deep stack
"""

from typing import cast

import torch
import torch.nn as nn


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
        )

        # Apply dropout separately to control its placement
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
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
        # Apply dropout before residual connection for stronger effect
        fused = self.dropout(fused)
        output = self.relu(fused + x)  # Residual connection

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
        first_block = cast(ResCNNBlock, self.blocks[0])
        max_kernel: int = max(first_block.kernel_sizes)
        return max_kernel + (self.num_blocks - 1) * (max_kernel - 1)


__all__ = ["ResCNNBlock", "ResCNNStack"]
