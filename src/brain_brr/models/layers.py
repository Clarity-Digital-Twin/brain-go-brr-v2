"""Basic neural network building blocks."""

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


__all__ = ["ConvBlock"]
