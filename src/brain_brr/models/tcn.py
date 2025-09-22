"""TCN (Temporal Convolutional Network) encoder for EEG seizure detection.

Replaces U-Net encoder/decoder + ResCNN with a modern TCN architecture.
Uses pytorch-tcn if available, falls back to minimal implementation.
"""

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import pytorch-tcn (optional dependency)
try:
    from pytorch_tcn import TCN as TemporalConvNet
    HAS_PYTORCH_TCN = True
except ImportError:
    HAS_PYTORCH_TCN = False
    warnings.warn(
        "pytorch-tcn not installed. Install with: uv sync --extra tcn\n"
        "Falling back to minimal TCN implementation."
    )


class MinimalTCN(nn.Module):
    """Minimal TCN implementation if pytorch-tcn is not available."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_channels: list[int],
        kernel_size: int = 7,
        dropout: float = 0.15,
        causal: bool = False,
    ):
        super().__init__()
        self.causal = causal

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            # Dilated convolution
            padding = (kernel_size - 1) * dilation_size if causal else (kernel_size - 1) * dilation_size // 2
            conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation_size
            )

            # Weight normalization
            conv = nn.utils.weight_norm(conv)

            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        # Final projection to output size
        self.projection = nn.Conv1d(num_channels[-1], output_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, L)

        Returns:
            Output tensor (B, output_size, L)
        """
        out = self.network(x)

        if self.causal:
            # Trim the output for causal convolution
            out = out[:, :, :x.size(2)]

        return self.projection(out)


class TCNEncoder(nn.Module):
    """TCN encoder that produces features for Bi-Mamba.

    Input:  (B, 19, 15360) - 60s of 19-channel EEG @ 256Hz
    Output: (B, 512, 960)   - Downsampled by 16x for Mamba
    """

    def __init__(
        self,
        input_channels: int = 19,
        output_channels: int = 512,
        num_layers: int = 8,
        kernel_size: int = 7,
        dropout: float = 0.15,
        causal: bool = False,
        stride_down: int = 16,
        use_cuda_optimizations: bool = True,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride_down = stride_down

        # Build channel list for TCN layers
        # Double the channels list to get enough layers
        base_channels = [64, 128, 256, 512]
        if num_layers > len(base_channels):
            num_channels = base_channels * (num_layers // len(base_channels) + 1)
            num_channels = num_channels[:num_layers]
        else:
            num_channels = base_channels[:num_layers]

        # CUDA optimizations
        if torch.cuda.is_available() and use_cuda_optimizations:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # TCN backbone
        if HAS_PYTORCH_TCN:
            # Use pytorch-tcn package
            self.tcn = TemporalConvNet(
                input_channels=input_channels,
                hidden_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                causal=causal
            )
            tcn_out_channels = num_channels[-1]
        else:
            # Use minimal fallback
            self.tcn = MinimalTCN(
                input_size=input_channels,
                output_size=num_channels[-1],
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                causal=causal
            )
            tcn_out_channels = num_channels[-1]

        # Projection to output channels
        self.channel_proj = nn.Conv1d(tcn_out_channels, output_channels, kernel_size=1)

        # Downsampling to match Mamba input size
        # 15360 / 16 = 960
        self.downsample = nn.Conv1d(
            output_channels,
            output_channels,
            kernel_size=stride_down,
            stride=stride_down
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN encoder.

        Args:
            x: Input EEG tensor (B, 19, 15360)

        Returns:
            Encoded features (B, 512, 960)
        """
        # Check input shape
        B, C, L = x.shape
        assert C == self.input_channels, f"Expected {self.input_channels} channels, got {C}"
        assert L == 15360, f"Expected 15360 samples, got {L}"

        # TCN processing
        x = self.tcn(x)  # (B, tcn_channels, 15360)

        # Project to output channels
        x = self.channel_proj(x)  # (B, 512, 15360)

        # Downsample for Mamba
        x = self.downsample(x)  # (B, 512, 960)

        return x


class ProjectionHead(nn.Module):
    """Projection + upsampling head to restore full temporal resolution.

    After Bi-Mamba processing, this head:
    1. Projects from 512 → 19 channels
    2. Upsamples from 960 → 15360 samples

    This preserves compatibility with existing loss and post-processing.
    """

    def __init__(
        self,
        input_channels: int = 512,
        output_channels: int = 19,
        upsample_factor: int = 16,
    ):
        super().__init__()

        # 1x1 conv to project channels
        self.proj = nn.Conv1d(input_channels, output_channels, kernel_size=1)

        # Upsample to restore temporal resolution
        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode='nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection head.

        Args:
            x: Mamba output (B, 512, 960)

        Returns:
            Restored resolution (B, 19, 15360)
        """
        x = self.proj(x)     # (B, 19, 960)
        x = self.upsample(x) # (B, 19, 15360)
        return x


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print(f"pytorch-tcn available: {HAS_PYTORCH_TCN}")

    # Test TCN encoder
    tcn = TCNEncoder()
    x = torch.randn(2, 19, 15360)
    out = tcn(x)
    print(f"TCN output shape: {out.shape}")
    print(f"TCN parameters: {count_parameters(tcn)/1e6:.2f}M")

    # Test projection head
    head = ProjectionHead()
    restored = head(out)
    print(f"Projection head output: {restored.shape}")
    print(f"Total parameters: {(count_parameters(tcn) + count_parameters(head))/1e6:.2f}M")