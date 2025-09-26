"""TCN (Temporal Convolutional Network) encoder for EEG seizure detection.

Replaces U-Net encoder/decoder + ResCNN with a modern TCN architecture.
Uses pytorch-tcn if available, falls back to minimal implementation.
"""

import warnings
from typing import cast

import torch
import torch.nn as nn

from src.brain_brr.utils.env import env

# Suppress deprecation warning for weight_norm - we use old API for torch.compile compat
warnings.filterwarnings(
    "ignore",
    message=".*weight_norm is deprecated.*",
    category=UserWarning,
    module="torch.nn.utils.weight_norm",
)

# Try to import pytorch-tcn (optional dependency)
try:
    from pytorch_tcn import TCN

    HAS_PYTORCH_TCN = True
except ImportError:
    HAS_PYTORCH_TCN = False
    warnings.warn(
        "pytorch-tcn not installed. Install with: uv sync --extra tcn\n"
        "Falling back to minimal TCN implementation.",
        stacklevel=2,
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

        layers: list[nn.Module] = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_size if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            # Dilated convolution
            padding = (
                (kernel_size - 1) * dilation_size
                if causal
                else (kernel_size - 1) * dilation_size // 2
            )
            conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, padding=padding, dilation=dilation_size
            )

            # Weight normalization (using old API for torch.compile compatibility)
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
        out = cast(torch.Tensor, self.network(x))

        if self.causal:
            # Trim the output for causal convolution
            out = out[:, :, : x.size(2)]

        return cast(torch.Tensor, self.projection(out))


class TCNEncoder(nn.Module):
    """TCN encoder that produces features for Bi-Mamba.

    Input:  (B, 19, L) where L is divisible by stride_down (default 16)
    Output: (B, 512, L/16)   - Downsampled by 16x for Mamba
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
        init_gain: float = 0.2,  # Dependency injection for initialization
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride_down = stride_down
        self.init_gain = init_gain

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

        # Choose backend: prefer external TCN only when explicitly enabled or forced
        # The external pytorch-tcn can hang on certain configurations
        force_ext = env.force_tcn_ext()
        force_internal = not force_ext

        use_external = HAS_PYTORCH_TCN and force_ext and not force_internal

        # TCN backbone
        if use_external:
            # Use pytorch-tcn package
            self.tcn = TCN(
                num_inputs=input_channels,
                num_channels=num_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                causal=causal,
                use_norm="weight_norm",
                activation="relu",
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
                causal=causal,
            )
            tcn_out_channels = num_channels[-1]

        # Projection to output channels
        self.channel_proj = nn.Conv1d(tcn_out_channels, output_channels, kernel_size=1)

        # Downsampling to match Mamba input size
        # 15360 / 16 = 960
        self.downsample = nn.Conv1d(
            output_channels, output_channels, kernel_size=stride_down, stride=stride_down
        )

        # Initialize weights conservatively
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize TCN encoder weights with conservative gains for stability."""
        # Use conservative initialization for production stability
        # Tests can pass higher init_gain if needed for gradient flow validation
        proj_gain = self.init_gain  # Default 0.2
        down_gain = self.init_gain * 0.5  # Half of proj_gain
        conv_scale = self.init_gain * 2.5  # 2.5x proj_gain

        # Channel projection
        nn.init.xavier_uniform_(self.channel_proj.weight, gain=proj_gain)
        if self.channel_proj.bias is not None:
            nn.init.zeros_(self.channel_proj.bias)

        # Downsampling
        nn.init.xavier_uniform_(self.downsample.weight, gain=down_gain)
        if self.downsample.bias is not None:
            nn.init.zeros_(self.downsample.bias)

        # TCN layers: mode-dependent initialization
        if hasattr(self.tcn, "network"):
            # MinimalTCN case
            for module in self.tcn.network.modules():
                if isinstance(module, nn.Conv1d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    module.weight.data *= conv_scale
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            # Projection layer in MinimalTCN
            if hasattr(self.tcn, "projection"):
                nn.init.xavier_uniform_(self.tcn.projection.weight, gain=proj_gain)
                if self.tcn.projection.bias is not None:
                    nn.init.zeros_(self.tcn.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN encoder.

        Args:
            x: Input EEG tensor (B, 19, L)

        Returns:
            Encoded features (B, 512, L/stride_down)
        """
        # Check input shape
        _b, c, length = x.shape
        assert self.input_channels == c, f"Expected {self.input_channels} channels, got {c}"
        if length % self.stride_down != 0:
            raise AssertionError(
                f"Input length {length} must be divisible by stride_down {self.stride_down}"
            )

        # CRITICAL: Input validation and clamping to prevent NaN propagation
        # Check for NaN/Inf in inputs
        if torch.isnan(x).any() or torch.isinf(x).any():
            # Replace NaN/Inf with zeros
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # Input tier clamping: [-10, 10] for normalized EEG data
        x = torch.clamp(x, min=-10.0, max=10.0)

        # TCN processing
        x = self.tcn(x)  # (B, tcn_channels, 15360)

        # Internal tier clamping: [-50, 50] for feature maps
        if env.safe_clamp():
            x = torch.clamp(x, min=-50.0, max=50.0)

        # Project to output channels
        x = self.channel_proj(x)  # (B, 512, 15360)

        # Maintain internal tier for features
        if env.safe_clamp():
            x = torch.clamp(x, min=-50.0, max=50.0)

        # Downsample for Mamba
        x = self.downsample(x)  # (B, 512, 960)

        # Final output uses internal tier (features, not logits)
        if env.safe_clamp():
            x = torch.clamp(x, min=-50.0, max=50.0)

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
        self.upsample = nn.Upsample(scale_factor=upsample_factor, mode="nearest")

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights conservatively to prevent NaN/explosion."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                # Very small initialization for projection head
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through projection head.

        Args:
            x: Mamba output (B, 512, 960)

        Returns:
            Restored resolution (B, 19, 15360)
        """
        x = self.proj(x)  # (B, 19, 960)
        x = self.upsample(x)  # (B, 19, 15360)
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
    print(f"TCN parameters: {count_parameters(tcn) / 1e6:.2f}M")

    # Test projection head
    head = ProjectionHead()
    restored = head(out)
    print(f"Projection head output: {restored.shape}")
    print(f"Total parameters: {(count_parameters(tcn) + count_parameters(head)) / 1e6:.2f}M")
