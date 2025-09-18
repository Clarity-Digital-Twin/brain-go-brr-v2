"""Neural architecture components for Bi-Mamba-2 + U-Net + ResCNN seizure detection.

Phase 2 implementation will land incrementally via TDD:
- UNetEncoder: 4-stage encoder with skip connections (implemented in Phase 2.1)
- ResCNNStack: Multi-scale residual CNN blocks (Phase 2.2)
- BiMamba2: Bidirectional Mamba-2 layers (Phase 2.3)
- UNetDecoder: 4-stage decoder with skip fusion (Phase 2.4)
- SeizureDetector: Full assembled model (Phase 2.5)
"""

import os
import warnings
from typing import TYPE_CHECKING, cast

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
    from mamba_ssm import Mamba2  # type: ignore[import-untyped]

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
        self.d_conv = d_conv  # Public/configured kernel (docs/schemas default=5)

        # Optional override to force fallback even if CUDA/Mamba are available
        self._force_fallback = os.getenv("SEIZURE_MAMBA_FORCE_FALLBACK", "0") == "1"

        # Mamba CUDA kernel supports width in {2, 3, 4}; coerce if needed (GPU path only)
        _allowed = (2, 3, 4)
        self._mamba_conv_k = d_conv if d_conv in _allowed else 4
        if MAMBA_AVAILABLE and d_conv not in _allowed:
            warnings.warn(
                f"Mamba CUDA path coerced conv kernel from {d_conv}→{self._mamba_conv_k} "
                "(CUDA op supports only 2-4). CPU fallback still uses configured kernel.",
                stacklevel=1,
            )

        # Always create both paths - decide at runtime
        if MAMBA_AVAILABLE:
            # Real Mamba-2 for GPU
            self.forward_mamba_real = Mamba2(
                d_model=d_model, d_state=d_state, d_conv=self._mamba_conv_k, expand=expand
            )
            self.backward_mamba_real = Mamba2(
                d_model=d_model, d_state=d_state, d_conv=self._mamba_conv_k, expand=expand
            )
        else:
            self.forward_mamba_real = None
            self.backward_mamba_real = None

        # Fallback for CPU/testing (Conv1d to match docs/tests; operates on (B, C, L))
        # WARNING: This is NOT functionally equivalent to Mamba-2 SSM!
        padding = max(0, self.d_conv // 2)
        self.forward_mamba_fallback = nn.Conv1d(
            d_model, d_model, kernel_size=self.d_conv, padding=padding
        )
        self.backward_mamba_fallback = nn.Conv1d(
            d_model, d_model, kernel_size=self.d_conv, padding=padding
        )

        # Fusion and normalization
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @property
    def forward_mamba(self) -> nn.Module:
        """Compatibility property for tests."""
        return (
            self.forward_mamba_real
            if self.forward_mamba_real is not None
            else self.forward_mamba_fallback
        )

    @property
    def backward_mamba(self) -> nn.Module:
        """Compatibility property for tests."""
        return (
            self.backward_mamba_real
            if self.backward_mamba_real is not None
            else self.backward_mamba_fallback
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input bidirectionally.

        Args:
            x: Input (B, L, D) where L=960, D=512

        Returns:
            Bidirectional output (B, L, D)
        """
        residual = x

        # Use real Mamba only if:
        # 1. Library is available
        # 2. CUDA is available
        # 3. Input is on CUDA
        # 4. Kernel width is supported (2-4 for causal_conv1d)
        use_mamba = (
            MAMBA_AVAILABLE
            and x.is_cuda  # check tensor first to avoid noisy CUDA init on CPU
            and torch.cuda.is_available()
            and self._mamba_conv_k in (2, 3, 4)  # causal_conv1d constraint
            and not self._force_fallback
        )

        # Forward direction
        x_forward = (
            self.forward_mamba_real(x)
            if use_mamba
            else self.forward_mamba_fallback(x.transpose(1, 2)).transpose(1, 2)
        )

        # Backward direction (flip sequence)
        x_backward = x.flip(dims=[1])
        if use_mamba:
            x_backward = self.backward_mamba_real(x_backward)
        else:
            # Conv1d fallback with transpose
            x_backward = self.backward_mamba_fallback(x_backward.transpose(1, 2)).transpose(1, 2)

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
            "skip_usage_order": "skips[3] -> skips[2] -> skips[1] -> skips[0]",
            "expected_skip_shapes": [
                (64, 15360),
                (128, 7680),
                (256, 3840),
                (512, 1920),
            ],
        }
        return info


# ============================================================================
# Phase 2.5: Full Model Assembly
# ============================================================================


if TYPE_CHECKING:  # Only for type checkers; avoids runtime import cycle
    from src.experiment.schemas import ModelConfig as _ModelConfig


class SeizureDetector(nn.Module):
    """Complete Bi-Mamba-2 + U-Net + ResCNN architecture for seizure detection.

    Flow:
        Input (B, 19, 15360)
          -> UNetEncoder (B, 512, 960) + skips
          -> ResCNNStack (B, 512, 960)
          -> BiMamba2 (B, 512, 960)
          -> UNetDecoder (B, 19, 15360)
          -> 1x1 Conv + Sigmoid -> (B, 15360)
    """

    def __init__(
        self,
        *,
        in_channels: int = 19,
        base_channels: int = 64,
        encoder_depth: int = 4,
        # Mamba params
        mamba_layers: int = 6,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 5,
        # ResCNN params
        rescnn_blocks: int = 3,
        rescnn_kernels: list[int] | None = None,
        # Regularization
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if rescnn_kernels is None:
            rescnn_kernels = [3, 5, 7]

        # Persist minimal config snapshot for debugging/reporting
        self.config: dict[str, object] = {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "encoder_depth": encoder_depth,
            "mamba_layers": mamba_layers,
            "mamba_d_state": mamba_d_state,
            "mamba_d_conv": mamba_d_conv,
            "rescnn_blocks": rescnn_blocks,
            "rescnn_kernels": rescnn_kernels,
            "dropout": dropout,
        }

        bottleneck_channels = base_channels * (2 ** (encoder_depth - 1))  # 512 for defaults

        # Components
        self.encoder = UNetEncoder(
            in_channels=in_channels, base_channels=base_channels, depth=encoder_depth
        )
        self.rescnn = ResCNNStack(
            channels=bottleneck_channels,
            num_blocks=rescnn_blocks,
            kernel_sizes=rescnn_kernels,
            dropout=dropout,
        )
        self.mamba = BiMamba2(
            d_model=bottleneck_channels,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            num_layers=mamba_layers,
            dropout=dropout,
        )
        self.decoder = UNetDecoder(
            out_channels=in_channels, base_channels=base_channels, depth=encoder_depth
        )

        # Detection head: fuse 19 channels to 1 probability channel
        self.detection_head = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights (He/Xavier) for conv/linear/bn layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through complete architecture.

        Args:
            x: (B, 19, 15360) EEG window tensor

        Returns:
            (B, 15360) per-sample seizure probabilities in [0, 1].
        """
        encoded, skips = self.encoder(x)  # (B, 512, 960) + 4 skips
        features = self.rescnn(encoded)  # (B, 512, 960)
        temporal = self.mamba(features)  # (B, 512, 960)
        decoded = self.decoder(temporal, skips)  # (B, 19, 15360)
        output = self.detection_head(decoded)  # (B, 1, 15360)
        return cast(torch.Tensor, output.squeeze(1))  # (B, 15360)

    @classmethod
    def from_config(cls, cfg: "_ModelConfig") -> "SeizureDetector":
        """Instantiate from validated schema config (prevents name drift).

        Note: `in_channels` fixed at 19 for the 10-20 montage in this project.
        """
        return cls(
            in_channels=19,
            base_channels=cfg.encoder.channels[0],
            encoder_depth=cfg.encoder.stages,
            mamba_layers=cfg.mamba.n_layers,
            mamba_d_state=cfg.mamba.d_state,
            mamba_d_conv=cfg.mamba.conv_kernel,
            rescnn_blocks=cfg.rescnn.n_blocks,
            rescnn_kernels=cfg.rescnn.kernel_sizes,
            dropout=cfg.mamba.dropout,
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> dict[str, object]:
        """Get per-component and total parameter counts plus config snapshot."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        rescnn_params = sum(p.numel() for p in self.rescnn.parameters())
        mamba_params = sum(p.numel() for p in self.mamba.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        head_params = sum(p.numel() for p in self.detection_head.parameters())

        total_params = encoder_params + rescnn_params + mamba_params + decoder_params + head_params

        info: dict[str, object] = {
            "encoder_params": encoder_params,
            "rescnn_params": rescnn_params,
            "mamba_params": mamba_params,
            "decoder_params": decoder_params,
            "head_params": head_params,
            "total_params": total_params,
            "config": self.config,
        }
        return info

    def get_memory_usage(self, batch_size: int = 16) -> dict[str, float]:
        """Rough memory usage estimate in MB for parameters and activations."""
        # Model parameters (float32)
        param_bytes = self.count_parameters() * 4

        # Largest activation at input resolution (approx)
        activation_bytes = batch_size * 19 * 15360 * 4

        # Include some intermediate activations (rough multiplier)
        total_activation_bytes = activation_bytes * 3

        return {
            "model_size_mb": param_bytes / (1024**2),
            "activation_size_mb": total_activation_bytes / (1024**2),
            "total_size_mb": (param_bytes + total_activation_bytes) / (1024**2),
        }
