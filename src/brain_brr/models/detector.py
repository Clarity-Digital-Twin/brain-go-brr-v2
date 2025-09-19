"""Complete Bi-Mamba-2 + U-Net + ResCNN architecture for TUSZ seizure detection.

CRITICAL INNOVATION: First architecture combining:
- U-Net: Multi-scale hierarchical feature extraction (what Mamba lacks)
- ResCNN: Nonlinear local pattern detection (what SSMs miss)
- Bi-Mamba-2: O(N) global temporal modeling (what CNNs lack)

This synergy addresses TUSZ-specific challenges:
- Complex temporal dynamics (10Hz to 10min scales)
- Adult clinical patterns (vs pediatric CHB-MIT)
- High artifact/noise (real hospital data)
- Variable seizure morphologies (7+ types in TUSZ)
"""

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from .mamba import BiMamba2
from .rescnn import ResCNNStack
from .unet import UNetDecoder, UNetEncoder

if TYPE_CHECKING:  # Only for type checkers; avoids runtime import cycle
    from src.brain_brr.config.schemas import ModelConfig as _ModelConfig


class SeizureDetector(nn.Module):
    """Complete Bi-Mamba-2 + U-Net + ResCNN architecture for TUSZ seizure detection.

    Flow:
        Input (B, 19, 15360) - 60s @ 256Hz, 19 channels
          -> UNetEncoder (B, 512, 960) + skips [Multi-scale extraction]
          -> ResCNNStack (B, 512, 960) [Local pattern refinement]
          -> BiMamba2 (B, 512, 960) [Global bidirectional context]
          -> UNetDecoder (B, 19, 15360) [Multi-scale reconstruction]
          -> 1x1 Conv -> (B, 15360) [Per-sample logits]
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

        # Detection head: fuse 19 channels to 1 logit channel
        # Note: outputs raw logits; apply sigmoid at inference for probabilities
        self.detection_head = nn.Conv1d(in_channels, 1, kernel_size=1)

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
            (B, 15360) per-sample seizure logits (raw scores).
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


__all__ = ["SeizureDetector"]
