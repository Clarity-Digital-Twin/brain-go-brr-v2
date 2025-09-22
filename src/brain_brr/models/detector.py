"""TCN + Bi-Mamba-2 architecture for TUSZ seizure detection.

CRITICAL INNOVATION: Modern architecture combining:
- TCN: Multi-scale temporal features with dilated convolutions
- Bi-Mamba-2: O(N) global temporal modeling with state-space models

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
from .tcn import ProjectionHead, TCNEncoder

if TYPE_CHECKING:  # Only for type checkers; avoids runtime import cycle
    from src.brain_brr.config.schemas import ModelConfig as _ModelConfig


class SeizureDetector(nn.Module):
    """TCN + Bi-Mamba architecture for TUSZ seizure detection.

    Flow:
        Input (B, 19, 15360) - 60s @ 256Hz, 19 channels
          -> TCNEncoder (B, 512, 960) [Multi-scale temporal features]
          -> BiMamba2 (B, 512, 960) [Global bidirectional context]
          -> Projection + Upsample (B, 19, 15360) [Restore resolution]
          -> 1x1 Conv -> (B, 15360) [Per-sample logits]
    """

    # Keep a tag for reporting/backward-compat
    architecture: str | None = "tcn"

    def __init__(
        self,
        *,
        # Legacy params (accepted but ignored for backward compatibility)
        in_channels: int = 19,
        base_channels: int = 64,
        encoder_depth: int = 4,
        rescnn_blocks: int = 3,
        rescnn_kernels: list[int] | None = None,
        dropout: float = 0.1,
        # TCN params (new style)
        tcn_layers: int = 8,
        tcn_kernel_size: int = 7,
        tcn_dropout: float = 0.15,
        tcn_stride: int = 16,
        # Mamba params
        mamba_layers: int = 6,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_dropout: float | None = None,
    ) -> None:
        super().__init__()

        # Persist config snapshot
        self.config: dict[str, object] = {
            "tcn_layers": tcn_layers,
            "tcn_kernel_size": tcn_kernel_size,
            "tcn_dropout": tcn_dropout,
            "tcn_stride": tcn_stride,
            "mamba_layers": mamba_layers,
            "mamba_d_state": mamba_d_state,
            "mamba_d_conv": mamba_d_conv,
            "mamba_dropout": mamba_dropout,
            "architecture": "tcn",
        }

        # TCN encoder: 19 channels -> 512 channels, 15360 -> 960 samples
        self.tcn_encoder = TCNEncoder(
            input_channels=19,
            output_channels=512,
            num_layers=tcn_layers,
            kernel_size=tcn_kernel_size,
            dropout=tcn_dropout,
            causal=False,
            stride_down=tcn_stride,
        )

        # Bi-Mamba for temporal modeling
        self.mamba = BiMamba2(
            d_model=512,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            num_layers=mamba_layers,
            dropout=mamba_dropout,
        )

        # Projection head: 512 -> 19 channels, 960 -> 15360 samples
        self.proj_head = ProjectionHead(
            input_channels=512,
            output_channels=19,
            upsample_factor=tcn_stride,
        )

        # Detection head: 19 channels to 1 probability channel
        self.detection_head = nn.Conv1d(19, 1, kernel_size=1)

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
        """Forward pass through TCN + Bi-Mamba architecture.

        Args:
            x: (B, 19, 15360) EEG window tensor

        Returns:
            (B, 15360) per-sample seizure logits (raw scores).
        """
        # TCN encoder: extract multi-scale temporal features
        features = self.tcn_encoder(x)  # (B, 512, 960)

        # Bi-Mamba: capture long-range dependencies
        temporal = self.mamba(features)  # (B, 512, 960)

        # Project back to 19 channels and upsample to original resolution
        decoded = self.proj_head(temporal)  # (B, 19, 15360)
        output = self.detection_head(decoded)  # (B, 1, 15360)
        return cast(torch.Tensor, output.squeeze(1))

    @classmethod
    def from_config(cls, cfg: "_ModelConfig") -> "SeizureDetector":
        """Instantiate from validated schema config (TCN path)."""
        return cls(
            tcn_layers=cfg.tcn.num_layers,
            tcn_kernel_size=cfg.tcn.kernel_size,
            tcn_dropout=cfg.tcn.dropout,
            tcn_stride=cfg.tcn.stride_down,
            mamba_layers=cfg.mamba.n_layers,
            mamba_d_state=cfg.mamba.d_state,
            mamba_d_conv=cfg.mamba.conv_kernel,
            mamba_dropout=cfg.mamba.dropout,
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_info(self) -> dict[str, object]:
        """Get per-component and total parameter counts plus config snapshot."""

        def count(mod: nn.Module) -> int:
            return sum(p.numel() for p in mod.parameters())

        tcn_params = count(self.tcn_encoder)
        mamba_params = count(self.mamba)
        proj_params = count(self.proj_head)
        head_params = count(self.detection_head)

        total_params = tcn_params + mamba_params + proj_params + head_params

        # Provide parameter info
        info: dict[str, object] = {
            "tcn_params": tcn_params,
            "mamba_params": mamba_params,
            "head_params": head_params,
            # Also expose detailed keys
            "tcn_params": tcn_params,
            "proj_params": proj_params,
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
