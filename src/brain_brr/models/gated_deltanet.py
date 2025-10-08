"""Bidirectional Gated DeltaNet wrapper for EEG seizure detection.

Wraps FLA's GatedDeltaNet with bidirectional processing similar to BiMamba2.
This is a SHARED module that processes flattened (B*N, d_model, T) tensors.
"""

import torch
import torch.nn as nn

try:
    from fla.layers import GatedDeltaNet as FLAGatedDeltaNet

    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False


class BiGatedDeltaNet(nn.Module):
    """Bidirectional Gated DeltaNet wrapper for EEG seizure detection.

    Wraps FLA's GatedDeltaNet with bidirectional processing similar to BiMamba2.
    IMPORTANT: This is a SHARED module that processes flattened (B*N, d_model, T) tensors,
    NOT separate instances per electrode/pair.

    Args:
        d_model: Model dimension (64 for node stream, 16 for edge stream)
        headdim: Head dimension (8 for node, 4 for edge - MUST satisfy 0.75* constraint)
        num_layers: Number of bidirectional layers (6 for node, 2 for edge)
        conv_size: Short convolution kernel size (default 4, from config)
        dropout: Dropout after fusion (0.1 default)
        fusion_mode: 'sum' or 'concat' (A/B test both!)
        allow_neg_eigval: Research feature for β_t ∈ (0,2) (start False)
    """

    def __init__(
        self,
        d_model: int = 64,
        headdim: int = 8,
        num_layers: int = 6,
        conv_size: int = 4,
        dropout: float = 0.1,
        fusion_mode: str = "sum",
        allow_neg_eigval: bool = False,
    ) -> None:
        super().__init__()

        if not FLA_AVAILABLE:
            raise ImportError(
                "flash-linear-attention library required for BiGatedDeltaNet.\n"
                "Install with: make setup-fla"
            )

        self.d_model = d_model
        self.fusion_mode = fusion_mode

        assert (d_model * 0.75) % headdim == 0, (
            f"Invalid headdim={headdim}: num_heads * head_dim must equal "
            f"{d_model * 0.75} (0.75 * hidden_size)"
        )
        num_heads = int(d_model * 0.75 / headdim)

        print(
            f"[BiGatedDeltaNet] d_model={d_model}, headdim={headdim}, "
            f"num_heads={num_heads} (constraint: {num_heads}*{headdim}={num_heads * headdim}=0.75*{d_model})"
        )

        self.layers = nn.ModuleList()
        for _i in range(num_layers):
            layer = nn.ModuleDict(
                {
                    "fwd": FLAGatedDeltaNet(
                        hidden_size=d_model,
                        head_dim=headdim,
                        num_heads=num_heads,
                        expand_v=2.0,
                        mode="chunk",
                        use_short_conv=True,
                        conv_size=conv_size,
                        use_gate=True,
                        allow_neg_eigval=allow_neg_eigval,
                        conv_bias=False,
                        norm_eps=1e-5,
                    ),
                    "bwd": FLAGatedDeltaNet(
                        hidden_size=d_model,
                        head_dim=headdim,
                        num_heads=num_heads,
                        expand_v=2.0,
                        mode="chunk",
                        use_short_conv=True,
                        conv_size=conv_size,
                        use_gate=True,
                        allow_neg_eigval=allow_neg_eigval,
                        conv_bias=False,
                        norm_eps=1e-5,
                    ),
                }
            )
            self.layers.append(layer)

        self.dropout = nn.Dropout(dropout)

        self.fusion_proj: nn.Linear | None
        if fusion_mode == "concat":
            self.fusion_proj = nn.Linear(d_model * 2, d_model, bias=False)
            nn.init.xavier_uniform_(self.fusion_proj.weight, gain=0.2)
        else:
            self.fusion_proj = None

        self.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional processing: forward + backward (flipped).

        Args:
            x: (B, C, L) where C=d_model (64 or 16), L=960 (sequence length)
               B can be B*19 for node stream or B*171 for edge stream

        Returns:
            x: (B, C, L) bidirectional output
        """
        input_dtype = x.dtype
        if input_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        x = x.transpose(1, 2).contiguous()

        for layer in self.layers:
            residual = x

            x_fwd, _, _ = layer["fwd"](x)

            x_bwd, _, _ = layer["bwd"](x.flip(dims=[1]).contiguous())
            x_bwd = x_bwd.flip(dims=[1])

            if self.fusion_mode == "sum":
                x = x_fwd + x_bwd
            else:
                x = torch.cat([x_fwd, x_bwd], dim=-1)
                if self.fusion_proj is not None:
                    x = self.fusion_proj(x)

            x = residual + self.dropout(x)

        x = x.transpose(1, 2).contiguous()

        if input_dtype != torch.bfloat16:
            x = x.to(input_dtype)

        return x
