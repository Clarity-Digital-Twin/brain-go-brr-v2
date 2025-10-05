"""Edge stream builder - per-edge BiMamba component with learned lift/project."""

from typing import TYPE_CHECKING

import torch.nn as nn

from src.brain_brr.constants import LAYERSCALE_ALPHA_FALLBACK

from ..mamba import BiMamba2
from ..norms import create_norm_layer

if TYPE_CHECKING:
    from src.brain_brr.config.schemas import ModelConfig


class EdgeStreamComponents:
    """Container for edge stream components (avoids tuple unpacking)."""

    def __init__(
        self,
        edge_mamba: BiMamba2,
        edge_in_proj: nn.Conv1d,
        edge_out_proj: nn.Conv1d,
        edge_activate: nn.Softplus,
        edge_lift_act: nn.Module | None,
        edge_lift_norm: nn.Module | None,
    ):
        self.edge_mamba = edge_mamba
        self.edge_in_proj = edge_in_proj
        self.edge_out_proj = edge_out_proj
        self.edge_activate = edge_activate
        self.edge_lift_act = edge_lift_act
        self.edge_lift_norm = edge_lift_norm


def build_edge_stream(cfg: "ModelConfig") -> EdgeStreamComponents:
    """Build edge stream: per-edge BiMamba with learned lift/project.

    V3 Architecture: Processes edge similarities (171 pairs) with BiMamba.
    Pipeline: 1D → lift(d_model) → BiMamba → project(1D) → Softplus

    Args:
        cfg: Model configuration containing graph and norms settings

    Returns:
        EdgeStreamComponents with all edge processing modules

    Notes:
        - edge_d_model must be multiple of 8 for CUDA alignment
        - headdim=4 ensures (16 * 2) / 4 = 8 which is multiple of 8
        - PR-2: Supports bounded edge stream (activation + norm on lift)
        - LayerScale enabled if boundary_norm != "none"
    """
    graph_cfg = cfg.graph
    norms_cfg = getattr(cfg, "norms", None)

    edge_layers = graph_cfg.edge_mamba_layers if graph_cfg else 2
    edge_d_state = graph_cfg.edge_mamba_d_state if graph_cfg else 8
    edge_d_model = graph_cfg.edge_mamba_d_model if graph_cfg else 16

    assert edge_d_model % 8 == 0, (
        f"edge_mamba_d_model must be multiple of 8 for CUDA, got {edge_d_model}"
    )
    assert edge_d_model > 0, f"edge_mamba_d_model must be positive, got {edge_d_model}"

    use_layerscale = bool(norms_cfg and norms_cfg.boundary_norm != "none")
    layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)

    edge_mamba = BiMamba2(
        d_model=edge_d_model,
        d_state=edge_d_state,
        d_conv=4,
        expand=2,
        headdim=4,
        num_layers=edge_layers,
        dropout=cfg.mamba.dropout,
        use_layerscale=use_layerscale,
        layerscale_init=layerscale_init,
    )

    edge_in_proj = nn.Conv1d(1, edge_d_model, kernel_size=1, bias=False)
    edge_out_proj = nn.Conv1d(edge_d_model, 1, kernel_size=1, bias=True)
    edge_activate = nn.Softplus()

    edge_lift_activation = graph_cfg.edge_lift_activation if graph_cfg else "none"
    edge_lift_norm_type = graph_cfg.edge_lift_norm if graph_cfg else "none"
    edge_lift_gain = graph_cfg.edge_lift_init_gain if graph_cfg else 0.1

    edge_lift_act: nn.Tanh | nn.Sigmoid | nn.SELU | None
    if edge_lift_activation == "tanh":
        edge_lift_act = nn.Tanh()
    elif edge_lift_activation == "sigmoid":
        edge_lift_act = nn.Sigmoid()
    elif edge_lift_activation == "selu":
        edge_lift_act = nn.SELU()
    else:
        edge_lift_act = None

    edge_lift_norm = create_norm_layer(edge_lift_norm_type, edge_d_model)

    nn.init.xavier_uniform_(edge_in_proj.weight, gain=edge_lift_gain)
    if edge_out_proj.bias is not None:
        nn.init.zeros_(edge_out_proj.bias)
    nn.init.xavier_uniform_(edge_out_proj.weight, gain=edge_lift_gain)

    return EdgeStreamComponents(
        edge_mamba=edge_mamba,
        edge_in_proj=edge_in_proj,
        edge_out_proj=edge_out_proj,
        edge_activate=edge_activate,
        edge_lift_act=edge_lift_act,
        edge_lift_norm=edge_lift_norm,
    )
