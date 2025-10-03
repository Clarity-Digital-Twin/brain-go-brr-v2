"""Regularization builder - boundary norms and LayerScale components."""

from typing import TYPE_CHECKING

import torch.nn as nn

from ..norms import LayerScale, create_norm_layer

if TYPE_CHECKING:
    from src.brain_brr.config.schemas import ModelConfig


class RegularizationComponents:
    """Container for regularization components (boundary norms + LayerScale)."""

    def __init__(self) -> None:
        self.norm_after_proj_to_electrodes: nn.Module | None = None
        self.norm_after_node_mamba: nn.Module | None = None
        self.norm_after_edge_mamba: nn.Module | None = None
        self.norm_after_gnn: nn.Module | None = None
        self.norm_before_decoder: nn.Module | None = None
        self.gnn_layerscale: LayerScale | None = None


def build_regularizers(cfg: "ModelConfig", edge_d_model: int) -> RegularizationComponents:
    """Build regularization components: boundary norms and LayerScale.

    PR-1: Boundary normalization at component interfaces for gradient stability.
    Supports: LayerNorm, RMSNorm, or none.

    Args:
        cfg: Model configuration containing norms and graph settings
        edge_d_model: Dimension of edge stream (needed for edge norm)

    Returns:
        RegularizationComponents with all norm layers and LayerScale

    Notes:
        - Boundary norms: Stabilize activations at module interfaces
        - LayerScale: Scale GNN residual connection (alpha warmup support)
        - Only created if norms.boundary_norm != "none"
    """
    components = RegularizationComponents()

    norms_cfg = getattr(cfg, "norms", None)
    if not norms_cfg or norms_cfg.boundary_norm == "none":
        return components

    graph_cfg = getattr(cfg, "graph", None)

    if norms_cfg.after_tcn_proj:
        components.norm_after_proj_to_electrodes = create_norm_layer(
            norms_cfg.boundary_norm, 64, norms_cfg.boundary_eps
        )

    if norms_cfg.after_node_mamba:
        components.norm_after_node_mamba = create_norm_layer(
            norms_cfg.boundary_norm, 64, norms_cfg.boundary_eps
        )

    if norms_cfg.after_edge_mamba:
        components.norm_after_edge_mamba = create_norm_layer(
            norms_cfg.boundary_norm, edge_d_model, norms_cfg.boundary_eps
        )

    if norms_cfg.after_gnn:
        components.norm_after_gnn = create_norm_layer(
            norms_cfg.boundary_norm, 64, norms_cfg.boundary_eps
        )

    if norms_cfg.before_decoder:
        components.norm_before_decoder = create_norm_layer(
            norms_cfg.boundary_norm, 512, norms_cfg.boundary_eps
        )

    if graph_cfg and graph_cfg.use_residual:
        components.gnn_layerscale = LayerScale(64, norms_cfg.layerscale_alpha)

    return components
