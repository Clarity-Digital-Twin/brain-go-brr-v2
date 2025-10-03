"""Fusion head builder - gated or multi-head fusion module."""

from typing import TYPE_CHECKING

from ..fusion import GatedFusion, MultiHeadGatedFusion

if TYPE_CHECKING:
    from src.brain_brr.config.schemas import ModelConfig


def build_fusion_head(cfg: "ModelConfig") -> tuple[str, GatedFusion | MultiHeadGatedFusion | None]:
    """Build fusion head for combining node and edge streams.

    PR-4: Supports gated fusion and multi-head gated fusion for stream combination.
    Default is None (additive fusion in forward()).

    Args:
        cfg: Model configuration containing fusion settings

    Returns:
        Tuple of (fusion_type, fusion_module)
        - fusion_type: "add", "gated", or "multihead" (preserves config value)
        - fusion_module: GatedFusion, MultiHeadGatedFusion, or None

    Notes:
        - Gated fusion: Learnable gate for node/edge weighting (d_model=64)
        - Multi-head fusion: Multiple attention heads for fusion
        - Add fusion: Simple addition (no learnable params, backward compat)
    """
    fusion_cfg = getattr(cfg, "fusion", None)

    if not fusion_cfg:
        return "add", None

    fusion_type = fusion_cfg.fusion_type

    if fusion_type == "gated":
        fusion_module: GatedFusion | MultiHeadGatedFusion = GatedFusion(
            64, fusion_cfg.fusion_dropout
        )
        return fusion_type, fusion_module
    elif fusion_type == "multihead":
        fusion_module = MultiHeadGatedFusion(64, fusion_cfg.fusion_heads, fusion_cfg.fusion_dropout)
        return fusion_type, fusion_module
    else:
        return fusion_type, None
