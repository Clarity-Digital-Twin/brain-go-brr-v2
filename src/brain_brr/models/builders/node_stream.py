"""Node stream builder - per-electrode BiMamba component."""

from typing import TYPE_CHECKING

from src.brain_brr.constants import LAYERSCALE_ALPHA_FALLBACK

from ..mamba import BiMamba2

if TYPE_CHECKING:
    from src.brain_brr.config.schemas import ModelConfig


def build_node_stream(cfg: "ModelConfig") -> BiMamba2:
    """Build node stream: per-electrode BiMamba module.

    V3 Architecture: Processes each electrode (19 channels) independently with BiMamba.
    Features: 64 dimensions per electrode, 6 layers, headdim=8 for CUDA alignment.

    Args:
        cfg: Model configuration containing norms and mamba settings

    Returns:
        BiMamba2 module for node-level temporal processing

    Notes:
        - headdim=8 ensures (64 * 2) / 8 = 16 which is multiple of 8 (CUDA requirement)
        - LayerScale enabled if boundary_norm != "none"
        - Fixed architecture params: d_model=64, d_state=16, num_layers=6
    """
    norms_cfg = getattr(cfg, "norms", None)
    use_layerscale = bool(norms_cfg and norms_cfg.boundary_norm != "none")
    layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)

    return BiMamba2(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=8,
        num_layers=6,
        dropout=cfg.mamba.dropout,
        use_layerscale=use_layerscale,
        layerscale_init=layerscale_init,
    )
