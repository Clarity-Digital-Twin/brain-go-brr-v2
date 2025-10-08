"""Node stream builder - per-electrode BiMamba component."""

from typing import TYPE_CHECKING, Union

from src.brain_brr.constants import (
    GDN_FUSION_MODE_DEFAULT,
    GDN_NODE_HEADDIM_DEFAULT,
    LAYERSCALE_ALPHA_FALLBACK,
    NODE_D_MODEL,
    NODE_D_STATE,
    NODE_EXPAND,
    NODE_HEADDIM_BIMAMBA2,
    NODE_NUM_LAYERS,
)

from ..mamba import BiMamba2

if TYPE_CHECKING:
    from src.brain_brr.config.schemas import ModelConfig

    from ..gated_deltanet import BiGatedDeltaNet

try:
    from fla.layers import GatedDeltaNet

    FLA_AVAILABLE = True
    del GatedDeltaNet
except ImportError:
    FLA_AVAILABLE = False


def build_node_stream(cfg: "ModelConfig") -> Union[BiMamba2, "BiGatedDeltaNet"]:
    """Build node stream: BiMamba2 (default) or BiGatedDeltaNet (experimental).

    Returns BiMamba2 by default for stability. BiGatedDeltaNet (GDN via FLA library)
    is only used if explicitly configured via temporal_type_node or temporal_type.

    V3 Architecture: Processes each electrode (19 channels) with shared SSM.
    Features: 64 dimensions per electrode, 6 layers, headdim=8 for CUDA alignment.

    Args:
        cfg: Model configuration with mamba settings

    Returns:
        Shared SSM module for node stream (processes 19 electrodes)

    Raises:
        ImportError: If GDN requested but FLA library not installed

    Notes:
        - BiMamba2: headdim=8 ensures (64 * 2) / 8 = 16 (CUDA requirement)
        - GDN: num_heads * headdim = 0.75 * 64 = 48 (6 * 8 = 48)
        - LayerScale enabled if boundary_norm != "none"
        - Fixed architecture params from constants
    """
    temporal_type = getattr(cfg.mamba, "temporal_type_node", None)
    if temporal_type is None:
        temporal_type = getattr(cfg.mamba, "temporal_type", "bimamba2")

    norms_cfg = getattr(cfg, "norms", None)
    use_layerscale = bool(norms_cfg and norms_cfg.boundary_norm != "none")
    layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)

    if temporal_type == "gated_deltanet":
        if not FLA_AVAILABLE:
            raise ImportError(
                "Gated DeltaNet requires flash-linear-attention library.\n"
                "Install: make setup-fla\n"
                "Or set temporal_type='bimamba2' in config to use stable baseline."
            )

        from ..gated_deltanet import BiGatedDeltaNet

        fusion_mode = getattr(cfg.mamba, "gdn_fusion_mode", GDN_FUSION_MODE_DEFAULT)
        allow_neg_eigval = getattr(cfg.mamba, "gdn_allow_neg_eigval", False)

        return BiGatedDeltaNet(
            d_model=NODE_D_MODEL,
            headdim=GDN_NODE_HEADDIM_DEFAULT,
            num_layers=NODE_NUM_LAYERS,
            conv_size=cfg.mamba.conv_kernel,
            dropout=cfg.mamba.dropout,
            fusion_mode=fusion_mode,
            allow_neg_eigval=allow_neg_eigval,
        )
    else:
        return BiMamba2(
            d_model=NODE_D_MODEL,
            d_state=NODE_D_STATE,
            d_conv=cfg.mamba.conv_kernel,
            expand=NODE_EXPAND,
            headdim=NODE_HEADDIM_BIMAMBA2,
            num_layers=NODE_NUM_LAYERS,
            dropout=cfg.mamba.dropout,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
        )
