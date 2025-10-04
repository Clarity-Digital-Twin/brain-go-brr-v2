"""TCN + Bi-Mamba-2 architecture for TUSZ seizure detection.

CRITICAL INNOVATION: Modern architecture combining:
- TCN: Multi-scale temporal features with dilated convolutions
- Bi-Mamba-2: O(N) global temporal modeling with state-space models
- Dynamic GNN (optional): Time-evolving brain network modeling

This synergy addresses TUSZ-specific challenges:
- Complex temporal dynamics (10Hz to 10min scales)
- Adult clinical patterns (vs pediatric CHB-MIT)
- High artifact/noise (real hospital data)
- Variable seizure morphologies (7+ types in TUSZ)
"""

from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from src.brain_brr.constants import EPSILON_LAPLACIAN

from .builders import (
    build_edge_stream,
    build_fusion_head,
    build_node_stream,
    build_regularizers,
)
from .debug_utils import assert_finite
from .mamba import BiMamba2
from .tcn import ProjectionHead, TCNEncoder

if TYPE_CHECKING:  # Only for type checkers; avoids runtime import cycle
    from src.brain_brr.config.schemas import ModelConfig as _ModelConfig
    from src.brain_brr.config.schemas import WarmupScheduleConfig


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
    architecture: str = "v3"

    def __init__(
        self,
        *,
        # Core (fixed input channels)
        in_channels: int = 19,
        # TCN params
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
        """Initialize SeizureDetector V3 dual-stream architecture.

        Args:
            in_channels: Number of EEG input channels (default: 19 for 10-20 montage)
            tcn_layers: Number of TCN residual blocks (default: 8)
            tcn_kernel_size: TCN convolution kernel size (default: 7)
            tcn_dropout: TCN dropout rate (default: 0.15)
            tcn_stride: Temporal downsampling factor (default: 16 → 960 samples)
            mamba_layers: Number of bidirectional Mamba layers (default: 6)
            mamba_d_state: Mamba state dimension (default: 16)
            mamba_d_conv: Mamba convolution dimension (default: 4)
            mamba_dropout: Mamba dropout rate (default: 0.1 if None)

        Note:
            GNN and V3 dual-stream components are initialized via from_config()
            if enabled in configuration. This __init__ only sets up core TCN/Mamba.
        """
        super().__init__()

        # GNN components (initialized as None, set by from_config if enabled)
        self.use_gnn: bool = False
        self.gnn: nn.Module | None = None
        self.proj_to_electrodes: nn.Conv1d | None = None
        self.proj_from_electrodes: nn.Conv1d | None = None

        # V3 dual-stream components (initialized as None, set by from_config if v3)
        self.node_mamba: nn.Module | None = None
        self.edge_mamba: nn.Module | None = None
        self.edge_in_proj: nn.Conv1d | None = None
        self.edge_out_proj: nn.Conv1d | None = None
        self.edge_activate: nn.Module | None = None

        # PR-1: Boundary normalization layers (initialized as None, set by from_config)
        self.norm_after_proj_to_electrodes: nn.Module | None = None
        self.norm_after_node_mamba: nn.Module | None = None
        self.norm_after_edge_mamba: nn.Module | None = None
        self.norm_after_gnn: nn.Module | None = None
        self.norm_before_decoder: nn.Module | None = None

        # PR-1: LayerScale for residual connections (initialized as None)
        self.gnn_layerscale: nn.Module | None = None

        # PR-2: Bounded edge stream components (initialized as None, set by from_config)
        self.edge_lift_act: nn.Module | None = None
        self.edge_lift_norm: nn.Module | None = None

        # PR-4: Fusion module for node/edge combination (initialized in from_config)
        self.fusion: nn.Module | None = None
        self.fusion_type: str = "add"

        # Warmup schedule state (updated by training loop)
        self.global_step: int = 0
        self.warmup_config: WarmupScheduleConfig | None = None

        # Backwards-compat: ensure mamba_dropout has a concrete value
        if mamba_dropout is None:
            mamba_dropout = 0.1

        # Persist config snapshot (include legacy keys for tests)
        self.config: dict[str, object] = {
            "in_channels": in_channels,
            "tcn_layers": tcn_layers,
            "tcn_kernel_size": tcn_kernel_size,
            "tcn_dropout": tcn_dropout,
            "tcn_stride": tcn_stride,
            "mamba_layers": mamba_layers,
            "mamba_d_state": mamba_d_state,
            "mamba_d_conv": mamba_d_conv,
            "mamba_dropout": mamba_dropout,
            "architecture": "v3",
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
        # headdim=64 with d_model=512 ensures (512*2)/64 = 16 which is multiple of 8
        self.mamba = BiMamba2(
            d_model=512,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=2,
            headdim=64,  # (512*2)/64 = 16 is multiple of 8
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

    def set_training_state(
        self,
        global_step: int,
        warmup_config: "WarmupScheduleConfig | None" = None,
    ) -> None:
        """Update training state for warmup schedules (v3.4.1).

        Propagates state to submodules that rely on warmup configuration.
        Safe to call every iteration; no-ops when components are absent.
        """

        self.global_step = global_step
        self.warmup_config = warmup_config

        if self.use_gnn and self.gnn is not None:
            if hasattr(self.gnn, "set_global_step"):
                self.gnn.set_global_step(global_step)
            else:  # Defensive fallback (legacy modules)
                object.__setattr__(self.gnn, "global_step", global_step)

            if hasattr(self.gnn, "set_warmup_config"):
                self.gnn.set_warmup_config(warmup_config)
            else:
                object.__setattr__(self.gnn, "warmup_config", warmup_config)

    def _initialize_weights(self) -> None:
        """Initialize weights with conservative gains to prevent NaN/explosion.

        Key principles:
        - Very small gains (0.01-0.2) for deep networks
        - Zero-init residual projections
        - Careful normalization layer init
        - Special handling for projections
        """
        # Detection head (1x1 conv): increased gain (v3.4.0 - trust normalization)
        nn.init.xavier_uniform_(self.detection_head.weight, gain=0.1)  # Was 0.01
        if self.detection_head.bias is not None:
            nn.init.constant_(self.detection_head.bias, 0)

        # Initialize projection layers with small gains
        if self.proj_to_electrodes is not None:
            nn.init.xavier_uniform_(self.proj_to_electrodes.weight, gain=0.1)
            if self.proj_to_electrodes.bias is not None:
                nn.init.zeros_(self.proj_to_electrodes.bias)

        if self.proj_from_electrodes is not None:
            # Residual-like projection: start near zero
            nn.init.xavier_uniform_(self.proj_from_electrodes.weight, gain=0.05)
            if self.proj_from_electrodes.bias is not None:
                nn.init.zeros_(self.proj_from_electrodes.bias)

        # Edge projection initialization (if present)
        if self.edge_in_proj is not None:
            nn.init.xavier_uniform_(self.edge_in_proj.weight, gain=0.5)  # v3.4.0: restored from 0.1
            if hasattr(self.edge_in_proj, "bias") and self.edge_in_proj.bias is not None:
                nn.init.zeros_(self.edge_in_proj.bias)

        if self.edge_out_proj is not None:
            nn.init.xavier_uniform_(
                self.edge_out_proj.weight, gain=0.5
            )  # v3.4.0: restored from 0.1
            if self.edge_out_proj.bias is not None:
                nn.init.zeros_(self.edge_out_proj.bias)

        # Initialize other conv layers conservatively
        for m in self.modules():
            if m is self.detection_head:
                continue  # Already initialized above
            # Skip projection layers already handled
            if hasattr(self, "proj_to_electrodes") and m is self.proj_to_electrodes:
                continue
            if hasattr(self, "proj_from_electrodes") and m is self.proj_from_electrodes:
                continue
            if hasattr(self, "edge_in_proj") and m is self.edge_in_proj:
                continue
            if hasattr(self, "edge_out_proj") and m is self.edge_out_proj:
                continue

            if isinstance(m, nn.Conv1d | nn.ConvTranspose1d):
                # v3.4.0: Trust Kaiming init (designed for ReLU)
                # Removed 5x scale-down; normalization layers handle stability
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # REMOVED: m.weight.data *= 0.2
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d | nn.LayerNorm | nn.GroupNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Small initialization for linear layers
                nn.init.xavier_uniform_(m.weight, gain=0.2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _run_node_stream(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run node stream: project to electrodes → BiMamba → normalize.

        Args:
            features: TCN output (B, 512, T)

        Returns:
            Tuple of (electrode_features, node_processed_features)
            - electrode_features: (B, 19, T, 64) for edge stream input
            - node_processed_features: (B, 19, T, 64) BiMamba output

        Notes:
            - Extracts per-electrode features from TCN bottleneck
            - Applies per-electrode BiMamba temporal modeling
            - Applies boundary normalization if configured
        """
        assert self.proj_to_electrodes is not None
        assert self.node_mamba is not None

        batch_size, _, seq_len = features.shape

        elec_flat = self.proj_to_electrodes(features)
        assert_finite("proj_to_electrodes", elec_flat)
        elec_feats = elec_flat.reshape(batch_size, 19, 64, seq_len).permute(0, 1, 3, 2)

        if self.norm_after_proj_to_electrodes:
            elec_feats = self.norm_after_proj_to_electrodes(elec_feats)

        node_flat = (
            elec_feats.permute(0, 1, 3, 2).reshape(batch_size * 19, 64, seq_len).contiguous()
        )
        node_processed = self.node_mamba(node_flat)
        assert_finite("node_mamba", node_processed)
        node_feats = node_processed.reshape(batch_size, 19, 64, seq_len).permute(0, 1, 3, 2)

        if self.norm_after_node_mamba:
            node_feats = self.norm_after_node_mamba(node_feats)

        return elec_feats, node_feats

    def _run_edge_stream(self, elec_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run edge stream: compute similarities → BiMamba → adjacency.

        Args:
            elec_feats: Electrode features (B, 19, T, 64)

        Returns:
            Tuple of (edge_weights, adjacency_matrix)
            - edge_weights: (B, 171, T) learned edge importances
            - adjacency_matrix: (B, T, 19, 19) assembled adjacency

        Notes:
            - Computes pairwise electrode similarities (171 edges)
            - Applies learned lift/project via BiMamba
            - Assembles adjacency via top-k + threshold
        """
        assert self.edge_in_proj is not None
        assert self.edge_mamba is not None
        assert self.edge_out_proj is not None
        assert self.edge_activate is not None

        from .edge_features import assemble_adjacency, edge_scalar_series

        batch_size, _, seq_len, _ = elec_feats.shape

        edge_metric = str(self.config.get("edge_metric", "cosine"))
        edge_similarity_margin = 0.01
        if "edge_similarity_margin" in self.config:
            margin_val = self.config["edge_similarity_margin"]
            if isinstance(margin_val, int | float):
                edge_similarity_margin = float(margin_val)

        edge_feats = edge_scalar_series(
            elec_feats, metric=edge_metric, edge_similarity_margin=edge_similarity_margin
        )

        if __debug__:
            lo, hi = edge_feats.amin(), edge_feats.amax()
            assert torch.isfinite(lo), "Non-finite minimum in edge features"
            assert torch.isfinite(hi), "Non-finite maximum in edge features"
            assert lo >= -1.001, f"Edge features lower bound violation: {lo}"
            assert hi <= 1.001, f"Edge features upper bound violation: {hi}"

        edge_flat = edge_feats.squeeze(-1).reshape(batch_size * 171, 1, seq_len)
        edge_in = self.edge_in_proj(edge_flat).contiguous()

        if hasattr(self, "edge_lift_act") and self.edge_lift_act is not None:
            edge_in = self.edge_lift_act(edge_in)
            if hasattr(self, "edge_lift_norm") and self.edge_lift_norm is not None:
                edge_in = edge_in.transpose(1, 2).contiguous()
                edge_in = self.edge_lift_norm(edge_in)
                edge_in = edge_in.transpose(1, 2).contiguous()
        else:
            edge_in = torch.clamp(edge_in, -3.0, 3.0)

        assert edge_in.is_contiguous(), "edge_in must be contiguous for Mamba"
        edge_processed = self.edge_mamba(edge_in)

        if self.norm_after_edge_mamba:
            edge_processed = edge_processed.transpose(1, 2).contiguous()
            edge_processed = self.norm_after_edge_mamba(edge_processed)
            edge_processed = edge_processed.transpose(1, 2).contiguous()

        edge_out = self.edge_out_proj(edge_processed)
        edge_weights = self.edge_activate(edge_out).reshape(batch_size, 171, seq_len)
        assert_finite("edge_weights", edge_weights)

        edge_top_k = cast(int, self.config.get("edge_top_k", 3))
        edge_threshold = cast(float, self.config.get("edge_threshold", EPSILON_LAPLACIAN))
        adj = assemble_adjacency(
            edge_weights, n_nodes=19, top_k=edge_top_k, threshold=edge_threshold
        )
        assert_finite("adjacency", adj)

        return edge_weights, adj

    def _apply_gnn_fusion(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Apply GNN with LayerScale residual and fusion.

        Args:
            node_feats: Node features (B, 19, T, 64)
            adj: Adjacency matrix (B, T, 19, 19)

        Returns:
            Enhanced features (B, 19, T, 64) after GNN + fusion

        Notes:
            - Applies GNN spatial mixing if enabled
            - Adds LayerScale residual connection
            - Applies gated fusion if configured
            - Normalizes after GNN if configured
        """
        elec_enhanced: torch.Tensor

        if self.gnn:
            gnn_out = self.gnn(node_feats, adj)
            if self.gnn_layerscale:
                gnn_out_scaled = self.gnn_layerscale(gnn_out)
                elec_enhanced = node_feats + gnn_out_scaled
            else:
                elec_enhanced = node_feats + gnn_out
        else:
            elec_enhanced = node_feats

        assert_finite("gnn_out", elec_enhanced)

        if self.norm_after_gnn:
            elec_enhanced = self.norm_after_gnn(elec_enhanced)

        if (
            self.fusion is not None
            and elec_enhanced is not node_feats
            and self.fusion_type in ("gated", "multihead")
        ):
            elec_enhanced = self.fusion(node_feats, elec_enhanced)

        return elec_enhanced

    def _decode_and_sanitize(self, temporal: torch.Tensor) -> torch.Tensor:
        """Decode to logits and apply final sanitization.

        Args:
            temporal: Temporal features (B, 512, T)

        Returns:
            Seizure logits (B, T) with NaN/Inf sanitization

        Notes:
            - Applies boundary normalization if configured
            - Projects to 19 channels and upsamples to original resolution
            - Clamps decoder features (tier 2) and logits (tier 3)
            - Applies nan_to_num for robust training
        """
        if self.norm_before_decoder:
            temporal = temporal.transpose(1, 2).contiguous()
            temporal = self.norm_before_decoder(temporal)
            temporal = temporal.transpose(1, 2).contiguous()

        decoded = self.proj_head(temporal)
        assert_finite("decoder_prelogits", decoded)

        decoded = torch.nan_to_num(decoded, nan=0.0, posinf=50.0, neginf=-50.0)
        decoded = torch.clamp(decoded, -50.0, 50.0)

        output = self.detection_head(decoded)
        assert_finite("final_logits", output)

        output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
        output = torch.clamp(output, -100.0, 100.0)

        return output.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through TCN + Bi-Mamba architecture.

        Refactored to use pipeline helpers for clarity and maintainability.
        Each helper encapsulates one processing stage.

        Args:
            x: (B, 19, 15360) EEG window tensor

        Returns:
            (B, 15360) per-sample seizure logits

        Notes:
            - forward reduced from 187 → ~60 lines via pipeline extraction
            - Helpers maintain identical behavior, just better organization
            - All intermediate assertions preserved for stability monitoring
        """
        features = self.tcn_encoder(x)
        assert_finite("tcn_out", features)

        if (
            self.node_mamba
            and self.edge_mamba
            and self.proj_to_electrodes
            and self.proj_from_electrodes
            and self.edge_in_proj
            and self.edge_out_proj
            and self.edge_activate
        ):
            batch_size, _, seq_len = features.shape

            elec_feats, node_feats = self._run_node_stream(features)
            _, adj = self._run_edge_stream(elec_feats)
            elec_enhanced = self._apply_gnn_fusion(node_feats, adj)

            elec_flat = elec_enhanced.permute(0, 1, 3, 2).reshape(batch_size, 19 * 64, seq_len)
            temporal = self.proj_from_electrodes(elec_flat)
            assert_finite("backproj", temporal)
        else:
            temporal = self.mamba(features)

        return self._decode_and_sanitize(temporal)

    @classmethod
    def from_config(
        cls,
        cfg: "_ModelConfig",
        warmup_schedule: "WarmupScheduleConfig | None" = None,
    ) -> "SeizureDetector":
        """Instantiate from validated schema config (V3).

        Refactored to use builder helpers for SRP compliance.
        Each builder encapsulates construction of one component family.

        Args:
            cfg: Model configuration
            warmup_schedule: Optional warmup schedule for gradient stabilization

        Returns:
            Configured SeizureDetector instance

        Notes:
            - from_config reduced from 199 → ~100 lines via builder extraction
            - Builders maintain identical behavior, just better organization
            - All attribute names unchanged (checkpoint compatibility preserved)
        """
        instance = cls(
            tcn_layers=cfg.tcn.num_layers,
            tcn_kernel_size=cfg.tcn.kernel_size,
            tcn_dropout=cfg.tcn.dropout,
            tcn_stride=cfg.tcn.stride_down,
            mamba_layers=cfg.mamba.n_layers,
            mamba_d_state=cfg.mamba.d_state,
            mamba_d_conv=cfg.mamba.conv_kernel,
            mamba_dropout=cfg.mamba.dropout,
        )

        instance.architecture = cfg.architecture
        instance.config["architecture"] = cfg.architecture

        if cfg.architecture == "v3":
            graph_cfg = cfg.graph

            instance.node_mamba = build_node_stream(cfg)

            edge_components = build_edge_stream(cfg)
            instance.edge_mamba = edge_components.edge_mamba
            instance.edge_in_proj = edge_components.edge_in_proj
            instance.edge_out_proj = edge_components.edge_out_proj
            instance.edge_activate = edge_components.edge_activate
            instance.edge_lift_act = edge_components.edge_lift_act
            instance.edge_lift_norm = edge_components.edge_lift_norm

            instance.proj_to_electrodes = nn.Conv1d(512, 19 * 64, kernel_size=1)
            instance.proj_from_electrodes = nn.Conv1d(19 * 64, 512, kernel_size=1)

            if graph_cfg:
                instance.config["edge_metric"] = graph_cfg.edge_features
                instance.config["edge_similarity_margin"] = graph_cfg.edge_similarity_margin
                instance.config["edge_top_k"] = graph_cfg.edge_top_k
                instance.config["edge_threshold"] = graph_cfg.edge_threshold

            edge_d_model = graph_cfg.edge_mamba_d_model if graph_cfg else 16
            reg_components = build_regularizers(cfg, edge_d_model)
            instance.norm_after_proj_to_electrodes = reg_components.norm_after_proj_to_electrodes
            instance.norm_after_node_mamba = reg_components.norm_after_node_mamba
            instance.norm_after_edge_mamba = reg_components.norm_after_edge_mamba
            instance.norm_after_gnn = reg_components.norm_after_gnn
            instance.norm_before_decoder = reg_components.norm_before_decoder
            instance.gnn_layerscale = reg_components.gnn_layerscale

        instance.fusion_type, instance.fusion = build_fusion_head(cfg)

        graph_cfg = getattr(cfg, "graph", None)
        instance.use_gnn = bool(graph_cfg and graph_cfg.enabled)

        if instance.use_gnn and graph_cfg is not None:
            try:
                from .gnn_pyg import GraphChannelMixerPyG

                is_v3 = True
                instance.gnn = GraphChannelMixerPyG(
                    d_model=64,
                    n_electrodes=19,
                    k_eigenvectors=graph_cfg.k_eigenvectors,
                    alpha=graph_cfg.alpha,
                    k_hops=2,
                    n_layers=graph_cfg.n_layers,
                    dropout=graph_cfg.dropout,
                    use_residual=False,
                    use_vectorized=is_v3,
                    use_dynamic_pe=graph_cfg.use_dynamic_pe,
                    bypass_edge_transform=is_v3,
                    semi_dynamic_interval=graph_cfg.semi_dynamic_interval,
                    pe_sign_consistency=graph_cfg.pe_sign_consistency,
                    adj_row_softmax=graph_cfg.adj_row_softmax,
                    adj_softmax_tau=graph_cfg.adj_softmax_tau,
                    adj_ema_beta=graph_cfg.adj_ema_beta,
                    adj_force_symmetric=graph_cfg.adj_force_symmetric,
                    laplacian_eps=graph_cfg.laplacian_eps,
                    laplacian_normalize=graph_cfg.laplacian_normalize,
                    warmup_config=warmup_schedule,
                )
            except ImportError as e:
                raise ImportError(
                    "PyTorch Geometric not installed. GNN requires PyG. "
                    "Install from prebuilt wheels for torch 2.5.0+cu124 "
                    "(see INSTALLATION.md) or run 'make setup-gpu'"
                ) from e

        return instance

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

        # Provide parameter info with legacy keys for tests
        info: dict[str, object] = {
            "tcn_params": tcn_params,
            "proj_params": proj_params,
            "mamba_params": mamba_params,
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
