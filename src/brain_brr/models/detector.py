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

from .debug_utils import assert_finite
from .mamba import BiMamba2
from .norms import LayerScale, create_norm_layer
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

    def _initialize_weights(self) -> None:
        """Initialize weights with conservative gains to prevent NaN/explosion.

        Key principles:
        - Very small gains (0.01-0.2) for deep networks
        - Zero-init residual projections
        - Careful normalization layer init
        - Special handling for projections
        """
        # Detection head (1x1 conv): very small to prevent saturation
        nn.init.xavier_uniform_(self.detection_head.weight, gain=0.01)
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
            nn.init.xavier_uniform_(self.edge_in_proj.weight, gain=0.1)  # Reduced from 0.5
            if hasattr(self.edge_in_proj, "bias") and self.edge_in_proj.bias is not None:
                nn.init.zeros_(self.edge_in_proj.bias)

        if self.edge_out_proj is not None:
            nn.init.xavier_uniform_(self.edge_out_proj.weight, gain=0.1)  # Reduced from 0.5
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

            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                # Conservative initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                m.weight.data *= 0.2  # Scale down by 5x
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Small initialization for linear layers
                nn.init.xavier_uniform_(m.weight, gain=0.2)
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
        assert_finite("tcn_out", features)
        # Optional safety clamp after TCN
        from src.brain_brr.utils.env import env as _env

        if _env.safe_clamp():
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            features = torch.clamp(features, _env.safe_clamp_min(), _env.safe_clamp_max())

        # V3 dual-stream if components are present
        if (
            self.node_mamba
            and self.edge_mamba
            and self.proj_to_electrodes
            and self.proj_from_electrodes
            and self.edge_in_proj
            and self.edge_out_proj
            and self.edge_activate
        ):
            # V3: Dual-stream architecture with learned adjacency
            from .edge_features import assemble_adjacency, edge_scalar_series

            batch_size, _, seq_len = features.shape

            # Project to electrode features
            elec_flat = self.proj_to_electrodes(features)  # (B, 19*64, 960)
            assert_finite("proj_to_electrodes", elec_flat)
            elec_feats = elec_flat.reshape(batch_size, 19, 64, seq_len).permute(
                0, 1, 3, 2
            )  # (B, 19, 960, 64)

            # PR-1: Normalize after projection to electrodes
            if self.norm_after_proj_to_electrodes:
                elec_feats = self.norm_after_proj_to_electrodes(elec_feats)

            # Node stream: per-electrode Mamba
            node_flat = (
                elec_feats.permute(0, 1, 3, 2).reshape(batch_size * 19, 64, seq_len).contiguous()
            )  # (B*19, 64, 960) - ensure contiguous for CUDA
            node_processed = self.node_mamba(node_flat)  # (B*19, 64, 960)
            assert_finite("node_mamba", node_processed)
            node_feats = node_processed.reshape(batch_size, 19, 64, seq_len).permute(
                0, 1, 3, 2
            )  # (B, 19, 960, 64)

            # PR-1: Normalize after node Mamba
            if self.norm_after_node_mamba:
                node_feats = self.norm_after_node_mamba(node_feats)

            # Edge stream: learned adjacency
            edge_metric = str(self.config.get("edge_metric", "cosine"))
            edge_feats = edge_scalar_series(elec_feats, metric=edge_metric)  # (B, 171, 960, 1)

            # Clamp cosine similarities to avoid extreme values
            edge_feats = torch.clamp(edge_feats, -0.99, 0.99)

            # Learnable lift 1→D channels for CUDA alignment & capacity
            edge_flat = edge_feats.squeeze(-1).reshape(batch_size * 171, 1, seq_len)  # (B*E,1,T)
            edge_in = self.edge_in_proj(edge_flat).contiguous()  # (B*E, D, T) where D=16

            # PR-2: Apply bounded activation and normalization
            if hasattr(self, "edge_lift_act") and self.edge_lift_act is not None:
                edge_in = self.edge_lift_act(edge_in)

                # Apply normalization after activation if configured
                if hasattr(self, "edge_lift_norm") and self.edge_lift_norm is not None:
                    # Transpose for LayerNorm on feature dimension
                    edge_in = edge_in.transpose(1, 2).contiguous()  # (B*E, T, D)
                    edge_in = self.edge_lift_norm(edge_in)
                    edge_in = edge_in.transpose(1, 2).contiguous()  # Back to (B*E, D, T)
            else:
                # Fallback: Keep original clamp if PR-2 not enabled
                edge_in = torch.clamp(edge_in, -3.0, 3.0)

            # Safety assertion for Mamba CUDA kernel
            assert edge_in.is_contiguous(), (
                "edge_in tensor must be contiguous for Mamba CUDA kernels"
            )

            edge_processed = self.edge_mamba(edge_in)  # (B*E, D, T)

            # PR-1: Normalize after edge Mamba (permute for LayerNorm on last dim)
            if self.norm_after_edge_mamba:
                # edge_processed is (B*E, D, T), need to normalize over D dimension
                edge_processed = edge_processed.transpose(1, 2).contiguous()  # (B*E, T, D)
                edge_processed = self.norm_after_edge_mamba(edge_processed)
                edge_processed = edge_processed.transpose(1, 2).contiguous()  # Back to (B*E, D, T)
            edge_out = self.edge_out_proj(edge_processed)  # (B*E, 1, T)
            edge_weights = self.edge_activate(edge_out).reshape(batch_size, 171, seq_len)  # (B,E,T)
            assert_finite("edge_weights", edge_weights)

            # Assemble adjacency
            edge_top_k = cast(int, self.config.get("edge_top_k", 3))
            edge_threshold = cast(float, self.config.get("edge_threshold", 1e-4))
            adj = assemble_adjacency(
                edge_weights,
                n_nodes=19,
                top_k=edge_top_k,
                threshold=edge_threshold,
            )  # (B, 960, 19, 19)
            assert_finite("adjacency", adj)

            # Apply GNN with optional LayerScale residual
            if self.gnn:
                gnn_out = self.gnn(node_feats, adj)
                # PR-1: Apply LayerScale to GNN residual if configured
                if self.gnn_layerscale and self.gnn.use_residual:
                    # GNN already adds residual internally, so we scale the increment
                    # gnn_out = node_feats + scale * (gnn_out - node_feats)
                    gnn_increment = gnn_out - node_feats
                    elec_enhanced = node_feats + self.gnn_layerscale(gnn_increment)
                else:
                    elec_enhanced = gnn_out
            else:
                elec_enhanced = node_feats
            assert_finite("gnn_out", elec_enhanced)

            # PR-1: Normalize after GNN
            if self.norm_after_gnn:
                elec_enhanced = self.norm_after_gnn(elec_enhanced)

            # Project back to bottleneck
            elec_flat = elec_enhanced.permute(0, 1, 3, 2).reshape(
                batch_size, 19 * 64, seq_len
            )  # (B, 19*64, 960)
            temporal = self.proj_from_electrodes(elec_flat)  # (B, 512, 960)
            assert_finite("backproj", temporal)

        else:
            # Fallback to 512-dim Mamba stack if V3 components not initialized
            temporal = self.mamba(features)  # (B, 512, 960)

        # PR-1: Normalize before decoder (temporal is B, 512, 960)
        if self.norm_before_decoder:
            # Need to permute to make 512 the last dimension for LayerNorm
            temporal = temporal.transpose(1, 2).contiguous()  # (B, 960, 512)
            temporal = self.norm_before_decoder(temporal)
            temporal = temporal.transpose(1, 2).contiguous()  # Back to (B, 512, 960)

        # Optional safety clamp after temporal modeling
        if _env.safe_clamp():
            temporal = torch.nan_to_num(temporal, nan=0.0, posinf=0.0, neginf=0.0)
            temporal = torch.clamp(temporal, _env.safe_clamp_min(), _env.safe_clamp_max())

        # Project back to 19 channels and upsample to original resolution
        decoded = self.proj_head(temporal)  # (B, 19, 15360)
        assert_finite("decoder_prelogits", decoded)

        # Internal tier clamping for features before final projection
        decoded = torch.nan_to_num(decoded, nan=0.0, posinf=50.0, neginf=-50.0)
        decoded = torch.clamp(decoded, -50.0, 50.0)

        output = self.detection_head(decoded)  # (B, 1, 15360)
        assert_finite("final_logits", output)

        # Final output sanitization to prevent non-finite logits
        output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
        output = torch.clamp(output, -100.0, 100.0)  # Tier 3: Output clamping for loss

        return cast(torch.Tensor, output.squeeze(1))

    @classmethod
    def from_config(cls, cfg: "_ModelConfig") -> "SeizureDetector":
        """Instantiate from validated schema config (V3)."""

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

        # Set architecture
        instance.architecture = cfg.architecture
        instance.config["architecture"] = cfg.architecture

        # Build v3-specific components if v3 architecture
        if cfg.architecture == "v3":
            # V3: Dual-stream architecture
            graph_cfg = cfg.graph  # Required for v3

            # Node stream: per-electrode Mamba
            # headdim=8 ensures (64 * 2) / 8 = 16 which is multiple of 8
            # PR-1: Pass LayerScale config to Mamba layers
            norms_cfg = getattr(cfg, "norms", None)
            use_layerscale_mamba = bool(norms_cfg and norms_cfg.boundary_norm != "none")
            layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else 0.1)

            instance.node_mamba = BiMamba2(
                d_model=64,
                d_state=16,  # Fixed for node stream
                d_conv=4,
                expand=2,
                headdim=8,  # Critical: (64*2)/8 = 16 is multiple of 8
                num_layers=6,  # Fixed for node stream
                dropout=cfg.mamba.dropout,
                use_layerscale=use_layerscale_mamba,
                layerscale_init=layerscale_init,
            )

            # Edge stream: per-edge Mamba (learned lift 1→D→1)
            edge_layers = graph_cfg.edge_mamba_layers if graph_cfg else 2
            edge_d_state = graph_cfg.edge_mamba_d_state if graph_cfg else 8
            edge_d_model = graph_cfg.edge_mamba_d_model if graph_cfg else 16

            # Safety assertions for CUDA kernel alignment
            assert edge_d_model % 8 == 0, (
                f"edge_mamba_d_model must be multiple of 8 for CUDA, got {edge_d_model}"
            )
            assert edge_d_model > 0, f"edge_mamba_d_model must be positive, got {edge_d_model}"

            # headdim=4 ensures (16 * 2) / 4 = 8 which is multiple of 8
            instance.edge_mamba = BiMamba2(
                d_model=edge_d_model,
                d_state=edge_d_state,
                d_conv=4,
                expand=2,
                headdim=4,  # Critical: (16*2)/4 = 8 is multiple of 8
                num_layers=edge_layers,
                dropout=cfg.mamba.dropout,
                use_layerscale=use_layerscale_mamba,
                layerscale_init=layerscale_init,
            )

            # Edge stream projections (learned lift/project) + activation
            instance.edge_in_proj = nn.Conv1d(1, edge_d_model, kernel_size=1, bias=False)
            instance.edge_out_proj = nn.Conv1d(edge_d_model, 1, kernel_size=1, bias=True)
            instance.edge_activate = nn.Softplus()

            # PR-2: Bounded edge stream components
            edge_lift_activation = graph_cfg.edge_lift_activation if graph_cfg else "none"
            edge_lift_norm = graph_cfg.edge_lift_norm if graph_cfg else "none"
            edge_lift_gain = graph_cfg.edge_lift_init_gain if graph_cfg else 0.1

            # Create activation function
            if edge_lift_activation == "tanh":
                instance.edge_lift_act = nn.Tanh()
            elif edge_lift_activation == "sigmoid":
                instance.edge_lift_act = nn.Sigmoid()
            elif edge_lift_activation == "selu":
                instance.edge_lift_act = nn.SELU()
            else:
                instance.edge_lift_act = None

            # Create normalization layer
            instance.edge_lift_norm = create_norm_layer(edge_lift_norm, edge_d_model)

            # Initialize edge projections with configured gain
            nn.init.xavier_uniform_(instance.edge_in_proj.weight, gain=edge_lift_gain)
            if instance.edge_out_proj.bias is not None:
                nn.init.zeros_(instance.edge_out_proj.bias)
            nn.init.xavier_uniform_(instance.edge_out_proj.weight, gain=edge_lift_gain)

            # Projections for electrode space
            instance.proj_to_electrodes = nn.Conv1d(512, 19 * 64, kernel_size=1)
            instance.proj_from_electrodes = nn.Conv1d(19 * 64, 512, kernel_size=1)

            # Store edge config in instance.config
            if graph_cfg:
                instance.config["edge_metric"] = graph_cfg.edge_features
                instance.config["edge_top_k"] = graph_cfg.edge_top_k
                instance.config["edge_threshold"] = graph_cfg.edge_threshold

            # PR-1: Initialize boundary normalization layers if configured
            norms_cfg = getattr(cfg, "norms", None)
            if norms_cfg and norms_cfg.boundary_norm != "none":
                # Create normalization layers at component boundaries
                if norms_cfg.after_tcn_proj:
                    instance.norm_after_proj_to_electrodes = create_norm_layer(
                        norms_cfg.boundary_norm, 64, norms_cfg.boundary_eps
                    )
                if norms_cfg.after_node_mamba:
                    instance.norm_after_node_mamba = create_norm_layer(
                        norms_cfg.boundary_norm, 64, norms_cfg.boundary_eps
                    )
                if norms_cfg.after_edge_mamba:
                    # Edge stream has 16 dimensions
                    instance.norm_after_edge_mamba = create_norm_layer(
                        norms_cfg.boundary_norm, edge_d_model, norms_cfg.boundary_eps
                    )
                if norms_cfg.after_gnn:
                    instance.norm_after_gnn = create_norm_layer(
                        norms_cfg.boundary_norm, 64, norms_cfg.boundary_eps
                    )
                if norms_cfg.before_decoder:
                    instance.norm_before_decoder = create_norm_layer(
                        norms_cfg.boundary_norm, 512, norms_cfg.boundary_eps
                    )

                # Initialize LayerScale for GNN residual if configured
                if graph_cfg and graph_cfg.use_residual:
                    instance.gnn_layerscale = LayerScale(64, norms_cfg.layerscale_alpha)

        # Optionally attach GNN components if enabled
        graph_cfg = getattr(cfg, "graph", None)
        instance.use_gnn = bool(graph_cfg and graph_cfg.enabled)

        if instance.use_gnn and graph_cfg is not None:
            # ONLY PyG implementation with Laplacian PE is supported
            try:
                from .gnn_pyg import GraphChannelMixerPyG

                # V3 uses vectorized GNN with configurable PE
                is_v3 = True
                instance.gnn = GraphChannelMixerPyG(
                    d_model=64,  # Per-electrode feature dimension
                    n_electrodes=19,
                    k_eigenvectors=graph_cfg.k_eigenvectors,
                    alpha=graph_cfg.alpha,
                    k_hops=2,  # 2-hop neighborhood
                    n_layers=graph_cfg.n_layers,
                    dropout=graph_cfg.dropout,
                    use_residual=graph_cfg.use_residual,
                    use_vectorized=is_v3,  # V3: vectorized batching
                    use_dynamic_pe=graph_cfg.use_dynamic_pe,  # Configurable PE mode
                    bypass_edge_transform=is_v3,  # V3: skip since we have Softplus upstream
                    semi_dynamic_interval=graph_cfg.semi_dynamic_interval,
                    pe_sign_consistency=graph_cfg.pe_sign_consistency,
                )
            except ImportError as e:
                raise ImportError(
                    "PyTorch Geometric not installed. GNN requires PyG. Install from prebuilt wheels for torch 2.2.2+cu121 (see INSTALLATION.md) or run 'make setup-gpu'"
                ) from e

            # V3 creates projections above; no V2 heuristic path remains

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
