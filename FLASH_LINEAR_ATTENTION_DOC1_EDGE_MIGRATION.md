# Doc 1: Edge Stream Migration - Implementation Plan

**Parent Document**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md) (Doc 0 - SSOT)
**Phase**: 1a (HIGHEST PRIORITY)
**Target**: Replace Edge Stream BiMamba2 → BiGatedDeltaNet
**Date**: October 7, 2025
**Version**: 1.0
**Status**: Ready for Implementation

---

## Executive Summary

This document provides **surgical implementation details** for migrating the edge stream (and ONLY the edge stream) from BiMamba2 to BiGatedDeltaNet. This is Phase 1a of the phased migration strategy.

**Scope of Changes**:
- ✅ Create `src/brain_brr/models/gated_deltanet.py` (BiGatedDeltaNet wrapper)
- ✅ Update `src/brain_brr/models/builders/edge_stream.py` (return BiGatedDeltaNet)
- ✅ Add config schema support in `src/brain_brr/config/schemas.py`
- ✅ Write unit tests (`tests/unit/models/test_gated_deltanet.py`)
- ✅ Write integration test (`tests/integration/test_edge_gdn_migration.py`)
- ❌ **DO NOT TOUCH**: Node stream, GNN, TCN, decoder (keep v3.8.3 baseline)

**Expected Outcome**:
- Edge stream: **10.3K params BiMamba2 → ~7.3K params BiGatedDeltaNet** (29% reduction due to 0.75× q/k projection)
- Hypothesis: +5-10% better connectivity modeling → +1-2% sensitivity @ 1 FA/24h
- Risk: **VERY LOW** (only ~1.8% of stream parameters affected, 7.3K out of 405K total)

**Timeline**: 2-3 days development + 6-8 hours integration test + 1 day analysis

---

## 📊 Parameter Count Analysis

**IMPORTANT**: GDN's 0.75× q/k projection (vs Mamba2's 1.0×) reduces parameter count by ~29%. **This is EXPECTED and BENEFICIAL**:

| Component | BiMamba2 (Baseline) | BiGatedDeltaNet (Phase 1a) | Reduction |
|-----------|---------------------|----------------------------|-----------|
| **Edge Stream** | 10,304 params | **~7,352 params** | **-29%** |
| **Node Stream** | 397,632 params | (unchanged - Phase 1b) | N/A |
| **Total Streams** | 407,936 params | **~404,984 params** | **-0.7%** |

**Why fewer parameters is GOOD**:
- ✅ **More parameter-efficient**: Same representational capacity with fewer params
- ✅ **Faster inference**: Fewer parameters = faster forward pass
- ✅ **Better generalization**: Reduced parameter count can improve generalization
- ✅ **By design**: GDN paper shows 0.75× allocation is intentional and performs well

**Key Insight**: The 0.75× reduction is part of GDN's **parameter efficiency design**, not a regression. Language models achieve +3.1% LongBench improvement DESPITE having fewer params.

---

## 1. Prerequisites

### 1.1. Environment Setup

```bash
# Install FLA library
pip install flash-linear-attention

# Verify dependencies
python -c "import torch; print(f'PyTorch: {torch.__version__}')"  # Should be 2.5.0+cu124
python -c "import triton; print(f'Triton: {triton.__version__}')"  # Should be 3.0.0+
python -c "from fla.layers import GatedDeltaNet; print('FLA OK')"

# Backup current state
git checkout -b feature/edge-gdn-migration
git tag v3.8.3-pre-edge-migration
```

### 1.2. Verify Current Edge Stream

```bash
# Run baseline to establish metrics
python -c "
from src.brain_brr.models.builders.edge_stream import build_edge_stream
from src.brain_brr.config.schemas import ModelConfig

cfg = ModelConfig()
edge_components = build_edge_stream(cfg)
print(f'Edge Mamba params: {sum(p.numel() for p in edge_components.edge_mamba.parameters()):,}')
print(f'd_model: {edge_components.edge_mamba.d_model}')
print(f'headdim: {edge_components.edge_mamba.headdim}')
print(f'num_layers: {edge_components.edge_mamba.num_layers}')
"
# Expected output (BiMamba2 baseline):
# Edge Mamba params: 10,304
# d_model: 16
# headdim: 4
# num_layers: 2

# Expected output (BiGatedDeltaNet - after migration):
# Edge GDN params: ~7,352 (29% reduction due to GDN's 0.75× q/k projection)
```

---

## 2. Implementation: BiGatedDeltaNet Wrapper

### 2.1. File: `src/brain_brr/models/gated_deltanet.py`

**Purpose**: Bidirectional wrapper around FLA's GatedDeltaNet, compatible with our BiMamba2 interface.

**Implementation**:

```python
"""Bidirectional Gated DeltaNet wrapper for EEG seizure detection.

Replaces BiMamba2 with FLA's GatedDeltaNet while maintaining interface compatibility.
This is a SHARED module that processes flattened (B*N, d_model, T) tensors.
"""

import logging
import torch
import torch.nn as nn

try:
    from fla.layers import GatedDeltaNet as FLAGatedDeltaNet
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False

logger = logging.getLogger(__name__)


class BiGatedDeltaNet(nn.Module):
    """Bidirectional Gated DeltaNet wrapper for EEG seizure detection.

    Wraps FLA's GatedDeltaNet with bidirectional processing similar to BiMamba2.
    IMPORTANT: This is a SHARED module that processes flattened (B*N, d_model, T) tensors,
    NOT separate instances per electrode/pair.

    Args:
        d_model: Model dimension (16 for edge stream, 64 for node stream)
        headdim: Head dimension (4 for edge, 8 for node - MUST satisfy 0.75× constraint)
        num_layers: Number of bidirectional layers (2 for edge, 6 for node)
        dropout: Dropout after fusion (0.1 default)
        fusion_mode: 'sum' or 'concat' (A/B test both!)
        allow_neg_eigval: Research feature for β_t ∈ (0,2) (start False)
        use_layerscale: Enable LayerScale on residuals (match BiMamba2)
        layerscale_init: LayerScale initial value (match BiMamba2)

    Raises:
        ImportError: If FLA library not installed
        AssertionError: If headdim doesn't satisfy 0.75× constraint
    """

    def __init__(
        self,
        d_model: int = 16,
        headdim: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        fusion_mode: str = 'sum',
        allow_neg_eigval: bool = False,
        use_layerscale: bool = False,
        layerscale_init: float = 0.1,
        **kwargs
    ):
        super().__init__()

        if not FLA_AVAILABLE:
            raise ImportError(
                "FLA library not installed. Run: pip install flash-linear-attention"
            )

        self.d_model = d_model
        self.headdim = headdim
        self.num_layers = num_layers
        self.fusion_mode = fusion_mode

        # CONSTRAINT: num_heads × head_dim = 0.75 × hidden_size
        # This is due to GDN's 0.75× q/k projection vs Mamba2's 1.0×
        constraint_value = d_model * 0.75
        assert constraint_value % headdim == 0, (
            f"Invalid headdim={headdim} for d_model={d_model}: "
            f"num_heads × head_dim must equal {constraint_value} (0.75 × hidden_size). "
            f"Valid headdim values: {[int(constraint_value / i) for i in range(1, 13) if constraint_value % i == 0]}"
        )
        num_heads = int(constraint_value / headdim)

        logger.info(
            f"BiGatedDeltaNet init: d_model={d_model}, headdim={headdim}, "
            f"num_heads={num_heads}, num_layers={num_layers}, fusion_mode={fusion_mode}"
        )
        logger.debug(
            f"GDN constraint satisfied: {num_heads} × {headdim} = "
            f"{num_heads * headdim} = 0.75 × {d_model}"
        )

        # Create bidirectional GDN layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.ModuleDict({
                'forward': FLAGatedDeltaNet(
                    hidden_size=d_model,
                    head_dim=headdim,
                    num_heads=num_heads,
                    expand_v=2.0,              # Match BiMamba2 expand
                    mode='chunk',              # REQUIRED for training
                    use_short_conv=True,       # CRUCIAL (ablation: 5.6% drop if False)
                    conv_size=4,               # Match BiMamba2 d_conv
                    use_gate=True,             # CRUCIAL (ablation: 6.5% drop if False)
                    allow_neg_eigval=allow_neg_eigval,  # Start False
                    conv_bias=False,           # Match BiMamba2 (no bias)
                    norm_eps=1e-5,             # Match BiMamba2 RMSNorm eps
                ),
                'backward': FLAGatedDeltaNet(
                    hidden_size=d_model,
                    head_dim=headdim,
                    num_heads=num_heads,
                    expand_v=2.0,
                    mode='chunk',
                    use_short_conv=True,
                    conv_size=4,
                    use_gate=True,
                    allow_neg_eigval=allow_neg_eigval,
                    conv_bias=False,
                    norm_eps=1e-5,
                ),
            })
            self.layers.append(layer)

        self.dropout = nn.Dropout(dropout)

        # Fusion projection (only if concat mode)
        if fusion_mode == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model, bias=False)
            nn.init.xavier_uniform_(self.fusion_proj.weight, gain=0.2)  # Conservative init
            logger.debug(f"Using concat fusion with projection ({d_model*2} → {d_model})")
        else:
            self.fusion_proj = None
            logger.debug(f"Using sum fusion (no projection)")

        # LayerScale (optional, to match BiMamba2)
        if use_layerscale:
            from src.brain_brr.models.norms import LayerScale
            self.layerscale = LayerScale(d_model, init_value=layerscale_init)
            logger.debug(f"LayerScale enabled (init={layerscale_init})")
        else:
            self.layerscale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional processing: forward + backward (flipped).

        Args:
            x: (B, C, L) where:
               - B can be B*171 for edge stream or B*19 for node stream
               - C = d_model (16 for edge, 64 for node)
               - L = 960 (sequence length after TCN downsampling)

        Returns:
            x: (B, C, L) bidirectional output
        """
        # Transpose to sequence-first for GDN: (B, L, C)
        x = x.transpose(1, 2).contiguous()

        for layer in self.layers:
            residual = x  # Save for residual connection

            # Forward pass
            x_fwd, _, _ = layer['forward'](x)

            # Backward pass (flip sequence)
            x_bwd, _, _ = layer['backward'](x.flip(dims=[1]).contiguous())
            x_bwd = x_bwd.flip(dims=[1])  # Flip back to original order

            # Fusion: sum or concat
            if self.fusion_mode == 'sum':
                # Additive fusion (lower capacity, fewer params)
                x = x_fwd + x_bwd
            else:  # concat
                # Concatenative fusion (higher capacity, more params)
                x = torch.cat([x_fwd, x_bwd], dim=-1)  # (B, L, 2C)
                x = self.fusion_proj(x)  # (B, L, C)

            # Apply LayerScale if enabled
            if self.layerscale is not None:
                x = self.layerscale(x)

            # Dropout + residual
            x = residual + self.dropout(x)

        # Transpose back to channel-first: (B, C, L)
        return x.transpose(1, 2).contiguous()

    def __repr__(self) -> str:
        return (
            f"BiGatedDeltaNet(d_model={self.d_model}, headdim={self.headdim}, "
            f"num_layers={self.num_layers}, fusion_mode={self.fusion_mode})"
        )
```

**Testing the wrapper**:

```python
# Quick sanity check
python -c "
from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet
import torch

# Edge stream config
edge_gdn = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2, dropout=0.1)
print(f'Params: {sum(p.numel() for p in edge_gdn.parameters()):,}')

# Test forward pass
x = torch.randn(8*171, 16, 960)  # (B*171, 16, 960) - edge stream
y = edge_gdn(x)
print(f'Input: {x.shape}, Output: {y.shape}')
assert x.shape == y.shape, 'Shape mismatch!'
print('✅ Wrapper works!')
"
```

---

## 3. Implementation: Edge Stream Builder Update

### 3.1. File: `src/brain_brr/models/builders/edge_stream.py`

**Changes Required**:

1. Import BiGatedDeltaNet
2. Conditionally build BiGatedDeltaNet or BiMamba2 based on config
3. Update EdgeStreamComponents to handle both types

**Implementation**:

```python
"""Edge stream builder - per-edge BiMamba/BiGatedDeltaNet component with learned lift/project."""

import logging
from typing import TYPE_CHECKING

import torch.nn as nn

from src.brain_brr.constants import LAYERSCALE_ALPHA_FALLBACK

from ..mamba import BiMamba2
from ..norms import create_norm_layer

# Import GDN conditionally
try:
    from ..gated_deltanet import BiGatedDeltaNet
    GDN_AVAILABLE = True
except ImportError:
    GDN_AVAILABLE = False

if TYPE_CHECKING:
    from src.brain_brr.config.schemas import ModelConfig

logger = logging.getLogger(__name__)


class EdgeStreamComponents:
    """Container for edge stream components (avoids tuple unpacking)."""

    def __init__(
        self,
        edge_mamba: BiMamba2 | "BiGatedDeltaNet",  # Can be either type
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
    """Build edge stream: per-edge BiMamba/BiGatedDeltaNet with learned lift/project.

    V3 Architecture: Processes edge similarities (171 pairs) with BiMamba or BiGatedDeltaNet.
    Pipeline: 1D → lift(d_model) → BiMamba/GDN → project(1D) → Softplus

    Args:
        cfg: Model configuration containing graph and norms settings

    Returns:
        EdgeStreamComponents with all edge processing modules

    Notes:
        - edge_d_model must be multiple of 8 for CUDA alignment
        - headdim=4 ensures (16 * 2) / 4 = 8 which is multiple of 8 (BiMamba2)
        - For GDN: headdim=4, num_heads=3 (3 × 4 = 12 = 0.75 × 16)
        - PR-2: Supports bounded edge stream (activation + norm on lift)
        - LayerScale enabled if boundary_norm != "none"
    """
    graph_cfg = cfg.graph
    norms_cfg = getattr(cfg, "norms", None)
    mamba_cfg = cfg.mamba

    edge_layers = graph_cfg.edge_mamba_layers if graph_cfg else 2
    edge_d_state = graph_cfg.edge_mamba_d_state if graph_cfg else 8
    edge_d_model = graph_cfg.edge_mamba_d_model if graph_cfg else 16

    assert edge_d_model % 8 == 0, (
        f"edge_mamba_d_model must be multiple of 8 for CUDA, got {edge_d_model}"
    )
    assert edge_d_model > 0, f"edge_mamba_d_model must be positive, got {edge_d_model}"

    use_layerscale = bool(norms_cfg and norms_cfg.boundary_norm != "none")
    layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)

    # Determine which temporal model to use (STREAM-SPECIFIC for Phase 1a isolation)
    # Priority: temporal_type_edge > temporal_type (fallback)
    temporal_type = getattr(mamba_cfg, "temporal_type_edge", None)
    if temporal_type is None:
        temporal_type = getattr(mamba_cfg, "temporal_type", "bimamba2")

    logger.debug(f"Edge stream temporal_type: {temporal_type} (stream-specific or fallback)")

    if temporal_type == "gated_deltanet":
        if not GDN_AVAILABLE:
            raise ImportError(
                "Gated DeltaNet requested but not available. "
                "Install FLA: pip install flash-linear-attention"
            )

        # Build BiGatedDeltaNet for edge stream
        edge_mamba = BiGatedDeltaNet(
            d_model=edge_d_model,
            headdim=4,  # 3 × 4 = 12 = 0.75 × 16 ✅
            num_layers=edge_layers,
            dropout=mamba_cfg.dropout,
            fusion_mode=getattr(mamba_cfg, "fusion_mode", "sum"),
            allow_neg_eigval=getattr(mamba_cfg, "allow_neg_eigval", False),
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
        )
        logger.info(f"Edge stream: BiGatedDeltaNet (d_model={edge_d_model}, headdim=4, layers={edge_layers})")
    else:
        # Build BiMamba2 for edge stream (default/baseline)
        edge_mamba = BiMamba2(
            d_model=edge_d_model,
            d_state=edge_d_state,
            d_conv=4,
            expand=2,
            headdim=4,
            num_layers=edge_layers,
            dropout=mamba_cfg.dropout,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
        )
        logger.info(f"Edge stream: BiMamba2 (d_model={edge_d_model}, headdim=4, layers={edge_layers})")

    # Lift/project layers (unchanged)
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
```

---

## 4. Implementation: Config Schema Update

### 4.1. File: `src/brain_brr/config/schemas.py`

**Add to MambaConfig**:

```python
class MambaConfig(BaseModel):
    """Mamba/GDN configuration."""
    n_layers: int = Field(default=6, ge=1, le=12)
    d_model: int = Field(default=512, ge=64, le=2048)
    d_state: int = Field(default=16, ge=8, le=64)
    conv_kernel: int = Field(default=4, ge=2, le=4)
    dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    # NEW: Temporal model type
    temporal_type: str = Field(
        default="bimamba2",
        description="Temporal model type: 'bimamba2' or 'gated_deltanet'"
    )

    # NEW: GDN-specific settings
    fusion_mode: str = Field(
        default="sum",
        description="Bidirectional fusion: 'sum' or 'concat'"
    )
    allow_neg_eigval: bool = Field(
        default=False,
        description="Allow negative eigenvalues (β_t ∈ (0,2))"
    )

    @field_validator("temporal_type")
    @classmethod
    def validate_temporal_type(cls, v: str) -> str:
        if v not in ["bimamba2", "gated_deltanet"]:
            raise ValueError(f"temporal_type must be 'bimamba2' or 'gated_deltanet', got {v}")
        return v

    @field_validator("fusion_mode")
    @classmethod
    def validate_fusion_mode(cls, v: str) -> str:
        if v not in ["sum", "concat"]:
            raise ValueError(f"fusion_mode must be 'sum' or 'concat', got {v}")
        return v
```

---

## 5. Implementation: Config File Update

### 5.1. File: `configs/local/edge_gdn_test.yaml`

**Create new config for Phase 1a testing**:

```yaml
# Edge Stream GDN Migration Test Config
# Phase 1a: Replace ONLY edge stream, keep node stream as BiMamba2

experiment:
  name: edge_gdn_migration_test
  description: "Phase 1a - Edge stream BiGatedDeltaNet test"
  seed: 42
  output_dir: results/edge_gdn_test
  cache_dir: cache/tusz_mmap
  device: cuda
  log_level: INFO
  save_model: true
  save_best_only: true

  wandb:
    enabled: true
    project: seizure-v3-edge-gdn
    entity: null

model:
  architecture: v3

  # PR-1: Boundary Normalization (ENABLED)
  norms:
    boundary_norm: layernorm
    boundary_eps: 1.0e-5
    layerscale_alpha: 0.1
    after_tcn_proj: true
    after_node_mamba: true
    after_edge_mamba: true
    after_gnn: true
    before_decoder: true

  tcn:
    num_layers: 8
    kernel_size: 7
    dropout: 0.15
    causal: false
    stride_down: 16
    use_cuda_optimizations: true

  mamba:
    n_layers: 6
    d_model: 512
    d_state: 16
    conv_kernel: 4
    dropout: 0.1

    # PHASE 1a: Enable GDN for edge stream ONLY
    temporal_type: gated_deltanet  # NEW: Applies to edge stream
    fusion_mode: sum               # NEW: Start with simpler sum fusion
    allow_neg_eigval: false        # NEW: Conservative start

  # Graph configuration (V3)
  graph:
    enabled: true

    # PR-2: Bounded Edge Stream (ENABLED)
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1

    # V3: Edge stream config
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2       # Edge GDN layers
    edge_mamba_d_state: 8      # Ignored for GDN (uses internal A_log)
    edge_mamba_d_model: 16     # Edge GDN model dim
    edge_similarity_margin: 0.01

    # PR-3: Adjacency Conditioning (ENABLED)
    adj_row_softmax: true
    adj_softmax_tau: 1.0
    adj_ema_beta: 0.95
    adj_force_symmetric: true
    laplacian_eps: 1.0e-3
    laplacian_normalize: true

    # GNN architecture
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16

    # Dynamic PE config (v3) - OPTIMIZED FOR RTX 4090
    use_dynamic_pe: true
    semi_dynamic_interval: 5
    pe_sign_consistency: true

data:
  dataset: tuh_eeg
  data_dir: data_ext4/tusz/edf
  cache_dir: cache/tusz_mmap
  sampling_rate: 256
  n_channels: 19
  window_size: 60
  stride: 10
  use_balanced_sampling: true
  num_workers: 0
  pin_memory: true
  persistent_workers: false
  prefetch_factor: 2

preprocessing:
  montage: "10-20"
  bandpass: [0.5, 120.0]
  notch_freq: 60
  normalize: true

training:
  epochs: 10  # Short test run

  batch_size: 8

  learning_rate: 1.0e-4
  weight_decay: 0.01
  optimizer: adamw

  gradient_clip: 0.5

  mixed_precision: false

  loss: focal
  focal_alpha: 0.5
  focal_gamma: 2.0

  scheduler:
    type: cosine
    warmup_ratio: 0.03

  warmup_schedule:
    enabled: true
    warmup_steps: 1000
    adj_temperature_enabled: true
    adj_temperature_start: 2.0
    adj_temperature_end: 1.0
    focal_gamma_enabled: true
    focal_gamma_start: 1.0
    focal_gamma_end: 2.0

  early_stopping:
    patience: 5
    metric: sensitivity_at_10fa

  checkpoint_interval: 1
  mid_checkpoint_interval_s: 1800
  mid_epoch_keep: 3
  gradient_accumulation_steps: 1

postprocessing:
  hysteresis:
    tau_on: 0.86
    tau_off: 0.78
  morphology:
    opening_kernel: 11
    closing_kernel: 31
  duration:
    min_duration_s: 3.0
    max_duration_s: 600.0
  events:
    tau_merge: 2.0
    confidence_method: mean

evaluation:
  fa_rates: [10, 5, 2.5, 1]
  save_predictions: false
  save_plots: false

logging:
  log_every_n_steps: 50
  log_gradients: false
  log_weights: false
```

---

## 6. Testing Strategy

### 6.1. Unit Tests: `tests/unit/models/test_gated_deltanet.py`

```python
"""Unit tests for BiGatedDeltaNet wrapper."""

import pytest
import torch
import torch.nn as nn

from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet

# Skip if FLA not installed
pytest.importorskip("fla")


class TestBiGatedDeltaNet:
    """Test BiGatedDeltaNet wrapper."""

    def test_initialization_edge_config(self):
        """Test initialization with edge stream config."""
        model = BiGatedDeltaNet(
            d_model=16,
            headdim=4,
            num_layers=2,
            dropout=0.1,
            fusion_mode='sum',
        )
        assert model.d_model == 16
        assert model.headdim == 4
        assert model.num_layers == 2
        assert model.fusion_mode == 'sum'

    def test_initialization_node_config(self):
        """Test initialization with node stream config."""
        model = BiGatedDeltaNet(
            d_model=64,
            headdim=8,
            num_layers=6,
            dropout=0.1,
            fusion_mode='sum',
        )
        assert model.d_model == 64
        assert model.headdim == 8
        assert model.num_layers == 6

    def test_constraint_validation_valid(self):
        """Test valid headdim values satisfy 0.75× constraint."""
        # Edge: 3 × 4 = 12 = 0.75 × 16 ✅
        model_edge = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2)
        assert model_edge is not None

        # Node: 6 × 8 = 48 = 0.75 × 64 ✅
        model_node = BiGatedDeltaNet(d_model=64, headdim=8, num_layers=6)
        assert model_node is not None

    def test_constraint_validation_invalid(self):
        """Test invalid headdim values raise AssertionError."""
        # Invalid: 0.75 × 16 = 12, but headdim=5 doesn't divide 12
        with pytest.raises(AssertionError, match="num_heads × head_dim must equal"):
            BiGatedDeltaNet(d_model=16, headdim=5, num_layers=2)

    def test_forward_shape_edge_stream(self):
        """Test forward pass with edge stream dimensions."""
        model = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2, dropout=0.0)
        batch_size = 8
        num_edges = 171
        seq_len = 960

        x = torch.randn(batch_size * num_edges, 16, seq_len)
        y = model(x)

        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_forward_shape_node_stream(self):
        """Test forward pass with node stream dimensions."""
        model = BiGatedDeltaNet(d_model=64, headdim=8, num_layers=6, dropout=0.0)
        batch_size = 8
        num_nodes = 19
        seq_len = 960

        x = torch.randn(batch_size * num_nodes, 64, seq_len)
        y = model(x)

        assert y.shape == x.shape, f"Expected {x.shape}, got {y.shape}"

    def test_fusion_mode_sum(self):
        """Test sum fusion mode."""
        model = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2, fusion_mode='sum')
        assert model.fusion_mode == 'sum'
        assert model.fusion_proj is None

    def test_fusion_mode_concat(self):
        """Test concat fusion mode."""
        model = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2, fusion_mode='concat')
        assert model.fusion_mode == 'concat'
        assert model.fusion_proj is not None
        assert isinstance(model.fusion_proj, nn.Linear)

    def test_gradient_flow(self):
        """Test gradients flow through BiGatedDeltaNet."""
        model = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2, dropout=0.0)
        x = torch.randn(8, 16, 960, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None, "Gradients not flowing to input"
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradients not flowing to {name}"

    def test_no_nans_in_forward(self):
        """Test forward pass produces no NaNs."""
        model = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2, dropout=0.0)
        x = torch.randn(8, 16, 960)
        y = model(x)

        assert not torch.isnan(y).any(), "NaNs in output"
        assert not torch.isinf(y).any(), "Infs in output"

    def test_parameter_count_edge(self):
        """Test parameter count is reasonable for edge stream.

        NOTE: GDN has ~29% fewer params than BiMamba2 due to 0.75× q/k projection.
        Expected: ~7.3K params (vs BiMamba2's 10.3K).
        """
        model = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2, dropout=0.0)
        param_count = sum(p.numel() for p in model.parameters())

        # GDN has 0.75× q/k projection → fewer params than BiMamba2
        # Expected range: 6K-9K (with 7.3K being typical)
        assert 6_000 < param_count < 9_000, (
            f"Parameter count outside expected range: {param_count:,} "
            f"(expected ~7.3K due to GDN's 0.75× q/k projection)"
        )

    def test_parameter_count_node(self):
        """Test parameter count is reasonable for node stream."""
        model = BiGatedDeltaNet(d_model=64, headdim=8, num_layers=6, dropout=0.0)
        param_count = sum(p.numel() for p in model.parameters())

        # Should be similar to BiMamba2 (~398K params)
        assert 300_000 < param_count < 600_000, f"Unexpected param count: {param_count:,}"

    def test_cuda_compatibility(self):
        """Test CUDA compatibility (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2).cuda()
        x = torch.randn(8, 16, 960).cuda()
        y = model(x)

        assert y.is_cuda
        assert y.shape == x.shape

    def test_repr(self):
        """Test string representation."""
        model = BiGatedDeltaNet(d_model=16, headdim=4, num_layers=2, fusion_mode='sum')
        repr_str = repr(model)

        assert "BiGatedDeltaNet" in repr_str
        assert "d_model=16" in repr_str
        assert "headdim=4" in repr_str
        assert "num_layers=2" in repr_str
        assert "fusion_mode=sum" in repr_str
```

**Run unit tests**:

```bash
pytest tests/unit/models/test_gated_deltanet.py -xvs
```

### 6.2. Integration Test: `tests/integration/test_edge_gdn_migration.py`

```python
"""Integration test for edge stream GDN migration."""

import pytest
import torch

from src.brain_brr.models.builders.edge_stream import build_edge_stream
from src.brain_brr.config.schemas import ModelConfig, MambaConfig, GraphConfig

# Skip if FLA not installed
pytest.importorskip("fla")


class TestEdgeStreamGDNMigration:
    """Test edge stream migration from BiMamba2 to BiGatedDeltaNet."""

    @pytest.fixture
    def base_config(self):
        """Base config with edge stream settings."""
        return ModelConfig(
            mamba=MambaConfig(
                n_layers=6,
                d_model=512,
                d_state=16,
                conv_kernel=4,
                dropout=0.1,
                temporal_type="bimamba2",  # Start with baseline
            ),
            graph=GraphConfig(
                enabled=True,
                edge_mamba_layers=2,
                edge_mamba_d_state=8,
                edge_mamba_d_model=16,
            ),
        )

    def test_build_edge_stream_bimamba2(self, base_config):
        """Test building edge stream with BiMamba2 (baseline)."""
        from src.brain_brr.models.mamba import BiMamba2

        components = build_edge_stream(base_config)

        assert isinstance(components.edge_mamba, BiMamba2)
        assert components.edge_mamba.d_model == 16
        assert components.edge_mamba.headdim == 4
        assert components.edge_mamba.num_layers == 2

    def test_build_edge_stream_gdn(self, base_config):
        """Test building edge stream with BiGatedDeltaNet."""
        from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet

        # Enable GDN
        base_config.mamba.temporal_type = "gated_deltanet"
        base_config.mamba.fusion_mode = "sum"
        base_config.mamba.allow_neg_eigval = False

        components = build_edge_stream(base_config)

        assert isinstance(components.edge_mamba, BiGatedDeltaNet)
        assert components.edge_mamba.d_model == 16
        assert components.edge_mamba.headdim == 4
        assert components.edge_mamba.num_layers == 2
        assert components.edge_mamba.fusion_mode == "sum"

    def test_forward_pass_compatibility(self, base_config):
        """Test forward pass with both BiMamba2 and BiGatedDeltaNet."""
        batch_size = 2
        num_edges = 171
        seq_len = 960

        # BiMamba2 baseline
        base_config.mamba.temporal_type = "bimamba2"
        components_bm = build_edge_stream(base_config)

        # BiGatedDeltaNet
        base_config.mamba.temporal_type = "gated_deltanet"
        components_gdn = build_edge_stream(base_config)

        # Test input (after edge_in_proj)
        x = torch.randn(batch_size * num_edges, 16, seq_len)

        # Forward pass
        y_bm = components_bm.edge_mamba(x)
        y_gdn = components_gdn.edge_mamba(x)

        # Check shapes match
        assert y_bm.shape == y_gdn.shape == x.shape

        # Check no NaNs
        assert not torch.isnan(y_bm).any()
        assert not torch.isnan(y_gdn).any()

    def test_parameter_count_similar(self, base_config):
        """Test parameter counts are in same order of magnitude.

        NOTE: GDN has ~29% fewer params than BiMamba2 due to 0.75× q/k projection.
        This is EXPECTED and not a regression - it's part of GDN's design.
        """
        # BiMamba2 baseline
        base_config.mamba.temporal_type = "bimamba2"
        components_bm = build_edge_stream(base_config)
        params_bm = sum(p.numel() for p in components_bm.edge_mamba.parameters())

        # BiGatedDeltaNet
        base_config.mamba.temporal_type = "gated_deltanet"
        components_gdn = build_edge_stream(base_config)
        params_gdn = sum(p.numel() for p in components_gdn.edge_mamba.parameters())

        print(f"BiMamba2 params: {params_bm:,}")
        print(f"BiGatedDeltaNet params: {params_gdn:,}")
        print(f"Reduction: {(1 - params_gdn/params_bm)*100:.1f}% (expected ~29%)")

        # GDN should have 65-85% of BiMamba2 params (due to 0.75× q/k projection)
        # BiMamba2: ~10.3K, GDN: ~7.3K → 71% of original
        assert 0.65 * params_bm < params_gdn < 0.85 * params_bm, (
            f"Parameter count outside expected range: "
            f"BiMamba2={params_bm:,}, GDN={params_gdn:,} "
            f"(expected GDN to be 65-85% of BiMamba2 due to 0.75× q/k projection)"
        )

    def test_edge_stream_in_detector_context(self, base_config):
        """Test edge stream works in detector-like context."""
        from src.brain_brr.models.edge_features import edge_scalar_series

        batch_size = 2
        num_electrodes = 19
        seq_len = 960
        d_features = 64

        # Enable GDN
        base_config.mamba.temporal_type = "gated_deltanet"
        components = build_edge_stream(base_config)

        # Simulate detector flow
        elec_feats = torch.randn(batch_size, num_electrodes, seq_len, d_features)

        # Compute edge features (cosine similarity)
        edge_feats = edge_scalar_series(elec_feats, metric="cosine", edge_similarity_margin=0.01)
        assert edge_feats.shape == (batch_size, 171, seq_len, 1)

        # Flatten for edge Mamba
        edge_flat = edge_feats.squeeze(-1).reshape(batch_size * 171, 1, seq_len)

        # Lift 1→16
        edge_in = components.edge_in_proj(edge_flat)
        assert edge_in.shape == (batch_size * 171, 16, seq_len)

        # Process through GDN
        edge_processed = components.edge_mamba(edge_in)
        assert edge_processed.shape == (batch_size * 171, 16, seq_len)

        # Project 16→1
        edge_out = components.edge_out_proj(edge_processed)
        assert edge_out.shape == (batch_size * 171, 1, seq_len)

        # Unflatten
        edge_weights = edge_out.squeeze(1).reshape(batch_size, 171, seq_len)
        assert edge_weights.shape == (batch_size, 171, seq_len)

        # No NaNs
        assert not torch.isnan(edge_weights).any()

    def test_gradient_flow_through_edge_stream(self, base_config):
        """Test gradients flow through edge stream."""
        base_config.mamba.temporal_type = "gated_deltanet"
        components = build_edge_stream(base_config)

        x = torch.randn(8 * 171, 16, 960, requires_grad=True)
        y = components.edge_mamba(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in components.edge_mamba.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
```

**Run integration tests**:

```bash
pytest tests/integration/test_edge_gdn_migration.py -xvs
```

---

## 7. Validation & Benchmarking

### 7.1. Smoke Test (3 files, 1 epoch)

**Purpose**: Verify no crashes, shapes correct, no NaNs

```bash
# Set environment
export BGB_SMOKE_TEST=1
export BGB_NAN_DEBUG=1

# Run smoke test
python -m src train configs/local/edge_gdn_test.yaml

# Expected output:
# - Loads 3 files
# - 1 epoch completes
# - No crashes, no NaNs
# - Checkpoint saved
```

### 7.2. Integration Test (50 files, 10 epochs)

**Purpose**: Validate convergence, compare against baseline

```bash
# Run integration test
export BGB_LIMIT_FILES=50
python -m src train configs/local/edge_gdn_test.yaml

# Monitor:
# - Loss curve (should decrease)
# - Gradient norms (should be stable)
# - Memory usage (should be ~20GB on RTX 4090)
# - Throughput (may be 5-10% slower than baseline)
```

### 7.3. A/B Comparison

**Baseline (BiMamba2 edge)**:
```bash
# Run baseline first
cp configs/local/train.yaml configs/local/baseline_edge.yaml
# Edit: set training.epochs: 10
export BGB_LIMIT_FILES=50
python -m src train configs/local/baseline_edge.yaml --experiment.name baseline_edge
```

**Phase 1a (GDN edge)**:
```bash
# Run Phase 1a
python -m src train configs/local/edge_gdn_test.yaml --experiment.name phase1a_gdn_edge
```

**Compare**:
```python
import wandb

api = wandb.Api()
runs = api.runs("seizure-v3-edge-gdn")

baseline = [r for r in runs if r.name == "baseline_edge"][0]
phase1a = [r for r in runs if r.name == "phase1a_gdn_edge"][0]

# Compare metrics
print(f"Baseline loss: {baseline.summary['val_loss']:.4f}")
print(f"Phase 1a loss: {phase1a.summary['val_loss']:.4f}")
print(f"Baseline sens@10FA: {baseline.summary['sensitivity_at_10fa']:.2%}")
print(f"Phase 1a sens@10FA: {phase1a.summary['sensitivity_at_10fa']:.2%}")

# Expected: Phase 1a slightly better (+1-2%)
```

---

## 8. Success Criteria

### 8.1. Technical Criteria

✅ **Unit tests pass**: All tests in `test_gated_deltanet.py` pass
✅ **Integration tests pass**: All tests in `test_edge_gdn_migration.py` pass
✅ **Smoke test completes**: 3 files, 1 epoch, no crashes
✅ **No NaNs**: Forward/backward passes produce finite values
✅ **Shapes correct**: Input/output shapes match BiMamba2
✅ **Parameter count**: ~7.3K params (29% reduction from BiMamba2's 10.3K - this is EXPECTED and GOOD)

### 8.2. Performance Criteria

✅ **Convergence**: Loss decreases over 10 epochs
✅ **No regression**: val_loss ≤ baseline + 0.05
✅ **Hypothesis validated**: sensitivity_at_10fa ≥ baseline + 0.01 (+1%)
✅ **Throughput acceptable**: ≤ 10% slower than baseline
✅ **Memory usage**: ≤ baseline + 2GB

### 8.3. Go/No-Go Decision

**GO → Phase 1b (Node Stream)** if:
- All technical criteria met
- Performance improvement ≥ +1% sensitivity
- No major regressions

**NO-GO → Revert** if:
- Performance regression > 2% sensitivity
- Training unstable (NaNs, divergence)
- Throughput regression > 20%

---

## 9. Rollback Plan

If Phase 1a fails, revert with:

```bash
# Revert code changes
git checkout v3.8.3-pre-edge-migration

# Restore configs
git checkout HEAD~1 configs/local/edge_gdn_test.yaml

# Clean up
rm src/brain_brr/models/gated_deltanet.py
git restore src/brain_brr/models/builders/edge_stream.py
git restore src/brain_brr/config/schemas.py

# Verify baseline restored
pytest tests/unit/models/ -xvs
make smoke-test
```

---

## 10. Timeline & Checklist

### Day 1: Implementation
- [ ] Install FLA: `pip install flash-linear-attention`
- [ ] Create `gated_deltanet.py` wrapper
- [ ] Update `edge_stream.py` builder
- [ ] Update `schemas.py` config
- [ ] Create `edge_gdn_test.yaml` config
- [ ] Write unit tests
- [ ] Write integration tests
- [ ] All tests pass locally

### Day 2: Validation
- [ ] Smoke test (3 files, 1 epoch) - 10 min
- [ ] Integration test (50 files, 10 epochs) - 6-8 hours
- [ ] Baseline A/B comparison
- [ ] Analyze results

### Day 3: Decision
- [ ] Review metrics (loss, sensitivity, throughput)
- [ ] Go/No-Go decision
- [ ] Document findings
- [ ] If GO: Proceed to Doc 2 (Node Stream)
- [ ] If NO-GO: Execute rollback plan

**Total**: 2-3 days

---

## 11. Next Steps

If Phase 1a succeeds:

1. **Document results** in `PHASE1A_RESULTS.md`
2. **Create Doc 2**: Node Stream Migration Implementation Plan
3. **Schedule Phase 1b**: Node stream replacement
4. **Continue phased approach**: Doc 3 (Full Migration), Doc 4 (Hybrid/SWA)

If Phase 1a fails:

1. **Document failure mode** in `PHASE1A_POSTMORTEM.md`
2. **Root cause analysis**: Why did it fail?
3. **Iterate or pivot**: Fix issues or consider alternatives (GLA, HGRN2)

---

## 12. References

- **Doc 0 (SSOT)**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md)
- **FLA Library**: https://github.com/fla-org/flash-linear-attention
- **Gated DeltaNet Paper**: https://arxiv.org/abs/2412.06464
- **Current v3.8.3 Baseline**: RELEASE_NOTES.md

---

**Document Status**: ✅ Ready for Implementation
**Next Document**: Doc 2 (Node Stream Migration) - pending Phase 1a success
