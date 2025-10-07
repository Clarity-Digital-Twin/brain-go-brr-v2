# Doc 2: Node Stream Migration - Implementation Plan

**Parent Document**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md) (Doc 0 - SSOT)
**Phase**: 1b (Validation Phase)
**Target**: Replace Node Stream BiMamba2 → BiGatedDeltaNet
**Date**: October 7, 2025
**Version**: 1.1 (Config workflow + W&B analysis fixes)
**Status**: Ready for Implementation (pending Phase 1a success)

**Changelog**:
- v1.1 (Oct 7, 2025): Fixed CLI commands to use proper config workflow (no --experiment.name hacks)
- v1.1 (Oct 7, 2025): Fixed W&B analysis to query by config.experiment.name (robust to name suffixes)
- v1.0 (Oct 7, 2025): Initial version

---

## Executive Summary

This document provides **surgical implementation details** for migrating the node stream (and ONLY the node stream) from BiMamba2 to BiGatedDeltaNet. This is Phase 1b of the phased migration strategy.

**Scope of Changes**:
- ✅ Update `src/brain_brr/models/builders/node_stream.py` (return BiGatedDeltaNet)
- ✅ Reuse `src/brain_brr/models/gated_deltanet.py` from Phase 1a (already exists)
- ✅ Add node-specific config in `configs/local/node_gdn_test.yaml`
- ✅ Write integration test (`tests/integration/test_node_gdn_migration.py`)
- ❌ **DO NOT TOUCH**: Edge stream (keep BiMamba2), GNN, TCN, decoder

**Expected Outcome**:
- Node stream: **398K params BiMamba2 → ~284K params BiGatedDeltaNet** (29% reduction due to 0.75× q/k projection)
- Hypothesis: +5-10% better per-electrode memory → +1-2% sensitivity @ 1 FA/24h
- Risk: **MEDIUM** (larger parameter count than edge, 284K out of 405K total = 70% of stream parameters)

**Timeline**: 1-2 days development + 6-8 hours integration test + 1 day analysis

---

## 📊 Parameter Count Analysis

**IMPORTANT**: GDN's 0.75× q/k projection (vs Mamba2's 1.0×) reduces parameter count by ~29%. **This is EXPECTED and BENEFICIAL**:

| Component | BiMamba2 (Baseline) | BiGatedDeltaNet (Phase 1b) | Reduction |
|-----------|---------------------|----------------------------|-----------|
| **Node Stream** | 397,632 params | **~284,000 params** | **-29%** |
| **Edge Stream** | 10,304 params | (unchanged - Phase 1a) | N/A |
| **Total Streams** | 407,936 params | **~294,304 params** | **-28%** |

**Why fewer parameters is GOOD**:
- ✅ **More parameter-efficient**: Same representational capacity with fewer params
- ✅ **Faster inference**: Fewer parameters = faster forward pass
- ✅ **Better generalization**: Reduced parameter count can improve generalization
- ✅ **By design**: GDN paper shows 0.75× allocation is intentional and performs well

**Key Insight**: The node stream has **39× more parameters** than edge stream (398K vs 10K), so this is a more significant test of GDN's benefits. However, the 29% reduction is still expected and part of GDN's design.

---

## 1. Prerequisites

### 1.1. Environment Setup

```bash
# Verify Phase 1a completed successfully
git log --oneline | head -5  # Should show Phase 1a commits

# Verify BiGatedDeltaNet wrapper exists
python -c "from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet; print('✅ Wrapper exists')"

# Create Phase 1b branch
git checkout -b feature/node-gdn-migration
git tag v3.8.3-pre-node-migration
```

### 1.2. Verify Current Node Stream

```bash
# Run baseline to establish metrics
python -c "
from src.brain_brr.models.builders.node_stream import build_node_stream
from src.brain_brr.config.schemas import ModelConfig

cfg = ModelConfig()
node_mamba = build_node_stream(cfg)
print(f'Node Mamba params: {sum(p.numel() for p in node_mamba.parameters()):,}')
print(f'd_model: {node_mamba.d_model}')
print(f'headdim: {node_mamba.headdim}')
print(f'num_layers: {node_mamba.num_layers}')
"
# Expected output (BiMamba2 baseline):
# Node Mamba params: 397,632
# d_model: 64
# headdim: 8
# num_layers: 6

# Expected output (BiGatedDeltaNet - after migration):
# Node GDN params: ~284,000 (29% reduction due to GDN's 0.75× q/k projection)
```

---

## 2. Implementation: Node Stream Builder Update

### 2.1. File: `src/brain_brr/models/builders/node_stream.py`

**Changes Required**:

1. Import BiGatedDeltaNet (from Phase 1a)
2. Conditionally build BiGatedDeltaNet or BiMamba2 based on config
3. Keep same interface (return BiMamba2 | BiGatedDeltaNet)

**Implementation**:

```python
"""Node stream builder - per-electrode BiMamba/BiGatedDeltaNet component."""

import logging
from typing import TYPE_CHECKING

from src.brain_brr.constants import LAYERSCALE_ALPHA_FALLBACK

from ..mamba import BiMamba2
from ..norms import LayerScale

# Import GDN conditionally (from Phase 1a)
try:
    from ..gated_deltanet import BiGatedDeltaNet
    GDN_AVAILABLE = True
except ImportError:
    GDN_AVAILABLE = False

if TYPE_CHECKING:
    from src.brain_brr.config.schemas import ModelConfig

logger = logging.getLogger(__name__)


def build_node_stream(cfg: "ModelConfig") -> BiMamba2 | "BiGatedDeltaNet":
    """Build node stream: per-electrode BiMamba or BiGatedDeltaNet.

    V3 Architecture: Processes per-electrode temporal features with BiMamba or BiGatedDeltaNet.
    This is a SHARED module that processes flattened (B*19, d_model, T) tensors.

    Args:
        cfg: Model configuration containing mamba and norms settings

    Returns:
        BiMamba2 or BiGatedDeltaNet module

    Notes:
        - d_model=64 (per-electrode feature dimension)
        - headdim=8 for BiMamba2 (8 heads × 8 dim = 64)
        - For GDN: headdim=8, num_heads=6 (6 × 8 = 48 = 0.75 × 64)
        - num_layers=6 (deeper than edge stream)
        - LayerScale enabled if boundary_norm != "none"
    """
    norms_cfg = getattr(cfg, "norms", None)
    mamba_cfg = cfg.mamba

    use_layerscale = bool(norms_cfg and norms_cfg.boundary_norm != "none")
    layerscale_init = float(norms_cfg.layerscale_alpha if norms_cfg else LAYERSCALE_ALPHA_FALLBACK)

    # Determine which temporal model to use (STREAM-SPECIFIC for Phase 1b isolation)
    # Priority: temporal_type_node > temporal_type (fallback)
    temporal_type = getattr(mamba_cfg, "temporal_type_node", None)
    if temporal_type is None:
        temporal_type = getattr(mamba_cfg, "temporal_type", "bimamba2")

    logger.debug(f"Node stream temporal_type: {temporal_type} (stream-specific or fallback)")

    if temporal_type == "gated_deltanet":
        if not GDN_AVAILABLE:
            raise ImportError(
                "Gated DeltaNet requested but not available. "
                "Ensure Phase 1a completed: pip install flash-linear-attention"
            )

        # Build BiGatedDeltaNet for node stream
        node_mamba = BiGatedDeltaNet(
            d_model=64,
            headdim=8,  # 6 × 8 = 48 = 0.75 × 64 ✅
            num_layers=6,
            dropout=mamba_cfg.dropout,
            fusion_mode=getattr(mamba_cfg, "fusion_mode", "sum"),
            allow_neg_eigval=getattr(mamba_cfg, "allow_neg_eigval", False),
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
        )
        logger.info(
            f"Node stream: BiGatedDeltaNet (d_model=64, headdim=8, layers=6, "
            f"fusion_mode={getattr(mamba_cfg, 'fusion_mode', 'sum')})"
        )
    else:
        # Build BiMamba2 for node stream (default/baseline)
        node_mamba = BiMamba2(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
            headdim=8,
            num_layers=6,
            dropout=mamba_cfg.dropout,
            use_layerscale=use_layerscale,
            layerscale_init=layerscale_init,
        )
        logger.info("Node stream: BiMamba2 (d_model=64, headdim=8, layers=6)")

    return node_mamba
```

---

## 3. Implementation: Config File Update

### 3.1. File: `configs/local/node_gdn_test.yaml`

**Create new config for Phase 1b testing**:

```yaml
# Node Stream GDN Migration Test Config
# Phase 1b: Replace ONLY node stream, keep edge stream as BiMamba2

experiment:
  name: node_gdn_migration_test
  description: "Phase 1b - Node stream BiGatedDeltaNet test"
  seed: 42
  output_dir: results/node_gdn_test
  cache_dir: cache/tusz_mmap
  device: cuda
  log_level: INFO
  save_model: true
  save_best_only: true

  wandb:
    enabled: true
    project: seizure-v3-node-gdn
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

    # PHASE 1b: Enable GDN for node stream ONLY (stream-specific control)
    temporal_type: bimamba2              # Fallback for edge stream
    temporal_type_node: gated_deltanet   # Override: node uses GDN
    temporal_type_edge: null             # Edge uses fallback (BiMamba2)
    fusion_mode: sum                     # Start with simpler sum fusion
    allow_neg_eigval: false              # Conservative start

  # Graph configuration (V3)
  graph:
    enabled: true

    # PR-2: Bounded Edge Stream (ENABLED)
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1

    # V3: Edge stream config (KEEP BiMamba2 for Phase 1b)
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2       # Edge BiMamba2 layers
    edge_mamba_d_state: 8      # Edge BiMamba2 d_state
    edge_mamba_d_model: 16     # Edge BiMamba2 model dim
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

## 4. Testing Strategy

### 4.1. Integration Test: `tests/integration/test_node_gdn_migration.py`

```python
"""Integration test for node stream GDN migration."""

import pytest
import torch

from src.brain_brr.models.builders.node_stream import build_node_stream
from src.brain_brr.config.schemas import ModelConfig, MambaConfig

# Skip if FLA not installed
pytest.importorskip("fla")


class TestNodeStreamGDNMigration:
    """Test node stream migration from BiMamba2 to BiGatedDeltaNet."""

    @pytest.fixture
    def base_config(self):
        """Base config with node stream settings."""
        return ModelConfig(
            mamba=MambaConfig(
                n_layers=6,
                d_model=512,
                d_state=16,
                conv_kernel=4,
                dropout=0.1,
                temporal_type="bimamba2",  # Start with baseline
            ),
        )

    def test_build_node_stream_bimamba2(self, base_config):
        """Test building node stream with BiMamba2 (baseline)."""
        from src.brain_brr.models.mamba import BiMamba2

        node_mamba = build_node_stream(base_config)

        assert isinstance(node_mamba, BiMamba2)
        assert node_mamba.d_model == 64
        assert node_mamba.headdim == 8
        assert node_mamba.num_layers == 6

    def test_build_node_stream_gdn(self, base_config):
        """Test building node stream with BiGatedDeltaNet."""
        from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet

        # Enable GDN
        base_config.mamba.temporal_type = "gated_deltanet"
        base_config.mamba.fusion_mode = "sum"
        base_config.mamba.allow_neg_eigval = False

        node_mamba = build_node_stream(base_config)

        assert isinstance(node_mamba, BiGatedDeltaNet)
        assert node_mamba.d_model == 64
        assert node_mamba.headdim == 8
        assert node_mamba.num_layers == 6
        assert node_mamba.fusion_mode == "sum"

    def test_forward_pass_compatibility(self, base_config):
        """Test forward pass with both BiMamba2 and BiGatedDeltaNet."""
        batch_size = 2
        num_electrodes = 19
        seq_len = 960

        # BiMamba2 baseline
        base_config.mamba.temporal_type = "bimamba2"
        node_mamba_bm = build_node_stream(base_config)

        # BiGatedDeltaNet
        base_config.mamba.temporal_type = "gated_deltanet"
        node_mamba_gdn = build_node_stream(base_config)

        # Test input (after flattening in detector)
        x = torch.randn(batch_size * num_electrodes, 64, seq_len)

        # Forward pass
        y_bm = node_mamba_bm(x)
        y_gdn = node_mamba_gdn(x)

        # Check shapes match
        assert y_bm.shape == y_gdn.shape == x.shape

        # Check no NaNs
        assert not torch.isnan(y_bm).any()
        assert not torch.isnan(y_gdn).any()

    def test_parameter_count_comparison(self, base_config):
        """Test parameter counts - GDN should have ~29% fewer params.

        NOTE: GDN has ~29% fewer params than BiMamba2 due to 0.75× q/k projection.
        This is EXPECTED and part of GDN's parameter efficiency design.
        """
        # BiMamba2 baseline
        base_config.mamba.temporal_type = "bimamba2"
        node_mamba_bm = build_node_stream(base_config)
        params_bm = sum(p.numel() for p in node_mamba_bm.parameters())

        # BiGatedDeltaNet
        base_config.mamba.temporal_type = "gated_deltanet"
        node_mamba_gdn = build_node_stream(base_config)
        params_gdn = sum(p.numel() for p in node_mamba_gdn.parameters())

        print(f"BiMamba2 params: {params_bm:,}")
        print(f"BiGatedDeltaNet params: {params_gdn:,}")
        print(f"Reduction: {(1 - params_gdn/params_bm)*100:.1f}% (expected ~29%)")

        # GDN should have 65-85% of BiMamba2 params (due to 0.75× q/k projection)
        # BiMamba2: ~398K, GDN: ~284K → 71% of original
        assert 0.65 * params_bm < params_gdn < 0.85 * params_bm, (
            f"Parameter count outside expected range: "
            f"BiMamba2={params_bm:,}, GDN={params_gdn:,} "
            f"(expected GDN to be 65-85% of BiMamba2 due to 0.75× q/k projection)"
        )

    def test_node_stream_in_detector_context(self, base_config):
        """Test node stream works in detector-like context."""
        batch_size = 2
        num_electrodes = 19
        seq_len = 960
        d_features = 64

        # Enable GDN
        base_config.mamba.temporal_type = "gated_deltanet"
        node_mamba = build_node_stream(base_config)

        # Simulate detector flow (after TCN projection to electrodes)
        elec_feats = torch.randn(batch_size, num_electrodes, seq_len, d_features)

        # Flatten for node Mamba (B, 19, L, 64) → (B*19, 64, L)
        node_flat = elec_feats.permute(0, 1, 3, 2).reshape(
            batch_size * num_electrodes, d_features, seq_len
        )
        assert node_flat.shape == (batch_size * num_electrodes, 64, seq_len)

        # Process through GDN
        node_processed = node_mamba(node_flat)
        assert node_processed.shape == (batch_size * num_electrodes, 64, seq_len)

        # Unflatten (B*19, 64, L) → (B, 19, 64, L)
        node_feats = node_processed.reshape(batch_size, num_electrodes, d_features, seq_len)
        assert node_feats.shape == (batch_size, num_electrodes, 64, seq_len)

        # No NaNs
        assert not torch.isnan(node_feats).any()

    def test_gradient_flow_through_node_stream(self, base_config):
        """Test gradients flow through node stream."""
        base_config.mamba.temporal_type = "gated_deltanet"
        node_mamba = build_node_stream(base_config)

        x = torch.randn(8 * 19, 64, 960, requires_grad=True)
        y = node_mamba(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        for name, param in node_mamba.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_cuda_compatibility(self, base_config):
        """Test CUDA compatibility (if available)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        base_config.mamba.temporal_type = "gated_deltanet"
        node_mamba = build_node_stream(base_config).cuda()

        x = torch.randn(8 * 19, 64, 960).cuda()
        y = node_mamba(x)

        assert y.is_cuda
        assert y.shape == x.shape
        assert not torch.isnan(y).any()

    def test_parameter_count_absolute(self, base_config):
        """Test absolute parameter count is in expected range."""
        base_config.mamba.temporal_type = "gated_deltanet"
        node_mamba = build_node_stream(base_config)
        params = sum(p.numel() for p in node_mamba.parameters())

        # Expected: ~284K params (29% reduction from 398K BiMamba2 baseline)
        assert 250_000 < params < 350_000, (
            f"Parameter count outside expected range: {params:,} "
            f"(expected ~284K due to GDN's 0.75× q/k projection)"
        )

    def test_stream_isolation_phase1b(self, base_config):
        """Test Phase 1b isolation: Node GDN, Edge BiMamba2.

        CRITICAL: Verify edge stream remains BiMamba2 when node uses GDN.
        This is the WHOLE POINT of stream-specific config fields.
        """
        from src.brain_brr.models.builders.edge_stream import build_edge_stream
        from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet
        from src.brain_brr.models.mamba import BiMamba2

        # Phase 1b config: node=GDN, edge=BiMamba2
        base_config.mamba.temporal_type = "bimamba2"           # Fallback
        base_config.mamba.temporal_type_node = "gated_deltanet"  # Override node
        base_config.mamba.temporal_type_edge = None            # Use fallback

        # Add GraphConfig for edge stream
        from src.brain_brr.config.schemas import GraphConfig
        base_config.graph = GraphConfig(
            enabled=True,
            edge_mamba_layers=2,
            edge_mamba_d_state=8,
            edge_mamba_d_model=16,
        )

        node_mamba = build_node_stream(base_config)
        edge_components = build_edge_stream(base_config)

        # Verify isolation
        assert isinstance(node_mamba, BiGatedDeltaNet), "Node should be GDN in Phase 1b"
        assert isinstance(edge_components.edge_mamba, BiMamba2), "Edge should be BiMamba2 in Phase 1b"
```

**Run integration tests**:

```bash
pytest tests/integration/test_node_gdn_migration.py -xvs
```

---

## 5. Validation & Benchmarking

### 5.1. Smoke Test (3 files, 1 epoch)

**Purpose**: Verify no crashes, shapes correct, no NaNs

```bash
# Set environment
export BGB_SMOKE_TEST=1
export BGB_NAN_DEBUG=1

# Run smoke test
python -m src train configs/local/node_gdn_test.yaml

# Expected output:
# - Loads 3 files
# - 1 epoch completes
# - No crashes, no NaNs
# - Checkpoint saved
```

### 5.2. Integration Test (50 files, 10 epochs)

**Purpose**: Validate convergence, compare against baseline

```bash
# Run integration test
export BGB_LIMIT_FILES=50
python -m src train configs/local/node_gdn_test.yaml

# Monitor:
# - Loss curve (should decrease)
# - Gradient norms (should be stable)
# - Memory usage (should be ~20GB on RTX 4090)
# - Throughput (may be 5-10% slower than baseline)
```

### 5.3. A/B Comparison

**CRITICAL**: CLI does NOT support `--experiment.name` overrides. Create separate configs for each experiment.

**Baseline (BiMamba2 node)**:
```bash
# Run baseline first
cp configs/local/train.yaml configs/local/baseline_node.yaml
# Edit configs/local/baseline_node.yaml:
#   experiment.name: "baseline_node"
#   training.epochs: 10
export BGB_LIMIT_FILES=50
python -m src train configs/local/baseline_node.yaml
```

**Phase 1b (GDN node)**:
```bash
# Run Phase 1b
cp configs/local/node_gdn_test.yaml configs/local/phase1b_gdn_node.yaml
# Edit configs/local/phase1b_gdn_node.yaml:
#   experiment.name: "phase1b_gdn_node"
#   training.epochs: 10
python -m src train configs/local/phase1b_gdn_node.yaml
```

**Compare** (using robust W&B analysis):
```python
"""Compare Phase 1b results against baseline.

USAGE:
    python scripts/analyze_phase1b_results.py --project seizure-v3-node-gdn
"""
import wandb


def find_run_by_experiment_name(runs: list, experiment_name: str):
    """Find run by experiment.name config field (robust to W&B name suffixes).

    Args:
        runs: List of W&B runs
        experiment_name: Expected value of config.experiment.name

    Returns:
        Matching run or None if not found
    """
    matches = [
        r for r in runs
        if r.config.get('experiment', {}).get('name') == experiment_name
    ]

    if not matches:
        print(f"ERROR: No runs found with experiment.name='{experiment_name}'")
        return None
    if len(matches) > 1:
        print(f"Warning: Found {len(matches)} runs with experiment.name='{experiment_name}', using first")
        print(f"  Run IDs: {[r.id for r in matches]}")

    return matches[0]


# Initialize W&B API
api = wandb.Api()
runs = api.runs("seizure-v3-node-gdn")

# Find runs by experiment.name (robust to W&B suffixes like "baseline_node-1")
baseline = find_run_by_experiment_name(runs, "baseline_node")
phase1b = find_run_by_experiment_name(runs, "phase1b_gdn_node")

if not baseline or not phase1b:
    print("\nAvailable runs:")
    for r in runs:
        exp_name = r.config.get('experiment', {}).get('name', 'UNKNOWN')
        print(f"  - {r.id}: {r.name} (experiment.name={exp_name})")
    exit(1)

# Compare metrics
print(f"Baseline loss: {baseline.summary.get('val_loss', 'N/A'):.4f}")
print(f"Phase 1b loss: {phase1b.summary.get('val_loss', 'N/A'):.4f}")
print(f"Baseline sens@10FA: {baseline.summary.get('sensitivity_at_10fa', 0):.2%}")
print(f"Phase 1b sens@10FA: {phase1b.summary.get('sensitivity_at_10fa', 0):.2%}")

# Expected: Phase 1b slightly better (+1-2%)
```

---

## 6. Success Criteria

### 6.1. Technical Criteria

✅ **Integration tests pass**: All tests in `test_node_gdn_migration.py` pass
✅ **Smoke test completes**: 3 files, 1 epoch, no crashes
✅ **No NaNs**: Forward/backward passes produce finite values
✅ **Shapes correct**: Input/output shapes match BiMamba2
✅ **Parameter count**: ~284K params (29% reduction from BiMamba2's 398K - this is EXPECTED and GOOD)

### 6.2. Performance Criteria

✅ **Convergence**: Loss decreases over 10 epochs
✅ **No regression**: val_loss ≤ baseline + 0.05
✅ **Hypothesis validated**: sensitivity_at_10fa ≥ baseline + 0.01 (+1%)
✅ **Throughput acceptable**: ≤ 10% slower than baseline
✅ **Memory usage**: ≤ baseline + 2GB

### 6.3. Go/No-Go Decision

**GO → Phase 2 (Both Streams)** if:
- All technical criteria met
- Performance improvement ≥ +1% sensitivity
- No major regressions
- Phase 1a also succeeded

**NO-GO → Revert** if:
- Performance regression > 2% sensitivity
- Training unstable (NaNs, divergence)
- Throughput regression > 20%

---

## 7. Rollback Plan

If Phase 1b fails, revert with:

```bash
# Revert code changes
git checkout v3.8.3-pre-node-migration

# Restore configs
git checkout HEAD~1 configs/local/node_gdn_test.yaml

# Clean up
git restore src/brain_brr/models/builders/node_stream.py

# Verify baseline restored
pytest tests/unit/models/ -xvs
make smoke-test
```

---

## 8. Timeline & Checklist

### Day 1: Implementation
- [ ] Verify Phase 1a completed successfully
- [ ] Update `node_stream.py` builder
- [ ] Create `node_gdn_test.yaml` config
- [ ] Write integration tests
- [ ] All tests pass locally

### Day 2: Validation
- [ ] Smoke test (3 files, 1 epoch) - 10 min
- [ ] Integration test (50 files, 10 epochs) - 6-8 hours
- [ ] Baseline A/B comparison
- [ ] Analyze results

### Day 3: Decision
- [ ] Review metrics (loss, sensitivity, throughput)
- [ ] Compare Phase 1a vs Phase 1b results
- [ ] Go/No-Go decision for Phase 2
- [ ] Document findings
- [ ] If GO: Proceed to Doc 3 (Full Migration - Both Streams)
- [ ] If NO-GO: Execute rollback plan

**Total**: 2-3 days

---

## 9. Risk Analysis

### 9.1. Risk Comparison: Phase 1a vs Phase 1b

| Risk Factor | Phase 1a (Edge) | Phase 1b (Node) | Assessment |
|-------------|-----------------|-----------------|------------|
| **Parameter count** | 10K → 7.3K | 398K → 284K | Node has **39× more params** |
| **% of total params** | 1.8% | 70% | Node is **39× more impactful** |
| **Architectural complexity** | Simple (171 pairs) | Complex (19 electrodes) | Similar (both shared modules) |
| **Expected gain** | +5-10% edge modeling | +5-10% node memory | Similar hypotheses |
| **Rollback difficulty** | Easy | Easy | Same (single builder file) |

**Key Insight**: Phase 1b affects **70% of stream parameters** (vs 1.8% for Phase 1a), making it a more significant test of GDN's benefits but also higher risk.

### 9.2. Why Phase 1b After Phase 1a?

**Strategic Reasoning**:
1. **Phase 1a (edge) validates delta rule benefits** on small scale (10K params)
2. **Phase 1b (node) validates standard SSM improvements** on larger scale (398K params)
3. **Isolates contributions**: Edge vs node stream benefits measured independently
4. **Reduces risk**: Test smaller component first, then larger component, then combined

---

## 10. Next Steps

### If Both Phase 1a AND Phase 1b Succeed:

1. **Analyze combined potential**:
   - Phase 1a: +X% sensitivity (edge stream only)
   - Phase 1b: +Y% sensitivity (node stream only)
   - Expected Phase 2: +Z% sensitivity (both streams, Z may not equal X+Y)

2. **Create Doc 3**: Full Migration Implementation Plan
   - Replace BOTH streams with BiGatedDeltaNet
   - Test fusion mode combinations (sum vs concat)
   - A/B test: Phase 1a + Phase 1b vs Phase 2 (combined)

3. **Schedule Phase 2**: Full migration with combined validation

### If Either Phase 1a OR Phase 1b Fails:

1. **Partial migration strategy**:
   - If Phase 1a succeeds but Phase 1b fails: Deploy edge stream only
   - If Phase 1b succeeds but Phase 1a fails: Deploy node stream only
   - Mixed architecture: BiGatedDeltaNet for winner, BiMamba2 for loser

2. **Root cause analysis**: Why did one stream benefit but not the other?

3. **Alternative architectures**: Consider GLA, HGRN2, or hybrid approaches

---

## 11. References

- **Doc 0 (SSOT)**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md)
- **Doc 1 (Edge Stream)**: [FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md)
- **FLA Library**: https://github.com/fla-org/flash-linear-attention
- **Gated DeltaNet Paper**: https://arxiv.org/abs/2412.06464
- **Current v3.8.3 Baseline**: RELEASE_NOTES.md

---

**Document Status**: ✅ Ready for Implementation (pending Phase 1a success)
**Next Document**: Doc 3 (Full Migration - Both Streams) - pending Phase 1b success
**Prerequisites**: Phase 1a must complete successfully before starting Phase 1b
