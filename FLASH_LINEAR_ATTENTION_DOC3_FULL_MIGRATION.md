# Doc 3: Full Migration - Both Streams → BiGatedDeltaNet

**Parent Document**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md) (Doc 0 - SSOT)
**Phase**: 2 (Full Migration)
**Target**: Replace BOTH Edge + Node Streams BiMamba2 → BiGatedDeltaNet
**Date**: October 7, 2025
**Version**: 1.0
**Status**: Ready for Implementation (pending Phase 1a AND Phase 1b success)

---

## Executive Summary

This document provides **surgical implementation details** for migrating BOTH edge and node streams from BiMamba2 to BiGatedDeltaNet simultaneously. This is Phase 2 of the phased migration strategy.

**Scope of Changes**:
- ✅ Set both streams to BiGatedDeltaNet via config
- ✅ Reuse existing `gated_deltanet.py` wrapper (from Phase 1a)
- ✅ Reuse existing builder logic (from Phases 1a/1b)
- ✅ Write integration test (`tests/integration/test_full_gdn_migration.py`)
- ✅ A/B test fusion modes (sum vs concat) for both streams
- ❌ **DO NOT TOUCH**: GNN, TCN, decoder (keep v3.8.3 baseline)

**Expected Outcome**:
- Edge stream: **10.3K → ~7.3K params** (29% reduction)
- Node stream: **398K → ~284K params** (29% reduction)
- **Total streams: 408K → ~291K params** (29% reduction, -117K params)
- Hypothesis: **+3-5% sensitivity @ 1 FA/24h** (combined effect from both streams)
- Risk: **HIGH** (100% of stream parameters affected, ~291K out of 291K)

**Timeline**: 1 day config + 6-8 hours integration test + 1 day analysis

---

## 📊 Parameter Count Analysis

**IMPORTANT**: GDN's 0.75× q/k projection reduces parameter count by ~29% for **BOTH streams**:

| Component | BiMamba2 (Baseline) | BiGatedDeltaNet (Phase 2) | Reduction |
|-----------|---------------------|---------------------------|-----------|
| **Edge Stream** | 10,304 params | **~7,352 params** | **-29%** |
| **Node Stream** | 397,632 params | **~284,000 params** | **-29%** |
| **Total Streams** | **407,936 params** | **~291,352 params** | **-29%** |

**Why this is EXPECTED and GOOD**:
- ✅ **Consistent efficiency**: Both streams benefit from 0.75× design
- ✅ **Faster inference**: -117K params = faster forward pass
- ✅ **Proven design**: GDN paper + Qwen3-Next production use
- ✅ **Combined gains**: Language models show +3.1% LongBench with shared weights

---

## 1. Prerequisites

### 1.1. Phase 1a AND Phase 1b Must Succeed

**Critical Decision Point**:
```bash
# Review Phase 1a results
cat PHASE1A_RESULTS.md  # Should show +1-2% sensitivity, no regressions

# Review Phase 1b results
cat PHASE1B_RESULTS.md  # Should show +1-2% sensitivity, no regressions

# Decision matrix:
# ✅ Both succeed → Proceed to Phase 2
# ❌ Either fails → Deploy only the successful phase, skip Phase 2
```

### 1.2. Environment Setup

```bash
# Verify both phases completed
git log --oneline | head -10  # Should show Phase 1a and 1b commits

# Verify BiGatedDeltaNet wrapper exists
python -c "from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet; print('✅ Wrapper exists')"

# Create Phase 2 branch
git checkout -b feature/full-gdn-migration
git tag v3.8.3-pre-full-migration
```

---

## 2. Implementation: Config Update Only

**Key Insight**: Phase 2 requires **ZERO new code** - only config changes!

### 2.1. File: `configs/local/full_gdn_test.yaml`

**Create new config for Phase 2 testing**:

```yaml
# Full Migration Test Config
# Phase 2: Replace BOTH edge and node streams with BiGatedDeltaNet

experiment:
  name: full_gdn_migration_test
  description: "Phase 2 - Both streams BiGatedDeltaNet test"
  seed: 42
  output_dir: results/full_gdn_test
  cache_dir: cache/tusz_mmap
  device: cuda
  log_level: INFO
  save_model: true
  save_best_only: true

  wandb:
    enabled: true
    project: seizure-v3-full-gdn
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

    # PHASE 2: Enable GDN for BOTH streams (simplified fallback approach)
    temporal_type: gated_deltanet  # Fallback applies to both streams
    # temporal_type_node: null     # Not set → uses fallback (GDN)
    # temporal_type_edge: null     # Not set → uses fallback (GDN)
    fusion_mode: sum               # Start with sum (A/B test concat later)
    allow_neg_eigval: false        # Conservative start

  # Graph configuration (V3)
  graph:
    enabled: true

    # PR-2: Bounded Edge Stream (ENABLED)
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1

    # V3: Edge stream config (now using GDN)
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2       # Edge GDN layers
    edge_mamba_d_state: 8      # Ignored for GDN
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

**Alternative: Explicit Stream-Specific Config** (if you want to be extra clear):

```yaml
mamba:
  # Explicit approach (same result as fallback)
  temporal_type: bimamba2            # Not used (overrides present)
  temporal_type_node: gated_deltanet  # Node uses GDN
  temporal_type_edge: gated_deltanet  # Edge uses GDN
  fusion_mode: sum
  allow_neg_eigval: false
```

---

## 3. Testing Strategy

### 3.1. Integration Test: `tests/integration/test_full_gdn_migration.py`

```python
"""Integration test for full GDN migration (both streams)."""

import pytest
import torch

from src.brain_brr.models.builders.edge_stream import build_edge_stream
from src.brain_brr.models.builders.node_stream import build_node_stream
from src.brain_brr.config.schemas import ModelConfig, MambaConfig, GraphConfig

# Skip if FLA not installed
pytest.importorskip("fla")


class TestFullGDNMigration:
    """Test full migration: BOTH edge and node streams use BiGatedDeltaNet."""

    @pytest.fixture
    def base_config(self):
        """Base config with both streams enabled for GDN."""
        return ModelConfig(
            mamba=MambaConfig(
                n_layers=6,
                d_model=512,
                d_state=16,
                conv_kernel=4,
                dropout=0.1,
                temporal_type="gated_deltanet",  # Fallback for both streams
            ),
            graph=GraphConfig(
                enabled=True,
                edge_mamba_layers=2,
                edge_mamba_d_state=8,
                edge_mamba_d_model=16,
            ),
        )

    def test_both_streams_use_gdn(self, base_config):
        """Test both streams build as BiGatedDeltaNet."""
        from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet

        node_mamba = build_node_stream(base_config)
        edge_components = build_edge_stream(base_config)

        # Verify both are GDN
        assert isinstance(node_mamba, BiGatedDeltaNet), "Node should be BiGatedDeltaNet"
        assert isinstance(edge_components.edge_mamba, BiGatedDeltaNet), "Edge should be BiGatedDeltaNet"

    def test_combined_parameter_count(self, base_config):
        """Test combined parameter count matches expected reduction.

        Both streams should have ~29% fewer params than BiMamba2 baseline.
        Total: 408K → ~291K params.
        """
        node_mamba = build_node_stream(base_config)
        edge_components = build_edge_stream(base_config)

        node_params = sum(p.numel() for p in node_mamba.parameters())
        edge_params = sum(p.numel() for p in edge_components.edge_mamba.parameters())
        total_params = node_params + edge_params

        print(f"Node GDN params: {node_params:,}")
        print(f"Edge GDN params: {edge_params:,}")
        print(f"Total GDN params: {total_params:,}")
        print(f"Expected: ~291K (29% reduction from 408K BiMamba2)")

        # Total should be 280K-310K (allowing some variance)
        assert 280_000 < total_params < 310_000, (
            f"Total parameter count outside expected range: {total_params:,} "
            f"(expected ~291K due to GDN's 0.75× q/k projection)"
        )

    def test_dual_stream_forward_pass(self, base_config):
        """Test full dual-stream forward pass (detector-like context)."""
        from src.brain_brr.models.edge_features import edge_scalar_series

        batch_size = 2
        num_electrodes = 19
        num_edges = 171
        seq_len = 960
        d_features = 64

        node_mamba = build_node_stream(base_config)
        edge_components = build_edge_stream(base_config)

        # Simulate detector flow
        # 1. TCN output projected to electrodes
        elec_feats = torch.randn(batch_size, num_electrodes, seq_len, d_features)

        # 2. Node stream processing
        node_flat = elec_feats.permute(0, 1, 3, 2).reshape(
            batch_size * num_electrodes, d_features, seq_len
        )
        node_processed = node_mamba(node_flat)
        node_feats = node_processed.reshape(batch_size, num_electrodes, d_features, seq_len)

        # 3. Edge stream processing
        edge_feats = edge_scalar_series(elec_feats, metric="cosine", edge_similarity_margin=0.01)
        edge_flat = edge_feats.squeeze(-1).reshape(batch_size * num_edges, 1, seq_len)
        edge_in = edge_components.edge_in_proj(edge_flat)
        edge_processed = edge_components.edge_mamba(edge_in)
        edge_out = edge_components.edge_out_proj(edge_processed)
        edge_weights = edge_out.squeeze(1).reshape(batch_size, num_edges, seq_len)

        # Verify shapes
        assert node_feats.shape == (batch_size, num_electrodes, d_features, seq_len)
        assert edge_weights.shape == (batch_size, num_edges, seq_len)

        # Verify no NaNs
        assert not torch.isnan(node_feats).any(), "NaNs in node stream output"
        assert not torch.isnan(edge_weights).any(), "NaNs in edge stream output"

    def test_gradient_flow_dual_stream(self, base_config):
        """Test gradients flow through both streams."""
        node_mamba = build_node_stream(base_config)
        edge_components = build_edge_stream(base_config)

        # Node stream
        node_x = torch.randn(8 * 19, 64, 960, requires_grad=True)
        node_y = node_mamba(node_x)
        node_loss = node_y.sum()
        node_loss.backward()
        assert node_x.grad is not None, "Node gradients not flowing"

        # Edge stream
        edge_x = torch.randn(8 * 171, 16, 960, requires_grad=True)
        edge_y = edge_components.edge_mamba(edge_x)
        edge_loss = edge_y.sum()
        edge_loss.backward()
        assert edge_x.grad is not None, "Edge gradients not flowing"

    def test_fusion_mode_sum(self, base_config):
        """Test sum fusion mode for both streams."""
        node_mamba = build_node_stream(base_config)
        edge_components = build_edge_stream(base_config)

        assert node_mamba.fusion_mode == "sum"
        assert edge_components.edge_mamba.fusion_mode == "sum"

    def test_cuda_compatibility_dual_stream(self, base_config):
        """Test CUDA compatibility with both streams."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        node_mamba = build_node_stream(base_config).cuda()
        edge_components = build_edge_stream(base_config)
        edge_components.edge_mamba = edge_components.edge_mamba.cuda()

        # Test node stream
        node_x = torch.randn(8 * 19, 64, 960).cuda()
        node_y = node_mamba(node_x)
        assert node_y.is_cuda
        assert not torch.isnan(node_y).any()

        # Test edge stream
        edge_x = torch.randn(8 * 171, 16, 960).cuda()
        edge_y = edge_components.edge_mamba(edge_x)
        assert edge_y.is_cuda
        assert not torch.isnan(edge_y).any()
```

**Run integration tests**:

```bash
pytest tests/integration/test_full_gdn_migration.py -xvs
```

---

## 4. A/B Testing Matrix

### 4.1. Fusion Mode Experiments

**Purpose**: Determine optimal fusion mode (sum vs concat) for both streams

```bash
# Config matrix (10 epochs each, 50 files)
export BGB_LIMIT_FILES=50

# 1. Baseline: BiMamba2 (both streams)
python -m src train configs/local/train.yaml --experiment.name baseline_bimamba2

# 2. Phase 2 Sum fusion (both streams)
python -m src train configs/local/full_gdn_test.yaml --experiment.name phase2_sum

# 3. Phase 2 Concat fusion (both streams)
# Edit full_gdn_test.yaml: fusion_mode: concat
python -m src train configs/local/full_gdn_test.yaml --experiment.name phase2_concat

# 4. Phase 2 Mixed: Node sum, Edge concat
# Edit: Add fusion_mode_node, fusion_mode_edge fields (future enhancement)
```

### 4.2. Comparison with Individual Phases

**Purpose**: Verify combined effect ≥ individual effects

```bash
# Compare:
# - Phase 1a (edge only): +X% sensitivity
# - Phase 1b (node only): +Y% sensitivity
# - Phase 2 (both): Expected +3-5% (may not equal X+Y due to interaction)
```

**Analysis script**:

```python
import wandb

api = wandb.Api()
runs = api.runs("seizure-v3-full-gdn")

baseline = [r for r in runs if r.name == "baseline_bimamba2"][0]
phase1a = [r for r in runs if r.name == "phase1a_gdn_edge"][0]
phase1b = [r for r in runs if r.name == "phase1b_gdn_node"][0]
phase2_sum = [r for r in runs if r.name == "phase2_sum"][0]
phase2_concat = [r for r in runs if r.name == "phase2_concat"][0]

# Compare sensitivity improvements
baseline_sens = baseline.summary['sensitivity_at_10fa']
phase1a_gain = (phase1a.summary['sensitivity_at_10fa'] - baseline_sens) / baseline_sens
phase1b_gain = (phase1b.summary['sensitivity_at_10fa'] - baseline_sens) / baseline_sens
phase2_sum_gain = (phase2_sum.summary['sensitivity_at_10fa'] - baseline_sens) / baseline_sens
phase2_concat_gain = (phase2_concat.summary['sensitivity_at_10fa'] - baseline_sens) / baseline_sens

print(f"Baseline sensitivity@10FA: {baseline_sens:.2%}")
print(f"Phase 1a (edge) gain: {phase1a_gain:+.2%}")
print(f"Phase 1b (node) gain: {phase1b_gain:+.2%}")
print(f"Phase 2 (sum) gain: {phase2_sum_gain:+.2%}")
print(f"Phase 2 (concat) gain: {phase2_concat_gain:+.2%}")

# Expected: Phase 2 ≥ max(Phase 1a, Phase 1b) and ideally Phase 2 ≈ Phase 1a + Phase 1b
```

---

## 5. Validation & Benchmarking

### 5.1. Smoke Test (3 files, 1 epoch)

```bash
export BGB_SMOKE_TEST=1
export BGB_NAN_DEBUG=1
python -m src train configs/local/full_gdn_test.yaml

# Expected:
# - Loads 3 files
# - 1 epoch completes
# - Both streams log "BiGatedDeltaNet"
# - No crashes, no NaNs
```

### 5.2. Integration Test (50 files, 10 epochs)

```bash
export BGB_LIMIT_FILES=50
python -m src train configs/local/full_gdn_test.yaml

# Monitor:
# - Loss curve (should decrease)
# - Gradient norms (should be stable)
# - Memory usage (~20GB RTX 4090)
# - Throughput (5-10% slower than BiMamba2 baseline)
```

### 5.3. Full Training (100 epochs)

**Only proceed if 10-epoch test shows promise (>= +2% sensitivity)**

```bash
# Local (RTX 4090)
python -m src train configs/local/full_gdn_test.yaml

# Modal (A100-80GB) - adjust config path
modal run --detach deploy/modal/app.py --action train --config configs/modal/full_gdn_train.yaml
```

---

## 6. Success Criteria

### 6.1. Technical Criteria

✅ **Integration tests pass**: All tests in `test_full_gdn_migration.py` pass
✅ **Smoke test completes**: 3 files, 1 epoch, no crashes
✅ **No NaNs**: Forward/backward passes produce finite values
✅ **Both streams GDN**: Logs show "Node stream: BiGatedDeltaNet" and "Edge stream: BiGatedDeltaNet"
✅ **Parameter count**: ~291K total stream params (29% reduction from 408K)

### 6.2. Performance Criteria

✅ **Convergence**: Loss decreases over 10 epochs
✅ **No regression**: val_loss ≤ baseline + 0.05
✅ **Hypothesis validated**: sensitivity_at_10fa ≥ baseline + 0.03 (+3%)
✅ **Combined effect**: Phase 2 gain ≥ max(Phase 1a, Phase 1b)
✅ **Throughput acceptable**: ≤ 10% slower than baseline
✅ **Memory usage**: ≤ baseline + 2GB

### 6.3. Go/No-Go Decision

**GO → Deploy Phase 2** if:
- All technical criteria met
- Performance improvement ≥ +3% sensitivity
- No major regressions vs baseline
- Combined effect justifies full migration

**NO-GO → Deploy Best Individual Phase** if:
- Phase 2 < max(Phase 1a, Phase 1b): Deploy whichever phase won
- Phase 2 shows regressions: Revert to best single-stream config
- Training unstable: Investigate fusion mode, learning rate, or fallback to BiMamba2

---

## 7. Rollback Plan

If Phase 2 fails or underperforms:

```bash
# Option 1: Revert to baseline (BiMamba2 both streams)
git checkout v3.8.3-pre-full-migration

# Option 2: Deploy best individual phase
# If Phase 1a (edge) won:
cp configs/local/edge_gdn_test.yaml configs/local/production.yaml

# If Phase 1b (node) won:
cp configs/local/node_gdn_test.yaml configs/local/production.yaml

# Option 3: Mixed deployment (keep whichever phase succeeded)
# Edit config to set only successful stream to GDN
```

---

## 8. Timeline & Checklist

### Day 1: Setup & Integration Test
- [ ] Verify Phases 1a and 1b both succeeded
- [ ] Review PHASE1A_RESULTS.md and PHASE1B_RESULTS.md
- [ ] Create `full_gdn_test.yaml` config
- [ ] Run integration tests
- [ ] Smoke test (3 files, 1 epoch) - 10 min
- [ ] Integration test (50 files, 10 epochs) - 6-8 hours

### Day 2: A/B Testing & Analysis
- [ ] Run fusion mode experiments (sum vs concat)
- [ ] Compare Phase 2 vs Phase 1a/1b
- [ ] Analyze W&B metrics
- [ ] Decision: sum vs concat for full run

### Day 3+: Full Training (if justified)
- [ ] Full training (100 epochs)
- [ ] Evaluate TAES metrics
- [ ] Compare vs v3.8.3 baseline
- [ ] Go/No-Go for production deployment

**Total**: 2-3 days validation + 8-12 days full training (if approved)

---

## 9. Risk Analysis

### 9.1. Risk Comparison Across Phases

| Risk Factor | Phase 1a (Edge) | Phase 1b (Node) | Phase 2 (Both) |
|-------------|-----------------|-----------------|----------------|
| **Parameters affected** | 10K (1.8%) | 398K (70%) | **408K (100%)** |
| **Expected gain** | +1-2% | +1-2% | **+3-5%** |
| **Risk level** | VERY LOW | MEDIUM | **HIGH** |
| **Rollback complexity** | Easy | Easy | **Moderate** |
| **Dependencies** | None | None | **Phases 1a+1b must succeed** |

**Key Insight**: Phase 2 is **highest risk** (100% of stream params) but also **highest reward** (+3-5% combined).

### 9.2. What Could Go Wrong

1. **Combined effect < individual effects**: If Phase 2 < max(Phase 1a, Phase 1b), deploy best individual phase
2. **Interaction penalties**: Node + edge changes may interfere (e.g., different optimal fusion modes)
3. **Training instability**: More parameters = more hyperparameter sensitivity
4. **Throughput regression**: 5-10% slower may compound with other changes

**Mitigation**: 10-epoch validation gate before committing to 100-epoch full run

---

## 10. Next Steps

### If Phase 2 Succeeds (>= +3% sensitivity):

1. **Production deployment**: Update `configs/local/train.yaml` and `configs/modal/train.yaml`
2. **Documentation**: Update README.md, ARCHITECTURE_EVOLUTION.md
3. **Benchmarking**: Run full TAES evaluation, publish results
4. **Consider Doc 4**: If short-duration seizure recall needs improvement, evaluate Hybrid/SWA

### If Phase 2 Fails or Underperforms:

1. **Deploy best individual phase**:
   - Phase 1a winner: Deploy edge GDN only
   - Phase 1b winner: Deploy node GDN only
   - Baseline: Revert to BiMamba2 if both regressed
2. **Root cause analysis**: Why didn't combined work?
   - Fusion mode mismatch?
   - Learning rate too high/low?
   - Gradient clipping too aggressive?
3. **Alternative strategies**:
   - Mixed deployment (one stream GDN, one BiMamba2)
   - Try GLA or HGRN2 instead of GDN
   - Explore Doc 4 (Hybrid/SWA) with BiMamba2 baseline

---

## 11. References

- **Doc 0 (SSOT)**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md)
- **Doc 1 (Edge Stream)**: [FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md)
- **Doc 2 (Node Stream)**: [FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md)
- **FLA Library**: https://github.com/fla-org/flash-linear-attention
- **Gated DeltaNet Paper**: https://arxiv.org/abs/2412.06464
- **Current v3.8.3 Baseline**: RELEASE_NOTES.md

---

**Document Status**: ✅ Ready for Implementation (pending Phase 1a AND Phase 1b success)
**Next Document**: Doc 4 (Optional Hybrid/SWA Expansion) - only if Phase 2 succeeds but short-event recall needs improvement
**Prerequisites**: Phases 1a AND 1b must both succeed before proceeding
