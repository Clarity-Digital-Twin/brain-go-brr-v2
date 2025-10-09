# Doc 2: Node Stream Validation - Implementation Plan

**Parent Document**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md) (Doc 0 - SSOT)
**Phase**: 1b (Validation - after Phase 0 infrastructure AND Phase 1a success)
**Target**: Validate Node Stream with BiGatedDeltaNet (edge stream determined by Phase 1a results)
**Date**: October 8, 2025
**Version**: 2.2 (Validation Strategy Update - Smoke-Only)
**Status**: Ready for Implementation (AFTER Phase 0 complete AND Phase 1a succeeds)

**⚠️ CRITICAL PREREQUISITES**:
1. Complete **Doc 0 Section 14 (Phase 0 Infrastructure)** FIRST (4-6 days)
2. Complete **Phase 1a (Edge Validation)** and confirm GO decision
3. This doc assumes config schema, constants, dependencies, builders, and tests already exist

**Changelog**:
- v2.2 (Oct 8, 2025): **VALIDATION STRATEGY UPDATE** - removed 50-file validation (smoke-only, medium validation deferred to Phase 2)
- v2.1 (Oct 8, 2025): **CRITICAL FIX**: Changed `temporal_type_edge: null` to explicit value (preserves Phase 1a choice)
- v2.1 (Oct 8, 2025): **CRITICAL FIX**: Fixed attribute path `model.edge_stream.edge_mamba` → `model.edge_mamba`
- v2.0 (Oct 8, 2025): Removed Phase 0 duplication (builder code moved to Doc 0)
- v2.0 (Oct 8, 2025): Added coexistence strategy (BiMamba2 default, GDN experimental)
- v2.0 (Oct 8, 2025): Added Phase 0 prerequisites verification checklist
- v2.0 (Oct 8, 2025): Changed "migration" → "validation" (not replacement)
- v2.0 (Oct 8, 2025): Added rollback via config (instant, not git revert)
- v1.1 (Oct 7, 2025): Fixed CLI commands to use proper config workflow
- v1.1 (Oct 7, 2025): Fixed W&B analysis to query by config.experiment.name
- v1.0 (Oct 7, 2025): Initial version

---

## Executive Summary

This document provides **validation workflow** for **testing node stream with BiGatedDeltaNet** after Phase 1a determines edge stream architecture. This is Phase 1b of the coexistence validation strategy.

**Key Philosophy**: This is NOT a "migration" or "replacement". BiMamba2 remains the **DEFAULT**. We are **VALIDATING** GDN as an experimental option by testing node stream isolation.

**Phase 1b Configuration** (assumes Phase 1a chose edge architecture):
```yaml
model:
  mamba:
    temporal_type: "bimamba2"              # Global default (stable)
    temporal_type_node: "gated_deltanet"   # Override node only (experimental)
    temporal_type_edge: "gated_deltanet"   # PRESERVE Phase 1a result (or "bimamba2" if Phase 1a kept BiMamba2)
```

**⚠️ CRITICAL**: Do NOT set `temporal_type_edge: null` - this reverts to global default (bimamba2), losing Phase 1a's choice!

**Scope of Phase 1b**:
- ✅ Node stream: BiMamba2 → BiGatedDeltaNet (experimental validation)
- ❌ **Edge stream**: KEEP Phase 1a result (stable from Phase 1a)
- ❌ **DO NOT TOUCH**: GNN, TCN, decoder (keep v3.9.0 baseline)

**What Should Already Exist** (Phase 0 infrastructure):
- ✅ Config schema with `temporal_type_node` field (Doc 0 Section 14.1)
- ✅ Constants extracted to `constants.py` (Doc 0 Section 14.2)
- ✅ `flash-linear-attention` in `pyproject.toml` (Doc 0 Section 14.3)
- ✅ Builder factory pattern (Doc 0 Section 14.4)
- ✅ BiGatedDeltaNet wrapper (Doc 0 Section 14.5)
- ✅ Test infrastructure (Doc 0 Section 14.6)
- ✅ Phase 1a completed successfully (edge stream architecture validated)

**Expected Outcome**:
- Node stream: **398K params BiMamba2 → ~284K params BiGatedDeltaNet** (29% reduction due to 0.75× q/k projection)
- Hypothesis: +5-10% better per-electrode memory → +1-2% sensitivity @ 1 FA/24h
- Risk: **MEDIUM** (larger parameter count than edge, 284K out of 398K total = 71% of stream parameters)

**Timeline**: 1 day config + smoke test (~5 min) - NO medium validation (deferred to Phase 2)

---

## 📊 Parameter Count Analysis

**⚠️ IMPORTANT - Parameter Analysis Needs Update**:

GDN's 0.75× q/k projection reduces params **within a fixed d_model**, but Phase 1a forced `edge_mamba_d_model=32` (not 16) due to FLA hardware requirements. This means edge stream params **INCREASE** compared to BiMamba2 baseline at d_model=16.

| Component | BiMamba2 (Baseline d_model) | BiGatedDeltaNet (Phase 1b) | Change |
|-----------|------------------------------|----------------------------|--------|
| **Node Stream** | 397,632 params (d_model=64) | **~284,000 params (d_model=64)** | **-29%** ✅ |
| **Edge Stream** | 10,304 params (d_model=16) | **~30K params (d_model=32, from Phase 1a)** | **+~190%** ⚠️ |
| **Total Streams** | 407,936 params | **~314K params (both GDN, edge d_model=32)** | **-23%** (mixed) |

**Key Insight**: The parameter "reduction" is NOT comparing apples-to-apples. Phase 1a increased edge capacity (d_model 16→32) for FLA compatibility, so total stream params are higher than naive 29% reduction would suggest.

**Why fewer parameters is GOOD**:
- ✅ **More parameter-efficient**: Same representational capacity with fewer params
- ✅ **Faster inference**: Fewer parameters = faster forward pass
- ✅ **Better generalization**: Reduced parameter count can improve generalization
- ✅ **By design**: GDN paper shows 0.75× allocation is intentional and performs well

**Key Insight**: The node stream has **~13× more parameters** than edge stream (398K vs ~30K after Phase 1a forced edge_d_model=32), making this a more significant test of GDN's benefits. The 29% node reduction is expected and part of GDN's parameter efficiency design.

---

## 1. Prerequisites (MUST Complete Phase 0 AND Phase 1a First)

**⚠️ BLOCKING**: You CANNOT proceed with Phase 1b until:
1. Phase 0 is complete (See **Doc 0 Section 14**)
2. Phase 1a is complete and resulted in **GO decision**

### 1.1. Verify Phase 0 Complete

**Checklist** (from Doc 0 Section 14.8):

```bash
# 1. Verify config schema has temporal_type fields
python -c "
from src.brain_brr.config.schemas import MambaConfig
cfg = MambaConfig()
assert hasattr(cfg, 'temporal_type_node'), 'Missing temporal_type_node field!'
print('✅ Config schema updated')
"

# 2. Verify constants exist
python -c "
from src.brain_brr.constants import (
    GDN_NODE_NUM_HEADS_DEFAULT,
    GDN_NODE_HEADDIM_DEFAULT,
    NODE_D_MODEL,
)
print(f'✅ Constants exist: NODE_D_MODEL={NODE_D_MODEL}')
"

# 3. Verify FLA installed
python -c "
from fla.layers import GatedDeltaNet
print('✅ FLA library installed')
"

# 4. Verify BiGatedDeltaNet wrapper exists
python -c "
from src.brain_brr.models.gated_deltanet import BiGatedDeltaNet
print('✅ BiGatedDeltaNet wrapper exists')
"

# 5. Verify builder factory pattern works
python -c "
from src.brain_brr.models.builders.node_stream import build_node_stream
from src.brain_brr.config.schemas import ModelConfig, MambaConfig

# Test BiMamba2 (default)
cfg = ModelConfig(mamba=MambaConfig(temporal_type='bimamba2'))
node = build_node_stream(cfg)
print(f'✅ Builder returns BiMamba2: {type(node).__name__}')

# Test GDN (experimental)
cfg.mamba.temporal_type_node = 'gated_deltanet'
node_gdn = build_node_stream(cfg)
print(f'✅ Builder returns BiGatedDeltaNet: {type(node_gdn).__name__}')
"
```

**If ANY check fails**: Go back to Doc 0 Section 14 and complete Phase 0 infrastructure.

### 1.2. Verify Phase 1a Complete

**Checklist**:

```bash
# 1. Check Phase 1a results exist
python -c "
import wandb
api = wandb.Api()
runs = api.runs('seizure-v3-fla-validation')  # Or your W&B project

# Find Phase 1a run
phase1a = [r for r in runs if r.config.get('experiment', {}).get('name') == 'phase1a_edge_gdn']
if not phase1a:
    print('❌ Phase 1a run not found')
    exit(1)

run = phase1a[0]
sens_10fa = run.summary.get('sensitivity_at_10fa', 0)
print(f'✅ Phase 1a complete: sensitivity@10FA={sens_10fa:.2%}')
"

# 2. Confirm GO decision
# Review Phase 1a results manually:
# - sensitivity@10FA improvement >= +1% OR no regression > -0.5%
# - No training instabilities (NaNs, divergence)
# - Throughput acceptable (<= 10% slower)

# If Phase 1a was NO-GO, you should NOT proceed to Phase 1b
# Instead, consider alternative strategies or revert to BiMamba2
```

**If Phase 1a was NO-GO**: Do NOT proceed with Phase 1b. Consider:
- Investigate why edge stream failed
- Try alternative architectures (GLA, pure DeltaNet)
- Revert to BiMamba2 baseline

### 1.3. Establish Baseline Metrics (Optional)

**Purpose**: Record current node stream performance for comparison (if Phase 1a changed edge stream).

```bash
# Run baseline with Phase 1a edge stream configuration
export BGB_LIMIT_FILES=50
cp configs/local/train.yaml configs/local/baseline_node.yaml

# Edit configs/local/baseline_node.yaml:
#   experiment.name: "baseline_node"
#   training.epochs: 10
#   model.mamba.temporal_type_edge: <result from Phase 1a>

python -m src train configs/local/baseline_node.yaml

# Record metrics from W&B:
# - val_loss
# - sensitivity_at_10fa
# - sensitivity_at_5fa
# - sensitivity_at_1fa
```

**Note**: This step is OPTIONAL if you already have Phase 1a baseline metrics with edge stream architecture fixed.

---

## 2. Phase 1b Configuration

**This is the ACTUAL Phase 1b work** - creating a test config that enables GDN for node stream only.

### 2.1. Create Phase 1b Config

```bash
# Copy Phase 1a config (to preserve edge stream choice)
cp configs/local/phase1a_edge_gdn.yaml configs/local/phase1b_node_gdn.yaml

# CRITICAL: The copied config MUST preserve Phase 1a's temporal_type_edge value
# - If Phase 1a chose GDN: temporal_type_edge: "gated_deltanet"
# - If Phase 1a kept BiMamba2: temporal_type_edge: "bimamba2"
# DO NOT set temporal_type_edge: null - this will revert to global default!
```

### 2.2. Edit Config - Key Changes

**File**: `configs/local/phase1b_node_gdn.yaml`

**Make these changes**:

```yaml
# 1. Update experiment section
experiment:
  name: phase1b_node_gdn  # CHANGE THIS
  description: "Phase 1b - Node stream GDN validation"
  output_dir: results/phase1b_node_gdn

  wandb:
    enabled: true
    project: seizure-v3-fla-validation  # CHANGE THIS
    entity: null

# 2. Enable GDN for node stream ONLY (stream-specific control)
model:
  mamba:
    temporal_type: bimamba2              # Global default (not used if stream-specific set)
    temporal_type_node: gated_deltanet   # Override: node uses GDN
    temporal_type_edge: "gated_deltanet" # PRESERVE Phase 1a result (change to "bimamba2" if Phase 1a kept BiMamba2)
    gdn_fusion_mode: sum                 # Start with simpler sum fusion
    gdn_allow_neg_eigval: false          # Conservative start

# ⚠️ CRITICAL: Do NOT set temporal_type_edge: null - this reverts to global default (bimamba2)
# Instead, explicitly set to Phase 1a's choice: "gated_deltanet" OR "bimamba2"

# 3. Short validation run
training:
  epochs: 10  # Short test (not full 100)
```

### 2.3. Complete Config Template

**File**: `configs/local/phase1b_node_gdn.yaml` (complete example)

```yaml
# Phase 1b: Node Stream GDN Validation
# Tests node stream with BiGatedDeltaNet while keeping edge stream from Phase 1a

experiment:
  name: phase1b_node_gdn
  description: "Phase 1b - Node stream GDN validation (edge from Phase 1a)"
  seed: 42
  output_dir: results/phase1b_node_gdn
  cache_dir: cache/tusz_mmap
  device: cuda
  log_level: INFO
  save_model: true
  save_best_only: true

  wandb:
    enabled: true
    project: seizure-v3-fla-validation
    entity: null

model:
  architecture: v3

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

    # PHASE 1b: GDN for node stream ONLY
    temporal_type: bimamba2              # Global default (not used if stream-specific set)
    temporal_type_node: gated_deltanet   # Override: node uses GDN
    temporal_type_edge: "gated_deltanet" # PRESERVE Phase 1a result (or "bimamba2" if Phase 1a kept BiMamba2)
    gdn_fusion_mode: sum                 # Bidirectional fusion
    gdn_allow_neg_eigval: false          # Conservative start

    # Edge GDN head config (required for edge_mamba_d_model=32)
    # Constraint: num_heads × headdim = 0.75 × d_model
    # 3 × 8 = 24 = 0.75 × 32 ✅
    gdn_edge_num_heads: 3
    gdn_edge_headdim: 8

  graph:
    enabled: true
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1
    edge_features: cosine
    edge_top_k: 3
    edge_threshold: 1.0e-4
    edge_mamba_layers: 2
    edge_mamba_d_state: 8
    edge_mamba_d_model: 32
    edge_similarity_margin: 0.01
    adj_row_softmax: true
    adj_softmax_tau: 1.0
    adj_ema_beta: 0.95
    adj_force_symmetric: true
    laplacian_eps: 1.0e-3
    laplacian_normalize: true
    n_layers: 2
    dropout: 0.1
    use_residual: true
    alpha: 0.05
    k_eigenvectors: 16
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
  epochs: 10  # Short validation run

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

## 3. Validation Workflow

### 3.1. Smoke Test (Optional - 3 files, 1 epoch)

**Purpose**: Quick sanity check before full validation.

```bash
# Run smoke test
export BGB_SMOKE_TEST=1
python -m src train configs/local/phase1b_node_gdn.yaml

# Expected output:
# - Loads 3 files
# - Completes 1 epoch
# - No crashes, no NaNs
# - Logs show "Node stream: BiGatedDeltaNet"
# - Logs show edge stream from Phase 1a
```

### 3.2. Full Validation (DEPRECATED - NEW STRATEGY)

**⚠️ NO LONGER APPLICABLE** (as of Oct 8, 2025):

Under the new validation strategy, Phase 1b uses **smoke test ONLY** (3 files). Full 50-file validation is **NOT performed** for individual phases.

**NEW STRATEGY**:
- Phase 1a/1b: Smoke tests only (3 files, quick validation)
- After Phase 2 complete: ONE medium validation run (40-50 files, 5-6 epochs)
- Then: Full Modal training for A/B comparison

**Rationale**: 50-file per-phase validation has high variance with 12:1 imbalance, insufficient for statistical significance. Full validation deferred to Phase 2 medium run.

**OLD WORKFLOW** (for reference - DO NOT USE):
```bash
# OLD: 50-file validation per phase (DEPRECATED)
export BGB_LIMIT_FILES=50
python -m src train configs/local/phase1b_node_gdn.yaml
# Training time: ~6-8 hours on RTX 4090
```

**See**: Doc 1 Implementation Status section for complete validation strategy

### 3.3. Verify Isolation (Node GDN, Edge from Phase 1a)

**Purpose**: Confirm edge stream kept Phase 1a result, node stream switched to GDN.

```bash
# Check logs for confirmation
grep "Node stream:" <logfile>
# Expected: "Node stream: BiGatedDeltaNet"

grep "Edge stream:" <logfile>
# Expected: "Edge stream: BiGatedDeltaNet" (if Phase 1a chose GDN)
#        OR "Edge stream: BiMamba2" (if Phase 1a kept BiMamba2)

# Or verify in Python (correct way - checkpoint is a dict, not model):
python -c "
import torch
from src.brain_brr.config.schemas import Config
from src.brain_brr.models.detector import SeizureDetector

# Load config
config = Config.from_yaml('configs/local/phase1b_node_gdn.yaml')

# Instantiate model
model = SeizureDetector(config.model)

# Load checkpoint dict
checkpoint = torch.load('results/phase1b_node_gdn/checkpoints/best.pt', map_location='cpu', weights_only=False)

# Load state dict into model
model.load_state_dict(checkpoint['model_state_dict'])

# Verify isolation
print(f'Node: {type(model.node_mamba).__name__}')
print(f'Edge: {type(model.edge_mamba).__name__}')
"
# Expected:
# Node: BiGatedDeltaNet
# Edge: BiGatedDeltaNet (if Phase 1a chose GDN) OR BiMamba2 (if Phase 1a kept BiMamba2)
```

---

## 4. A/B Comparison

### 4.1. Metrics to Compare

| Metric | Source | Why Important |
|--------|--------|---------------|
| `val_loss` | W&B | Overall convergence |
| `sensitivity_at_10fa` | W&B | Primary metric (10 FA/24h) |
| `sensitivity_at_5fa` | W&B | Secondary metric (5 FA/24h) |
| `sensitivity_at_1fa` | W&B | Stretch goal (1 FA/24h) |
| `throughput` | Logs | Training efficiency |
| `memory_peak` | Logs | Resource usage |

### 4.2. Analysis Script

**File**: `scripts/analyze_phase1b.py` (robust W&B analysis)

```python
"""Compare Phase 1b (node GDN) results against Phase 1a baseline.

USAGE:
    python scripts/analyze_phase1b.py --project seizure-v3-fla-validation
"""
import argparse
import sys
import wandb


def find_run_by_experiment_name(runs: list, experiment_name: str):
    """Find run by experiment.name config field (robust to W&B name suffixes)."""
    matches = [
        r for r in runs
        if r.config.get('experiment', {}).get('name') == experiment_name
    ]

    if not matches:
        print(f"ERROR: No runs found with experiment.name='{experiment_name}'")
        return None
    if len(matches) > 1:
        print(f"Warning: Found {len(matches)} runs, using first")

    return matches[0]


def get_metric(run, metric: str, default: float = 0.0) -> float:
    """Safely fetch a scalar metric from W&B run summary."""
    value = run.summary.get(metric)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 1b validation results")
    parser.add_argument('--project', required=True, help='W&B project name')
    parser.add_argument('--entity', default=None, help='W&B entity (optional)')
    args = parser.parse_args()

    # Initialize API
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(project_path)

    # Find runs by experiment.name (robust to W&B suffixes)
    phase1a = find_run_by_experiment_name(runs, "phase1a_edge_gdn")
    phase1b = find_run_by_experiment_name(runs, "phase1b_node_gdn")

    # Verify both found
    if not phase1a or not phase1b:
        print("\nAvailable runs:")
        for r in runs:
            exp_name = r.config.get('experiment', {}).get('name', 'UNKNOWN')
            print(f"  - {r.id}: {r.name} (experiment.name={exp_name})")
        sys.exit(1)

    # Extract metrics
    phase1a_loss = get_metric(phase1a, "val_loss")
    phase1b_loss = get_metric(phase1b, "val_loss")

    phase1a_sens_10fa = get_metric(phase1a, "sensitivity_at_10fa")
    phase1a_sens_5fa = get_metric(phase1a, "sensitivity_at_5fa")
    phase1a_sens_1fa = get_metric(phase1a, "sensitivity_at_1fa")

    phase1b_sens_10fa = get_metric(phase1b, "sensitivity_at_10fa")
    phase1b_sens_5fa = get_metric(phase1b, "sensitivity_at_5fa")
    phase1b_sens_1fa = get_metric(phase1b, "sensitivity_at_1fa")

    # Calculate gains
    loss_delta = (phase1a_loss - phase1b_loss) / phase1a_loss if phase1a_loss > 0 else 0
    sens_10fa_gain = (phase1b_sens_10fa - phase1a_sens_10fa) / phase1a_sens_10fa if phase1a_sens_10fa > 0 else 0
    sens_5fa_gain = (phase1b_sens_5fa - phase1a_sens_5fa) / phase1a_sens_5fa if phase1a_sens_5fa > 0 else 0
    sens_1fa_gain = (phase1b_sens_1fa - phase1a_sens_1fa) / phase1a_sens_1fa if phase1a_sens_1fa > 0 else 0

    # Print results
    print("=" * 80)
    print("Phase 1b: Node Stream GDN Validation Results")
    print("=" * 80)

    print(f"\n📊 Phase 1a (edge stream validated):")
    print(f"  val_loss: {phase1a_loss:.4f}")
    print(f"  sensitivity@10FA: {phase1a_sens_10fa:.2%}")
    print(f"  sensitivity@5FA:  {phase1a_sens_5fa:.2%}")
    print(f"  sensitivity@1FA:  {phase1a_sens_1fa:.2%}")

    print(f"\n📊 Phase 1b (node stream GDN):")
    print(f"  val_loss: {phase1b_loss:.4f} ({loss_delta:+.2%})")
    print(f"  sensitivity@10FA: {phase1b_sens_10fa:.2%} ({sens_10fa_gain:+.2%})")
    print(f"  sensitivity@5FA:  {phase1b_sens_5fa:.2%} ({sens_5fa_gain:+.2%})")
    print(f"  sensitivity@1FA:  {phase1b_sens_1fa:.2%} ({sens_1fa_gain:+.2%})")

    # Decision
    print("\n" + "=" * 80)
    print("📋 Go/No-Go Decision")
    print("=" * 80)

    if sens_10fa_gain >= 0.01:  # +1% or better
        print(f"\n✅ GO → Proceed to Phase 2 (Full Validation - Both Streams)")
        print(f"   Reason: Phase 1b shows {sens_10fa_gain:+.2%} improvement")
        print(f"   Node stream GDN validated successfully")
    elif sens_10fa_gain >= -0.005:  # No regression > -0.5%
        print(f"\n⚠️ MARGINAL → Consider proceeding with caution")
        print(f"   Reason: Phase 1b shows {sens_10fa_gain:+.2%} (neutral)")
        print(f"   No clear improvement, but no major regression")
    else:
        print(f"\n❌ NO-GO → Revert to Phase 1a configuration")
        print(f"   Reason: Phase 1b shows {sens_10fa_gain:+.2%} (regression)")
        print(f"   Node stream GDN does not improve performance")

    print("=" * 80)


if __name__ == '__main__':
    main()
```

**Run analysis**:

```bash
python scripts/analyze_phase1b.py --project seizure-v3-fla-validation
```

---

## 5. Rollback Procedure

**If Phase 1b underperforms or causes issues**, rollback is INSTANT via config change (no git operations needed).

### 5.1. Instant Rollback

**Edit config to revert node stream to BiMamba2**:

```yaml
# Edit configs/local/phase1b_node_gdn.yaml:
model:
  mamba:
    temporal_type: bimamba2              # Keep as is
    temporal_type_node: bimamba2         # CHANGE: revert to BiMamba2
    # temporal_type_node: gated_deltanet  # Comment out or remove
```

**Re-run training**:

```bash
# Restart with BiMamba2 node stream
export BGB_LIMIT_FILES=50
python -m src train configs/local/phase1b_node_gdn.yaml

# Instant rollback - no code changes needed!
```

### 5.2. Why This is Safe

- ✅ BiMamba2 code untouched (still works)
- ✅ GDN is additive (not replacement)
- ✅ Config flag controls behavior
- ✅ No checkpoint migration needed (separate checkpoints per architecture)
- ✅ Zero code changes required

### 5.3. Git Rollback (Last Resort)

**Only if config rollback fails**:

```bash
# Revert to pre-Phase-1b state (NOT RECOMMENDED - use config rollback instead)
git checkout v3.9.0-pre-phase1b

# Or revert specific commits
git revert HEAD~1

# Verify baseline restored
python -m src train configs/local/train.yaml
```

**Note**: Git rollback should NOT be needed - config rollback is instant and safer.

---

## 6. Success Criteria

### 6.1. Technical Criteria (Must Pass)

- [ ] ✅ Smoke test completes (3 files, 1 epoch, no crashes)
- [ ] ✅ No NaNs in forward/backward passes
- [ ] ✅ Shapes correct (input/output match BiMamba2)
- [ ] ✅ Isolation verified (node=GDN, edge=Phase 1a result)
- [ ] ✅ Parameter count ~284K node (29% reduction expected)
- [ ] ✅ Convergence over 10 epochs (loss decreases)

### 6.2. Performance Criteria (Go/No-Go)

**GO → Phase 2** if:
- [ ] ✅ sensitivity@10FA ≥ Phase 1a + 1% **OR** no regression > 0.5%
- [ ] ✅ No major regressions (loss ≤ Phase 1a + 5%)
- [ ] ✅ Throughput ≤ 10% slower than Phase 1a
- [ ] ✅ Memory usage ≤ Phase 1a + 2GB
- [ ] ✅ Phase 1a also succeeded (edge stream validated)

**NO-GO → Revert** if:
- [ ] ❌ sensitivity@10FA regression > 1%
- [ ] ❌ Training unstable (NaNs, divergence)
- [ ] ❌ Throughput regression > 20%

---

## 7. Next Steps

### If Both Phase 1a AND Phase 1b Succeed:

1. **Analyze combined potential** in `PHASE1B_RESULTS.md`
2. **Proceed to Doc 3**: Full Stream Validation (Phase 2)
3. **Continue phased validation**: Doc 4 (Hybrid SWA if needed)

### If Phase 1b Fails (but Phase 1a Succeeded):

1. **Document failure** in `PHASE1B_POSTMORTEM.md`
2. **Deploy Phase 1a only**: Keep edge stream GDN, revert node to BiMamba2
3. **Mixed architecture**: BiGatedDeltaNet edge + BiMamba2 node (partial benefit)
4. **Root cause analysis**: Why did node stream fail but edge succeeded?

### If Both Phase 1a AND Phase 1b Fail:

1. **Revert to v3.9.0 baseline**: BiMamba2 for both streams
2. **Alternative architectures**: Consider GLA, HGRN2, or hybrid approaches
3. **Investigate**: Why didn't FLA benefit this specific architecture?

---

## 8. Timeline & Checklist

### Day 1: Setup
- [ ] Verify Phase 0 complete (Section 1.1)
- [ ] Verify Phase 1a succeeded (Section 1.2)
- [ ] Create phase1b config (Section 2)
- [ ] Optional: Run baseline if needed (Section 1.3)
- [ ] Smoke test (Section 3.1) - 10 min

### Day 2: Validation
- [ ] Full validation run (Section 3.2) - 6-8 hours
- [ ] Monitor training (loss, gradients, memory)
- [ ] Verify isolation (Section 3.3)

### Day 3: Analysis & Decision
- [ ] Run A/B analysis (Section 4.2)
- [ ] Review metrics vs success criteria (Section 6)
- [ ] Go/No-Go decision
- [ ] Document results
- [ ] If GO: Proceed to Phase 2 (Doc 3)
- [ ] If NO-GO: Rollback (Section 5)

**Total Timeline**: 2-3 days (after Phase 0 AND Phase 1a complete)

---

## 9. References

- **Doc 0 (SSOT)**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md)
  - Section 14: Phase 0 Infrastructure (wrapper, builders, schema, tests)
  - Section 15: Coexistence Strategy
- **Doc 1 (Edge Validation)**: [FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md)
- **Doc 3 (Next)**: Full Stream Validation (Phase 2) - pending Phase 1b success
- **FLA Library**: https://github.com/fla-org/flash-linear-attention
- **Gated DeltaNet Paper**: https://arxiv.org/abs/2412.06464

---

## 10. Risk Analysis

### 10.1. Risk Comparison: Phase 1a vs Phase 1b

| Risk Factor | Phase 1a (Edge) | Phase 1b (Node) | Assessment |
|-------------|-----------------|-----------------|------------|
| **Parameter count** | 10K → 7.3K | 398K → 284K | Node has **39× more params** |
| **% of stream params** | 2.5% | 97.5% | Node is **39× more impactful** |
| **Architectural complexity** | Simple (171 pairs) | Complex (19 electrodes) | Similar (both shared modules) |
| **Expected gain** | +5-10% edge modeling | +5-10% node memory | Similar hypotheses |
| **Rollback difficulty** | Easy (config) | Easy (config) | Same (instant config rollback) |

**Key Insight**: Phase 1b affects **97.5% of stream parameters** (vs 2.5% for Phase 1a), making it the PRIMARY test of GDN's benefits.

### 10.2. Why Phase 1b After Phase 1a?

**Strategic Reasoning**:
1. **Phase 1a (edge) validates delta rule benefits** on small scale (10K params)
2. **Phase 1b (node) validates standard SSM improvements** on larger scale (398K params)
3. **Isolates contributions**: Edge vs node stream benefits measured independently
4. **Reduces risk**: Test smaller component first, then larger component, then combined (Phase 2)

---

**Document Status**: ✅ Ready for Implementation (AFTER Phase 0 complete AND Phase 1a succeeds)
**Dependencies**:
1. Doc 0 Section 14 (Phase 0 Infrastructure) must be complete
2. Phase 1a (Edge Validation) must succeed with GO decision
**Next Document**: Doc 3 (Full Stream Validation) - pending Phase 1b success
