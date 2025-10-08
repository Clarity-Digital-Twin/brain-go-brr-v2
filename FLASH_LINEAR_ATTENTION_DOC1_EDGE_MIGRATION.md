# Doc 1: Edge Stream Validation - Implementation Plan

**Parent Document**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md) (Doc 0 - SSOT)
**Phase**: 1a (Validation - after Phase 0 infrastructure)
**Target**: Validate Edge Stream with BiGatedDeltaNet (node stays BiMamba2)
**Date**: October 7, 2025
**Version**: 2.0 (Coexistence + Infrastructure Prerequisites)
**Status**: Ready for Implementation (AFTER Phase 0 complete)

**⚠️ CRITICAL PREREQUISITE**: Complete **Doc 0 Section 14 (Phase 0 Infrastructure)** FIRST (4-6 days). This doc assumes config schema, constants, dependencies, builders, and tests already exist.

**Changelog**:
- v2.0 (Oct 7, 2025): Removed Phase 0 duplication (wrapper/builder/schema code moved to Doc 0)
- v2.0 (Oct 7, 2025): Added coexistence strategy (BiMamba2 default, GDN experimental)
- v2.0 (Oct 7, 2025): Added Phase 0 prerequisites verification checklist
- v2.0 (Oct 7, 2025): Changed "migration" → "validation" (not replacement)
- v2.0 (Oct 7, 2025): Added rollback via config (instant, not git revert)
- v1.1 (Oct 7, 2025): Fixed CLI commands to use proper config workflow (no --experiment.name hacks)
- v1.1 (Oct 7, 2025): Fixed W&B analysis to query by config.experiment.name (robust to name suffixes)
- v1.0 (Oct 7, 2025): Initial version

---

## Executive Summary

This document provides **validation workflow** for **testing edge stream with BiGatedDeltaNet** while keeping the node stream on BiMamba2. This is Phase 1a of the coexistence validation strategy.

**Key Philosophy**: This is NOT a "migration" or "replacement". BiMamba2 remains the **DEFAULT**. We are **VALIDATING** GDN as an experimental option by testing edge stream in isolation.

**Phase 1a Configuration**:
```yaml
model:
  mamba:
    temporal_type: "bimamba2"              # Global default (stable)
    temporal_type_node: null               # null = use global (BiMamba2)
    temporal_type_edge: "gated_deltanet"   # Override edge only (experimental)
```

**Scope of Phase 1a**:
- ✅ Edge stream: BiMamba2 → BiGatedDeltaNet (experimental validation)
- ❌ **Node stream**: KEEP BiMamba2 (stable baseline)
- ❌ **DO NOT TOUCH**: GNN, TCN, decoder (keep v3.8.3 baseline)

**What Should Already Exist** (Phase 0 infrastructure):
- ✅ Config schema with `temporal_type_edge` field (Doc 0 Section 14.1)
- ✅ Constants extracted to `constants.py` (Doc 0 Section 14.2)
- ✅ `flash-linear-attention` in `pyproject.toml` (Doc 0 Section 14.3)
- ✅ Builder factory pattern (Doc 0 Section 14.4)
- ✅ BiGatedDeltaNet wrapper (Doc 0 Section 14.5)
- ✅ Test infrastructure (Doc 0 Section 14.6)

**Expected Outcome**:
- Edge stream: **10.3K params BiMamba2 → ~7.3K params BiGatedDeltaNet** (29% reduction due to 0.75× q/k projection)
- Hypothesis: +5-10% better connectivity modeling → +1-2% sensitivity @ 1 FA/24h
- Risk: **VERY LOW** (only ~1.8% of stream parameters affected, 7.3K out of 405K total)

**Timeline**: 1 day config + 6-8 hours validation + 1 day analysis (assumes Phase 0 complete)

---

## 📊 Parameter Count Analysis

**UPDATED (Oct 8, 2025)**: FLA's causal_conv1d requires `edge_d_model=32` (hardware alignment, not 16). This **INCREASES** edge stream parameters vs baseline:

| Component | BiMamba2 (Baseline) | BiGatedDeltaNet (Phase 1a) | Change |
|-----------|---------------------|----------------------------|--------|
| **Edge Stream** | 10,304 params (d_model=16) | **~TBD params (d_model=32)** | **+~100%** |
| **Node Stream** | 397,632 params | (unchanged - stays BiMamba2) | N/A |
| **Total Streams** | 407,936 params | **~TBD params** | **+~X%** |

**Why this is STILL VALID**:
- ✅ **Not comparing capacity**: Phase 1a tests GDN's **algorithm** (gating + delta rule), not parameter efficiency
- ✅ **Isolation test**: Edge stream is only ~2.5% of total model (31M params), so +100% edge params = <1% total increase
- ✅ **Fair A/B later**: For capacity comparison, run BiMamba2 edge at d_model=32 as matched baseline
- ✅ **Primary goal**: Validate GDN stability, dtype handling, and convergence on small stream first

**Key Insight**: Parameter count shifted due to hardware constraint, but **algorithmic validation** (Phase 1a goal) remains unaffected. Total model impact: <1%.

---

## 1. Prerequisites (MUST Complete Phase 0 First)

**⚠️ BLOCKING**: You CANNOT proceed with Phase 1a until Phase 0 is complete. See **Doc 0 Section 14** for full infrastructure setup.

### 1.1. Verify Phase 0 Complete

**Checklist** (from Doc 0 Section 14.8):

```bash
# 1. Verify config schema has temporal_type fields
python -c "
from src.brain_brr.config.schemas import MambaConfig
cfg = MambaConfig()
assert hasattr(cfg, 'temporal_type_edge'), 'Missing temporal_type_edge field!'
print('✅ Config schema updated')
"

# 2. Verify constants exist
python -c "
from src.brain_brr.constants import (
    GDN_EDGE_NUM_HEADS_DEFAULT,
    GDN_EDGE_HEADDIM_DEFAULT,
    EDGE_D_MODEL,
)
print(f'✅ Constants exist: EDGE_D_MODEL={EDGE_D_MODEL}')
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
from src.brain_brr.models.builders.edge_stream import build_edge_stream
from src.brain_brr.config.schemas import ModelConfig, MambaConfig

# Test BiMamba2 (default)
cfg = ModelConfig(mamba=MambaConfig(temporal_type='bimamba2'))
edge = build_edge_stream(cfg)
print(f'✅ Builder returns BiMamba2: {type(edge.edge_mamba).__name__}')

# Test GDN (experimental)
cfg.mamba.temporal_type_edge = 'gated_deltanet'
edge_gdn = build_edge_stream(cfg)
print(f'✅ Builder returns BiGatedDeltaNet: {type(edge_gdn.edge_mamba).__name__}')
"
```

**If ANY check fails**: Go back to Doc 0 Section 14 and complete Phase 0 infrastructure.

### 1.2. Establish Baseline Metrics (Optional)

**Purpose**: Record current BiMamba2 edge stream performance for comparison.

```bash
# Run baseline (BiMamba2 edge, 10 epochs, 50 files)
export BGB_LIMIT_FILES=50
cp configs/local/train.yaml configs/local/baseline_edge.yaml

# Edit configs/local/baseline_edge.yaml:
#   experiment.name: "baseline_edge"
#   training.epochs: 10

python -m src train configs/local/baseline_edge.yaml

# Record metrics from W&B:
# - val_loss
# - sensitivity_at_10fa
# - sensitivity_at_5fa
# - sensitivity_at_1fa
```

**Note**: This step is OPTIONAL if you already have v3.8.3 baseline metrics.

---

## 2. Phase 1a Configuration

**This is the ACTUAL Phase 1a work** - creating a test config that enables GDN for edge stream only.

### 2.1. Create Phase 1a Config

```bash
# Copy base config
cp configs/local/train.yaml configs/local/phase1a_edge_gdn.yaml
```

### 2.2. Edit Config - Key Changes

**File**: `configs/local/phase1a_edge_gdn.yaml`

**Make these changes**:

```yaml
# 1. Update experiment section
experiment:
  name: phase1a_edge_gdn  # CHANGE THIS
  description: "Phase 1a - Edge stream GDN validation"
  output_dir: results/phase1a_edge_gdn

  wandb:
    enabled: true
    project: seizure-v3-fla-validation  # CHANGE THIS
    entity: null

# 2. Enable GDN for edge stream ONLY (stream-specific control)
model:
  mamba:
    temporal_type: bimamba2              # Global default (fallback for node)
    temporal_type_node: null             # null = use global (BiMamba2)
    temporal_type_edge: gated_deltanet   # Override: edge uses GDN
    gdn_fusion_mode: sum                 # Start with simpler sum fusion
    gdn_allow_neg_eigval: false          # Conservative start

# 3. Short validation run
training:
  epochs: 10  # Short test (not full 100)
```

### 2.3. Complete Config Template

**File**: `configs/local/phase1a_edge_gdn.yaml` (complete example)

```yaml
# Phase 1a: Edge Stream GDN Validation
# Tests edge stream with BiGatedDeltaNet while keeping node stream on BiMamba2

experiment:
  name: phase1a_edge_gdn
  description: "Phase 1a - Edge stream GDN validation (node stays BiMamba2)"
  seed: 42
  output_dir: results/phase1a_edge_gdn
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

    # PHASE 1a: GDN for edge stream ONLY
    temporal_type: bimamba2              # Global default (node fallback)
    temporal_type_node: null             # null = use global (BiMamba2)
    temporal_type_edge: gated_deltanet   # Override: edge uses GDN
    gdn_fusion_mode: sum                 # Bidirectional fusion
    gdn_allow_neg_eigval: false          # Conservative start

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
    edge_mamba_d_model: 16
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
python -m src train configs/local/phase1a_edge_gdn.yaml

# Expected output:
# - Loads 3 files
# - Completes 1 epoch
# - No crashes, no NaNs
# - Logs show "Edge stream: BiGatedDeltaNet"
```

### 3.2. Full Validation (50 files, 10 epochs)

**Purpose**: Validate Phase 1a hypothesis against baseline.

```bash
# Run Phase 1a validation
export BGB_LIMIT_FILES=50
python -m src train configs/local/phase1a_edge_gdn.yaml

# Monitor during training:
# - Loss curve (should decrease)
# - Gradient norms (should be stable)
# - Memory usage (~20GB on RTX 4090, similar to baseline)
# - Throughput (may be 5-10% slower than baseline)

# Training time: ~6-8 hours on RTX 4090
```

### 3.3. Verify Isolation (Edge GDN, Node BiMamba2)

**Purpose**: Confirm node stream stayed on BiMamba2.

```bash
# Check logs for confirmation
grep "Node stream:" <logfile>
# Expected: "Node stream: BiMamba2"

grep "Edge stream:" <logfile>
# Expected: "Edge stream: BiGatedDeltaNet"

# Or verify in Python (correct way - checkpoint is a dict, not model):
python -c "
import torch
from src.brain_brr.config.schemas import Config
from src.brain_brr.models.detector import SeizureDetector

# Load config
config = Config.from_yaml('configs/local/phase1a_edge_gdn.yaml')

# Instantiate model
model = SeizureDetector(config.model)

# Load checkpoint dict
checkpoint = torch.load('results/phase1a_edge_gdn/checkpoints/best.pt', map_location='cpu', weights_only=False)

# Load state dict into model
model.load_state_dict(checkpoint['model_state_dict'])

# Verify isolation
print(f'Node: {type(model.node_mamba).__name__}')
print(f'Edge: {type(model.edge_stream.edge_mamba).__name__}')
"
# Expected:
# Node: BiMamba2
# Edge: BiGatedDeltaNet
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

**File**: `scripts/analyze_phase1a.py` (robust W&B analysis)

```python
"""Compare Phase 1a (edge GDN) results against baseline (both BiMamba2).

USAGE:
    python scripts/analyze_phase1a.py --project seizure-v3-fla-validation
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
    parser = argparse.ArgumentParser(description="Analyze Phase 1a validation results")
    parser.add_argument('--project', required=True, help='W&B project name')
    parser.add_argument('--entity', default=None, help='W&B entity (optional)')
    args = parser.parse_args()

    # Initialize API
    api = wandb.Api()
    project_path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(project_path)

    # Find runs by experiment.name (robust to W&B suffixes)
    baseline = find_run_by_experiment_name(runs, "baseline_edge")
    phase1a = find_run_by_experiment_name(runs, "phase1a_edge_gdn")

    # Verify both found
    if not baseline or not phase1a:
        print("\nAvailable runs:")
        for r in runs:
            exp_name = r.config.get('experiment', {}).get('name', 'UNKNOWN')
            print(f"  - {r.id}: {r.name} (experiment.name={exp_name})")
        sys.exit(1)

    # Extract metrics
    baseline_loss = get_metric(baseline, "val_loss")
    phase1a_loss = get_metric(phase1a, "val_loss")

    baseline_sens_10fa = get_metric(baseline, "sensitivity_at_10fa")
    baseline_sens_5fa = get_metric(baseline, "sensitivity_at_5fa")
    baseline_sens_1fa = get_metric(baseline, "sensitivity_at_1fa")

    phase1a_sens_10fa = get_metric(phase1a, "sensitivity_at_10fa")
    phase1a_sens_5fa = get_metric(phase1a, "sensitivity_at_5fa")
    phase1a_sens_1fa = get_metric(phase1a, "sensitivity_at_1fa")

    # Calculate gains
    loss_delta = (baseline_loss - phase1a_loss) / baseline_loss if baseline_loss > 0 else 0
    sens_10fa_gain = (phase1a_sens_10fa - baseline_sens_10fa) / baseline_sens_10fa if baseline_sens_10fa > 0 else 0
    sens_5fa_gain = (phase1a_sens_5fa - baseline_sens_5fa) / baseline_sens_5fa if baseline_sens_5fa > 0 else 0
    sens_1fa_gain = (phase1a_sens_1fa - baseline_sens_1fa) / baseline_sens_1fa if baseline_sens_1fa > 0 else 0

    # Print results
    print("=" * 80)
    print("Phase 1a: Edge Stream GDN Validation Results")
    print("=" * 80)

    print(f"\n📊 Baseline (BiMamba2 edge):")
    print(f"  val_loss: {baseline_loss:.4f}")
    print(f"  sensitivity@10FA: {baseline_sens_10fa:.2%}")
    print(f"  sensitivity@5FA:  {baseline_sens_5fa:.2%}")
    print(f"  sensitivity@1FA:  {baseline_sens_1fa:.2%}")

    print(f"\n📊 Phase 1a (GDN edge):")
    print(f"  val_loss: {phase1a_loss:.4f} ({loss_delta:+.2%})")
    print(f"  sensitivity@10FA: {phase1a_sens_10fa:.2%} ({sens_10fa_gain:+.2%})")
    print(f"  sensitivity@5FA:  {phase1a_sens_5fa:.2%} ({sens_5fa_gain:+.2%})")
    print(f"  sensitivity@1FA:  {phase1a_sens_1fa:.2%} ({sens_1fa_gain:+.2%})")

    # Decision
    print("\n" + "=" * 80)
    print("📋 Go/No-Go Decision")
    print("=" * 80)

    if sens_10fa_gain >= 0.01:  # +1% or better
        print(f"\n✅ GO → Proceed to Phase 1b (Node Stream Validation)")
        print(f"   Reason: Phase 1a shows {sens_10fa_gain:+.2%} improvement")
        print(f"   Edge stream GDN validated successfully")
    elif sens_10fa_gain >= -0.005:  # No regression > -0.5%
        print(f"\n⚠️ MARGINAL → Consider proceeding with caution")
        print(f"   Reason: Phase 1a shows {sens_10fa_gain:+.2%} (neutral)")
        print(f"   No clear improvement, but no major regression")
    else:
        print(f"\n❌ NO-GO → Revert to BiMamba2 (baseline)")
        print(f"   Reason: Phase 1a shows {sens_10fa_gain:+.2%} (regression)")
        print(f"   Edge stream GDN does not improve performance")

    print("=" * 80)


if __name__ == '__main__':
    main()
```

**Run analysis**:

```bash
python scripts/analyze_phase1a.py --project seizure-v3-fla-validation
```

---

## 5. Rollback Procedure

**If Phase 1a underperforms or causes issues**, rollback is INSTANT via config change (no git operations needed).

### 5.1. Instant Rollback

**Edit config to revert edge stream to BiMamba2**:

```yaml
# Edit configs/local/phase1a_edge_gdn.yaml:
model:
  mamba:
    temporal_type: bimamba2              # Keep as is
    temporal_type_edge: bimamba2         # CHANGE: revert to BiMamba2
    # temporal_type_edge: gated_deltanet  # Comment out or remove
```

**Re-run training**:

```bash
# Restart with BiMamba2 edge stream
export BGB_LIMIT_FILES=50
python -m src train configs/local/phase1a_edge_gdn.yaml

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
# Revert to pre-Phase-1a state (NOT RECOMMENDED - use config rollback instead)
git checkout v3.8.3-pre-phase1a

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
- [ ] ✅ Isolation verified (node=BiMamba2, edge=GDN)
- [ ] ✅ Parameter count ~7.3K edge (29% reduction expected)
- [ ] ✅ Convergence over 10 epochs (loss decreases)

### 6.2. Performance Criteria (Go/No-Go)

**GO → Phase 1b** if:
- [ ] ✅ sensitivity@10FA ≥ baseline + 1% **OR** no regression > 0.5%
- [ ] ✅ No major regressions (loss ≤ baseline + 5%)
- [ ] ✅ Throughput ≤ 10% slower than baseline
- [ ] ✅ Memory usage ≤ baseline + 2GB

**NO-GO → Revert** if:
- [ ] ❌ sensitivity@10FA regression > 1%
- [ ] ❌ Training unstable (NaNs, divergence)
- [ ] ❌ Throughput regression > 20%

---

## 7. Next Steps

### If Phase 1a Succeeds:

1. **Document results** in `PHASE1A_RESULTS.md`
2. **Proceed to Doc 2**: Node Stream Validation (Phase 1b)
3. **Continue phased validation**: Doc 3 (Full), Doc 4 (Hybrid)

### If Phase 1a Fails:

1. **Document failure** in `PHASE1A_POSTMORTEM.md`
2. **Root cause analysis**: Why did it fail?
   - Check logs for NaNs, gradient explosions
   - Verify Phase 0 infrastructure correctly implemented
   - Check for hyperparameter mismatches
3. **Iterate or pivot**:
   - Try different fusion_mode (concat instead of sum)
   - Adjust learning rate / gradient clip
   - Consider alternative architectures (GLA, pure DeltaNet)
4. **Revert to baseline**: Use config rollback (Section 5)

---

## 8. Timeline & Checklist

### Day 1: Setup
- [ ] Verify Phase 0 complete (Section 1.1)
- [ ] Create phase1a config (Section 2)
- [ ] Optional: Run baseline if needed (Section 1.2)
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
- [ ] If GO: Proceed to Phase 1b
- [ ] If NO-GO: Rollback (Section 5)

**Total Timeline**: 2-3 days

---

## 9. References

- **Doc 0 (SSOT)**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md)
  - Section 14: Phase 0 Infrastructure (wrapper, builders, schema, tests)
  - Section 15: Coexistence Strategy
- **Doc 2 (Next)**: Node Stream Validation (Phase 1b) - pending Phase 1a success
- **FLA Library**: https://github.com/fla-org/flash-linear-attention
- **Gated DeltaNet Paper**: https://arxiv.org/abs/2412.06464

---

**Document Status**: ✅ Ready for Implementation (AFTER Phase 0 complete)
**Dependencies**: Doc 0 Section 14 (Phase 0 Infrastructure) must be complete
**Next Document**: Doc 2 (Node Stream Validation) - pending Phase 1a success
