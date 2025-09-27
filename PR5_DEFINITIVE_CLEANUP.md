# PR‑5: Definitive Clamp Cleanup & Stable Defaults (CORRECTED)

Status: **REVISED** - Original PR-5 had significant errors

Owner: Core V3 architecture team

Scope: V3 dual‑stream stack (TCN + BiMamba + GNN + LapPE)

As‑of: current HEAD on `fix/architectural-stability` branch

## ⚠️ CRITICAL CORRECTIONS FROM ORIGINAL PR-5

1. **Line numbers for detector.py were ALL WRONG** (off by ~8 lines)
2. **Missing intervention**: edge_features.py:77 clamp not mentioned
3. **Incorrect counts**: 27 clamps not 28, 9 nan_to_num in models not 15
4. **Already conditional**: Some interventions are already behind `_env.safe_clamp()` checks

## Executive Summary

PR‑5 finalizes stability work by removing non‑essential clamps and nan_to_num calls made redundant by PR‑1/2/3/4. We keep only mathematically required guards (cosine/division), input sanity, pre‑loss/output clamps, and PE guards. The result is a cleaner, faster, and more stable V3.

## CORRECTED Intervention Counts

**Actual counts at HEAD:**
- `torch.clamp`: 23 in models + 4 in training loop = **27 total** (not 28)
- `torch.nan_to_num`: 9 in models + 3 in training = **12 total** (not 15)

## What Stays vs What Goes (CORRECTED LINE NUMBERS)

### ✅ Keep (Essential Guards)

**Input/Output Safety:**
- `tcn.py:241` - Input sanity clamp (-10, 10)
- `detector.py:385,386` - Decoder output clamps before loss
- `detector.py:392,393` - Final logit clamps (-100, 100)

**Mathematical Bounds:**
- `edge_features.py:73` - Division safety (norms min=1e-6)
- `edge_features.py:81,91` - Cosine similarity bounds (-1, 1)
- `edge_features.py:87` - Additional division safety

**Laplacian PE Guards:**
- `gnn_pyg.py:229` - Eigenvalue clamp (1e-6, 2.0)
- `gnn_pyg.py:249` - PE nan_to_num fallback

**Loss-Level Guards:**
- `loop.py:205` - Logit clamp for BCE
- `loop.py:212` - Probability clamp (avoid log(0))
- `loop.py:218` - p_t stability clamp
- `loop.py:223` - Loss explosion prevention

### ❌ Remove (Redundant After PR-1/2/3/4)

**TCN Internal Clamps (replaced by PR-1 norms):**
- `tcn.py:238` - Input nan_to_num
- `tcn.py:248,255,262` - Internal tier clamps

**Detector Clamps (replaced by PR-1/2):**
- `detector.py:233,234` - Feature safe clamps (already conditional!)
- `detector.py:281` - Edge feature clamp (redundant with edge_features.py)
- `detector.py:299` - Edge projection clamp (replaced by PR-2 tanh)
- `detector.py:377,378` - Temporal safe clamps (already conditional!)

**GNN Clamps (replaced by PR-3):**
- `gnn_pyg.py:358,359` - Batch safe clamps (already conditional!)

**Mamba Internal (monitor first, then remove):**
- `mamba.py:177,180` - Input clamps/nan_to_num
- `mamba.py:249,259` - Output clamps
- `mamba.py:328,329,339,342` - Additional safety clamps

### ⚠️ SPECIAL CASE: edge_features.py:77

**NOT MENTIONED IN ORIGINAL PR-5!**
- `edge_features.py:77` - x_norm clamp after normalization
- **Decision**: KEEP for now (provides safety after division)

## Stable Configuration Defaults

```yaml
model:
  # PR-1: Boundary normalization
  norms:
    boundary_norm: layernorm
    boundary_eps: 1.0e-5
    layerscale_alpha: 0.1
    after_tcn_proj: true
    after_node_mamba: true
    after_edge_mamba: true
    after_gnn: true
    before_decoder: true

  graph:
    # PR-2: Bounded edge stream
    edge_lift_activation: tanh
    edge_lift_norm: layernorm
    edge_lift_init_gain: 0.1

    # PR-3: Adjacency conditioning
    adj_row_softmax: true
    adj_softmax_tau: 1.0
    adj_ema_beta: 0.9
    adj_force_symmetric: true
    laplacian_eps: 1.0e-3
    laplacian_normalize: true

  # PR-4: Fusion & monitoring
  fusion:
    fusion_type: gated
    fusion_heads: 4
    fusion_dropout: 0.1

  clamp_retirement:
    remove_intermediate_clamps: true  # After monitoring confirms safety
    remove_nan_to_num: true
    keep_input_clamp: true
    keep_output_clamp: true
    keep_loss_clamps: true
    log_clamp_hits: false  # Enable for monitoring
    validate_finite: true
```

## Implementation Checklist (CORRECTED)

### Phase 1: Remove Redundant Interventions

**TCN (tcn.py):**
- [ ] Remove line 238 (nan_to_num)
- [ ] Remove lines 248, 255, 262 (internal clamps)
- [ ] KEEP line 241 (input sanity)

**Detector (detector.py):**
- [ ] Remove lines 233-234 (conditional safe clamps)
- [ ] Remove line 281 (edge_feats clamp - redundant)
- [ ] Remove line 299 (edge_in clamp - PR-2 handles)
- [ ] Remove lines 377-378 (conditional temporal clamps)
- [ ] KEEP lines 385-386, 392-393 (output/loss guards)

**GNN (gnn_pyg.py):**
- [ ] Remove lines 358-359 (conditional batch clamps)
- [ ] KEEP lines 229, 249 (PE guards)

**Mamba (mamba.py) - STAGED REMOVAL:**
- [ ] Stage 1: Remove lines 177, 328 (nan_to_num)
- [ ] Stage 2: Remove lines 180, 329 (input clamps)
- [ ] Stage 3: Remove lines 249, 259, 339, 342 (if monitoring clean)

### Phase 2: Update Configs
- [ ] Set PR-1/2/3/4 defaults in `configs/local/train.yaml`
- [ ] Update `configs/local/smoke.yaml` to match
- [ ] Set `clamp_retirement.remove_intermediate_clamps: true`

### Phase 3: Validation
- [ ] Run 10k batches with monitoring enabled
- [ ] Verify clamp hit rate = 0 at removed sites
- [ ] Check gradient norms remain stable
- [ ] Confirm no NaN/Inf in forward or backward

## Rollback Plan

If instability appears:
1. Set `clamp_retirement.log_clamp_hits: true`
2. Identify which removed clamp would have triggered
3. Temporarily restore that specific clamp
4. Investigate root cause (likely needs norm eps adjustment)

## Key Differences from Original PR-5

1. **Line number accuracy**: All detector.py lines corrected
2. **Complete inventory**: Added missing edge_features.py:77
3. **Conditional awareness**: Noted which clamps are already conditional
4. **Staged Mamba removal**: More conservative approach
5. **Accurate counts**: 27 clamps not 28, 12 nan_to_num not 15

## Risk Assessment

**Low Risk Removals** (already conditional or redundant):
- detector.py:233-234, 377-378 (already behind safe_clamp)
- gnn_pyg.py:358-359 (already behind safe_clamp)
- detector.py:281 (redundant with edge_features)

**Medium Risk Removals** (monitor carefully):
- tcn.py:248,255,262 (internal tier clamps)
- detector.py:299 (edge_in clamp)

**Higher Risk Removals** (stage carefully):
- All Mamba clamps (complex state-space dynamics)

## Success Metrics

- [ ] Zero NaN/Inf for 10,000+ consecutive batches
- [ ] Clamp monitoring shows 0 hits at removed sites
- [ ] TAES metrics unchanged or improved
- [ ] Latency improvement 5-10% from removed checks
- [ ] Clean mypy/ruff with all changes

## Final Assessment

**PR-5 concept is SOUND but original document had ERRORS:**
- ✅ Strategy of removing redundant clamps is correct
- ✅ PR-1/2/3/4 do provide the safety needed
- ❌ Original line numbers were wrong
- ❌ Original counts were inaccurate
- ⚠️ Some removals need more careful staging

**Recommendation**: Use this CORRECTED version for implementation.

---

**Note**: This corrected version is based on actual code inspection at HEAD of `fix/architectural-stability` branch. All line numbers have been verified.