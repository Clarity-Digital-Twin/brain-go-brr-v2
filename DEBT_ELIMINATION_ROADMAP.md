# Comprehensive Debt Elimination Roadmap

**Date:** 2025-10-04
**Status:** Post config-wiring completion - documentation and implementation cleanup phase
**Goal:** Zero technical debt, zero documentation divergence, production-ready codebase

---

## ✅ Phase 1: Config Wiring (COMPLETE)

All configuration fields now have runtime implementation paths:
- ✅ P0: `log_every_n_steps`, `save_predictions`, `save_plots`
- ✅ P1: `normalize`, `montage`, `max_samples`, `max_hours`, `log_gradients`, `log_weights`, `log_level`
- ✅ P2: `stitching.method`, `morphology.use_gpu`
- ✅ P3: Phantom features removed (`use_mne`, `evaluation.metrics`, `residual_scale_*`)

**Result**: 100% config-to-runtime alignment.

---

## 🔥 Phase 2: Documentation Debt (CRITICAL - NEXT)

### Priority 0: Schema Documentation (BLOCKS USERS)

**BD-01: Config Schema Docs vs Reality**
- **Issue**: `docs/03-configuration/config-schema.md:67-68` advertises `optimizer: adamw|adam|sgd`, `scheduler: cosine|linear|constant`, and `resources` block
- **Reality**: Only `adamw` and `cosine` exist, no `resources` field
- **Impact**: Users copy non-existent fields → validation errors
- **Fix**:
  ```markdown
  # In docs/03-configuration/config-schema.md:67-68
  - optimizer: adamw (only AdamW currently supported)
  - scheduler.type: cosine (only cosine currently supported)
  # Remove line 81 (resources block)
  ```
- **Effort**: 5 minutes

**BD-02: Modal Deployment Guide Stale Config**
- **Issue**: `docs/05-training/modal-deployment.md:88-114` shows `batch_size: 64`, `learning_rate: 3e-5`, `resources` block
- **Reality**: `configs/modal/train.yaml` uses `batch_size: 48`, `learning_rate: 8.0e-5`, no `resources`
- **Impact**: Wrong hyperparameters → slower training or OOM
- **Fix**: Replace embedded YAML snippet with actual config (lines 88-114)
  ```yaml
  training:
    batch_size: 48                   # Optimized for A100-80GB
    learning_rate: 8.0e-5            # Batch-size scaled
    gradient_clip: 0.5               # NaN protection
    mixed_precision: true            # A100 tensor cores
    mid_checkpoint_interval_s: 1800  # Every 30 min
    mid_epoch_keep: 3                # Keep last 3
  ```
- **Effort**: 10 minutes

### Priority 1: Training Workflow Docs (CONFUSES USERS)

**BD-06: Local Training Guide Stale Hyperparameters**
- **Issue**: `docs/05-training/local.md:22` says `batch_size: 4`
- **Reality**: `configs/local/train.yaml:128` uses `batch_size: 8`
- **Impact**: Users run at half speed
- **Fix**: Update line 22 to `batch_size: 8` with comment about 2x speedup
- **Effort**: 2 minutes

**BD-07: Checkpoint Docs Push Env Vars Over Config**
- **Issue**: Multiple docs (`checkpoint-strategy.md:13-17`, `resume.md:14-16`, `monitoring.md:26-28`, `local.md:60`) treat `BGB_MID_EPOCH_*` as primary
- **Reality**: Config fields `mid_checkpoint_interval_s` / `mid_epoch_keep` are primary, env vars are deprecated fallback
- **Impact**: Users use deprecated workflow, miss config-driven features
- **Fix**: In each file, add:
  ```markdown
  **Recommended (v3.6+)**: Set in config:
  ```yaml
  training:
    mid_checkpoint_interval_s: 1800  # Every 30 minutes
    mid_epoch_keep: 3                # Keep last 3
  ```

  **Legacy (deprecated)**: Environment variables (backward compatible):
  ```bash
  export BGB_MID_EPOCH_MINUTES=30
  export BGB_MID_EPOCH_KEEP=3
  ```
  ```
- **Effort**: 20 minutes (4 files)

**BD-05: Performance Env Vars Scope**
- **Issue**: `docs/03-configuration/env-vars.md` lists `BGB_PERF_*` as general-purpose
- **Reality**: Only used in `tests/performance/`, not training
- **Impact**: Users set flags expecting training behavior changes
- **Fix**: Add scope note:
  ```markdown
  **Scope**: Performance test suite only (not training runtime)
  ```
- **Effort**: 5 minutes

---

## 🧹 Phase 3: Code Cleanup (REMOVE DEAD CODE)

### Priority 2: Phantom Features (NO RUNTIME IMPACT)

**BD-03: BGB_NAN_DEBUG_MAX Implementation**
- **Issue**: Env var defined, exposed via `env.nan_debug_max()`, but NEVER used
- **Reality**: No runtime code calls `env.nan_debug_max()` or checks `BGB_NAN_DEBUG_MAX`
- **Impact**: Users set flag thinking it throttles NaN warnings, nothing happens
- **Options**:
  1. **Implement**: Add counter in `train/train_step.py` to limit NaN warnings
  2. **Remove**: Delete env var, remove from docs
- **Recommendation**: REMOVE (simpler, no proven need for throttling)
- **Fix**:
  ```python
  # In src/brain_brr/utils/env.py:47 - DELETE
  _NAN_DEBUG_MAX = int(os.getenv("BGB_NAN_DEBUG_MAX", "10"))

  # In src/brain_brr/utils/env.py:138-140 - DELETE
  @staticmethod
  def nan_debug_max() -> int:
      """Maximum NaN occurrences before stopping (default: 10)."""
      return _NAN_DEBUG_MAX
  ```
- **Effort**: 5 minutes + docs cleanup

---

## 📊 Phase 4: Architecture Decisions (FUTURE CONSIDERATION)

### Optional: Missing Features in Schema

**Optimizer/Scheduler Expansion**
- **Current**: Only `adamw` and `cosine` supported
- **Schema comment**: "adam/sgd coming v3.7", "linear/constant coming v3.7"
- **Decision needed**:
  1. Implement adam/sgd/linear/constant (effort: ~2 hours)
  2. Remove "coming v3.7" comments, accept adamw/cosine-only (effort: 2 minutes)
- **Recommendation**: Accept current state, remove future promises from comments
  - AdamW is best-in-class for transformers
  - Cosine schedule is proven for long training
  - No user requests for alternatives

**Resources Block**
- **Current**: Docs mention, schema doesn't have it
- **Purpose**: Would control distributed training, memory limits
- **Decision needed**:
  1. Implement `resources: { max_memory_gb, distributed }` (effort: ~4 hours)
  2. Remove from docs entirely (effort: 2 minutes)
- **Recommendation**: Remove from docs
  - Single-GPU training works perfectly
  - Distributed training not on roadmap
  - Memory managed via batch_size

---

## 🎯 Execution Plan (Time-Boxed)

### Week 1: Critical Documentation (1 hour total)
1. ✅ BD-01: Fix schema docs (5 min)
2. ✅ BD-02: Fix Modal deployment guide (10 min)
3. ✅ BD-06: Fix local training batch_size (2 min)
4. ✅ BD-07: Update checkpoint docs (20 min)
5. ✅ BD-05: Clarify perf env vars scope (5 min)
6. ✅ BD-03: Remove BGB_NAN_DEBUG_MAX (5 min + docs)
7. ✅ Clean schema comments about future features (10 min)

### Week 2: Validation & Smoke Tests
1. Run full test suite: `make test`
2. Smoke test local: `make s`
3. Smoke test Modal: `modal run deploy/modal/app.py --action train --config configs/modal/smoke.yaml`
4. Verify all docs match reality

### Week 3: Final Polish
1. Archive old docs if needed
2. Update CHANGELOG.md
3. Update CLAUDE.md with debt-free status

---

## 📋 Checklist Template (Copy for Each Item)

For each debt item, verify:
- [ ] Issue confirmed in codebase
- [ ] Fix implemented
- [ ] Tests pass (`make q && make test`)
- [ ] Documentation updated
- [ ] CHANGELOG entry added
- [ ] Smoke test validates change

---

## 🎉 Success Criteria

**Definition of "Debt-Free"**:
1. ✅ All config fields have runtime paths (CONFIG_ADDITIONAL_GAPS.md empty)
2. ✅ All docs match actual code behavior (no stale examples)
3. ✅ No phantom env vars (defined but unused)
4. ✅ No misleading schema comments (promises of unimplemented features)
5. ✅ All smoke tests pass (local + Modal)
6. ✅ Documentation review shows zero divergence

**Exit Criteria**:
- Documentation audit finds ZERO mismatches
- Users can copy ANY config snippet and it works
- No env vars exist without runtime usage
- Schema comments accurately reflect implementation

---

## 🚀 Post-Cleanup Benefits

1. **User Trust**: Every doc example works perfectly
2. **Developer Velocity**: No time wasted debugging stale docs
3. **Maintenance**: Single source of truth for all config
4. **Production Ready**: Zero surprises in deployment
5. **Professional Quality**: No half-implemented features

---

**Next Action**: Execute Week 1 tasks (1 hour investment for permanent debt elimination)
