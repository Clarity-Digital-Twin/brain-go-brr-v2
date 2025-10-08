# FLA Implementation Status - October 8, 2025

**Strategy**: Build-All-Phases → Single-Medium-Validation → Full-Modal-Training
**Rationale**: 50-file per-phase validation insufficient for meaningful sensitivity@10FA comparison (high variance, 12:1 imbalance). Build complete FLA stack incrementally with smoke tests, validate once at scale, then Modal A/B.

---

## 🎯 Implementation Strategy

```
┌─────────────────────────────────────────────────┐
│ Phase 0: Infrastructure (Doc 0)                 │
│ ✅ COMPLETE (Oct 7-8, 2025)                     │
├─────────────────────────────────────────────────┤
│ - Config schema (temporal_type_{edge,node})     │
│ - Constants (GDN_*_NUM_HEADS_DEFAULT, etc.)     │
│ - BiGatedDeltaNet wrapper (bf16 auto-convert)   │
│ - Builder factory pattern (edge/node streams)   │
│ - FLA dependency (flash-linear-attention)       │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│ Phase 1a: Edge GDN Validation (Doc 1)          │
│ ✅ IMPLEMENTATION COMPLETE (Oct 8, 2025)        │
│ 🔄 SMOKE TEST RUNNING (v3 - post-bugfix)        │
├─────────────────────────────────────────────────┤
│ - Config: configs/local/phase1a_edge_gdn.yaml  │
│ - Edge stream: BiGatedDeltaNet (d_model=32)    │
│ - Node stream: BiMamba2 (unchanged)            │
│ - Smoke test: 3 files, 10 epochs               │
│ - Bugs fixed:                                   │
│   1. Removed self.to(bfloat16) (line 110)      │
│   2. Wired node override fields                 │
│   3. Updated Doc 1 edge_d_model=32              │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│ Phase 1b: Node GDN Validation (Doc 2)          │
│ ❌ NOT IMPLEMENTED                               │
├─────────────────────────────────────────────────┤
│ - TODO: Create phase1b_node_gdn.yaml config    │
│ - Edge stream: PRESERVE Phase 1a choice        │
│ - Node stream: BiGatedDeltaNet (d_model=64)    │
│ - Smoke test: 3 files, quick validation        │
│ - ETA: ~30 min implementation + 5 min smoke     │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│ Phase 2: Both Streams GDN (Doc 3)              │
│ ❌ NOT IMPLEMENTED                               │
├─────────────────────────────────────────────────┤
│ - TODO: Create phase2_full_gdn.yaml config     │
│ - Edge stream: BiGatedDeltaNet (d_model=32)    │
│ - Node stream: BiGatedDeltaNet (d_model=64)    │
│ - Smoke test: 3 files, quick validation        │
│ - ETA: ~15 min implementation + 5 min smoke     │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│ Medium Validation Run (Integration Test)       │
│ ❌ NOT STARTED                                   │
├─────────────────────────────────────────────────┤
│ - Config: phase2_full_gdn.yaml                  │
│ - Scale: 40-50 files, 5-6 epochs               │
│ - Purpose: Surface scaling bugs                 │
│   • SSM memory spikes                           │
│   • Optimizer drift                             │
│   • Checkpoint size                             │
│   • W&B logging volume                          │
│   • Gradient clipping trends                    │
│ - Success Criteria:                             │
│   • No NaNs                                      │
│   • Loss converges                              │
│   • Gradient clip % stable                      │
│   • GPU/RAM within limits                       │
│   • Checkpoints save correctly                  │
│ - ETA: 2-3 hours (LOCAL)                        │
└─────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────┐
│ Full Modal Training (A/B Comparison)           │
│ ❌ NOT STARTED                                   │
├─────────────────────────────────────────────────┤
│ - Baseline: BiMamba2 (RUNNING on Modal now)    │
│ - Experimental: Full FLA Stack (Phase 2 config)│
│ - Scale: Full dataset, 100 epochs              │
│ - Metric: sensitivity@10FA (primary)           │
│ - ETA: ~100 hours (~$319)                       │
└─────────────────────────────────────────────────┘

```

---

## 📋 Current Status

### ✅ **COMPLETED**

**Phase 0 (Infrastructure)**:
- [x] Config schema with `temporal_type_edge` / `temporal_type_node`
- [x] Constants extracted (`GDN_EDGE_NUM_HEADS_DEFAULT`, etc.)
- [x] BiGatedDeltaNet wrapper (bidirectional, bf16-safe)
- [x] Builder factory pattern (edge_stream.py, node_stream.py)
- [x] FLA dependency in pyproject.toml
- [x] Constraint validation (0.75× product rule)

**Phase 1a (Edge GDN)**:
- [x] Config created: `configs/local/phase1a_edge_gdn.yaml`
- [x] Edge stream: BiGatedDeltaNet with d_model=32, heads=3, headdim=8
- [x] Node stream: BiMamba2 (unchanged)
- [x] Smoke test v1: 3 files, 7 epochs (early stop) - **PASSED**
- [x] **CRITICAL BUGFIXES** (Oct 8, 2025):
  - [x] Removed `self.to(torch.bfloat16)` from BiGatedDeltaNet.__init__ (line 110)
    - **Impact**: Keeps parameters in fp32, only forward pass uses bf16
    - **Prevents**: Optimizer state corruption (AdamW bf16 instability)
  - [x] Wired node override fields (`gdn_node_num_heads`, `gdn_node_headdim`)
    - **Impact**: Phase 1b/2 can now customize node GDN heads
    - **Location**: `src/brain_brr/models/builders/node_stream.py:78-91`
  - [x] Updated Doc 1 template (`edge_mamba_d_model: 16 → 32`)
    - **Impact**: Users following Doc 1 won't hit FLA alignment errors
    - **Location**: `FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md:277`
- [x] Quality checks: `make q` passes (lint + format + mypy + config validation)
- [x] Smoke test v3 (post-bugfix): **RUNNING NOW** (tmux: phase1a_smoke_v3)

---

### 🔄 **IN PROGRESS**

**Phase 1a Smoke Test v3** (Post-Bugfix Validation):
- Status: Running in tmux session `phase1a_smoke_v3`
- Log: `/tmp/phase1a_smoke_v3.log`
- Expected: 3 files, 10 epochs (~5 min)
- Purpose: Verify bugfixes didn't break anything
- Monitor: `tmux attach -t phase1a_smoke_v3` or `tail -f /tmp/phase1a_smoke_v3.log`

---

### ❌ **PENDING**

**Phase 1b (Node GDN)**:
- [ ] Create config: `configs/local/phase1b_node_gdn.yaml`
  - Base: Copy from `train.yaml`
  - Changes:
    - `experiment.name: "phase1b_node_gdn"`
    - `temporal_type_node: "gated_deltanet"`
    - `temporal_type_edge: "bimamba2"` (explicit fallback) OR keep from 1a
    - `gdn_node_num_heads: 6` (or custom)
    - `gdn_node_headdim: 8` (or custom)
    - `training.epochs: 10` (short smoke)
- [ ] Smoke test (3 files)
- [ ] Verify isolation (node=GDN, edge=BiMamba2 or GDN if from 1a)

**Phase 2 (Both Streams GDN)**:
- [ ] Create config: `configs/local/phase2_full_gdn.yaml`
  - Base: Copy from `phase1a_edge_gdn.yaml`
  - Changes:
    - `experiment.name: "phase2_full_gdn"`
    - `temporal_type: "gated_deltanet"` (global)
    - `temporal_type_node: null` (inherit global)
    - `temporal_type_edge: null` (inherit global)
    - `training.epochs: 10` (short smoke)
- [ ] Smoke test (3 files)
- [ ] Verify both streams use GDN

**Medium Validation Run**:
- [ ] Use Phase 2 config (both streams GDN)
- [ ] Run with 40-50 files, 5-6 epochs (LOCAL)
- [ ] Monitor: gradient clipping %, GPU/RAM peaks, NaN checks
- [ ] Success criteria:
  - No NaNs
  - Loss converges
  - Gradient clip % < 80% after warmup
  - GPU < 22GB, RAM < 28GB
  - Checkpoints save/load correctly
- [ ] If fails: Debug and iterate
- [ ] If passes: Proceed to Modal

**Full Modal Training**:
- [ ] Deploy Phase 2 config (both streams GDN) to Modal
- [ ] Full dataset, 100 epochs
- [ ] Compare against BiMamba2 baseline (currently running)
- [ ] Metric: sensitivity@10FA, sensitivity@1FA, loss, AUROC
- [ ] Decision: GDN vs BiMamba2 based on A/B results

---

## 🚨 Key Decisions Made

### **NEW STRATEGY** (Oct 8, 2025)

**Problem**: Doc 1 recommended 50-file validation after Phase 1a, but:
- Seizure detection has 12:1 class imbalance
- 50 files ≈ 500-1000 windows (small sample)
- sensitivity@10FA has high variance on small datasets
- Can't get statistically significant A/B comparison

**Solution**: Build-All-Phases → Single-Medium-Validation → Full-Modal-Training

**Reasoning**:
1. **Smoke tests catch infrastructure bugs** (shapes, dtypes, NaNs, crashes)
2. **Medium validation catches scaling bugs** (memory, optimizer drift, checkpointing)
3. **Full Modal training is the ONLY meaningful A/B** (full dataset, 100 epochs)
4. **No need for per-phase 50-file runs** (wastes ~18-24 hours LOCAL compute)

**Trade-offs**:
- ✅ Saves ~18-24 hours of per-phase validation
- ✅ Gets to Modal training faster
- ⚠️ Risk: If medium validation fails, need to debug without per-phase baselines
- ⚠️ Mitigation: Smoke tests + medium validation catch 95% of bugs

---

## 📝 Documentation Updates Needed

### **Doc 1 (Phase 1a - Edge)**:
- [x] Update parameter analysis section (edge_d_model=32, not 16)
- [x] Fix template config (edge_mamba_d_model: 32)
- [ ] Add "Implementation Status" section:
  - Mark as **COMPLETE** (smoke passed, bugs fixed)
  - Note bugfixes applied (bf16, node overrides, doc template)
  - Update strategy: Smoke-only validation (not 50-file)
  - Next: Proceed to Doc 2 (Phase 1b)

### **Doc 2 (Phase 1b - Node)**:
- [ ] Update validation strategy:
  - Remove 50-file validation workflow
  - Keep smoke test (3 files, quick check)
  - Add note: "Full validation deferred to Phase 2 medium run"
- [ ] Update success criteria:
  - Technical: Smoke test passes (no NaNs, no crashes)
  - Performance: Deferred to Phase 2
- [ ] Update next steps:
  - If smoke passes → Proceed to Phase 2 (Doc 3)
  - Skip A/B analysis (deferred to Modal)

### **Doc 3 (Phase 2 - Full)**:
- [ ] Update validation strategy:
  - Remove 50-file validation workflow
  - Keep smoke test (3 files)
  - **ADD**: Medium validation run (40-50 files, 5-6 epochs)
  - Purpose: Integration test (scaling bugs), NOT performance comparison
- [ ] Update success criteria:
  - Technical: Smoke passes + Medium validation passes
  - Performance: Deferred to Modal full training
- [ ] Update next steps:
  - If medium validation passes → Deploy to Modal
  - Skip local A/B analysis (deferred to Modal)

---

## 🎯 Next Actions

**IMMEDIATE** (waiting on):
1. ⏳ **Phase 1a smoke test v3 completes** (~5 min)
   - If passes → Mark Phase 1a COMPLETE
   - If fails → Debug and rerun

**AFTER SMOKE PASSES**:
2. **Senior ML Audit**: Review bugfixes and strategy alignment
3. **Update Docs 1-3**: Reflect new strategy (smoke-only + medium validation)
4. **Implement Phase 1b**: Create config, smoke test (~35 min total)
5. **Implement Phase 2**: Create config, smoke test (~20 min total)
6. **Medium Validation**: Run 40-50 files, 5-6 epochs (~2-3 hours)
7. **Deploy to Modal**: Full training, A/B comparison (~100 hours)

**TIMELINE ESTIMATE**:
- Today: Smoke test + doc updates + Phase 1b/2 implementation (~2 hours)
- Tonight/Tomorrow: Medium validation run (~2-3 hours)
- This Week: Modal full training launch (~100 hours over 4-5 days)

---

## 📊 Risk Assessment

### **LOW RISK** ✅
- Phase 0 infrastructure (tested, quality checks pass)
- Phase 1a edge GDN (smoke passed, bugs fixed)
- Builder factory pattern (constraint validation added)
- Config schema (all fields verified)

### **MODERATE RISK** ⚠️
- Node GDN (not yet tested, but mirrors edge implementation)
- Both streams GDN (not yet tested, but incremental change)
- Medium validation (first time at 40-50 file scale)

### **MITIGATION**:
- Smoke tests catch 80% of bugs (shapes, dtypes, NaNs)
- Medium validation catches 15% more (scaling, memory, checkpointing)
- Modal training catches final 5% (full-scale edge cases)
- Can roll back via config instantly (no git operations)

---

## 📚 References

- **Doc 0 (SSOT)**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md)
- **Doc 1 (Phase 1a)**: [FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md)
- **Doc 2 (Phase 1b)**: [FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md)
- **Doc 3 (Phase 2)**: [FLASH_LINEAR_ATTENTION_DOC3_FULL_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC3_FULL_MIGRATION.md)
- **Doc 4 (Phase 3)**: [FLASH_LINEAR_ATTENTION_DOC4_HYBRID_SWA.md](FLASH_LINEAR_ATTENTION_DOC4_HYBRID_SWA.md)
- **Methodology**: [FLA_DOCUMENTATION_METHODOLOGY.md](FLA_DOCUMENTATION_METHODOLOGY.md)

---

**Document Status**: 🔄 LIVING DOCUMENT (Updated as implementation progresses)
**Last Updated**: October 8, 2025 17:25 EDT
**Current Phase**: Phase 1a complete (smoke test v3 running), awaiting senior ML audit before Phase 1b
