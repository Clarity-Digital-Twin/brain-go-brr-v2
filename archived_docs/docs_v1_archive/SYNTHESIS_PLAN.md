# Documentation Synthesis Plan - October 1, 2025

**Strategy**: Consolidate `RECENT-WORK-COMBINED` into clean, accurate `RECENT-WORK-SYNTHESIZED` docs.

**Goal**: 100% accurate, non-redundant docs that match v3.4.1 codebase.

---

## 📋 AUDIT OF RECENT-WORK-COMBINED

### Category 1: NaN Prevention & Gradient Stability (CONSOLIDATE)

**Overlapping docs** (need synthesis):

1. **10-major-NAN-refactor/**
   - `NAN_CANONICAL.md` (canonical reference)
   - `GRADIENT_BEHAVIOR_GUIDE.md` (gradient expectations)
   - `ARCHITECTURAL_INSTABILITY_FIX.md` (eigendecomposition fix)
   - `PR1_BOUNDARY_NORMALIZATION_PLAN.md` + `PR1_FINAL_STATUS.md` + `PR1_IMPLEMENTATION_SUMMARY.md`
   - `PR2_BOUNDED_EDGE_STREAM_PLAN.md` + `PR2_FINAL_STATUS.md`
   - `PR3_ADJACENCY_CONDITIONING_PLAN.md`
   - `PR4_CLAMP_RETIREMENT_PLAN.md`
   - `PR5_DEFINITIVE_CLEANUP.md`

2. **archive/**
   - `NAN-PROTECTION-REFERENCE.md` (3-tier protection)
   - `NAN-REGRESSION-POST-DEPENDENCY-UPGRADE.md` (incident)

3. **archive-2/**
   - `ARCHITECTURAL_STABILITY_INVESTIGATION.md` (Sept 30 investigation)
   - `GRADIENT_STABILITY_ANALYSIS_OCT1.md` (batch 59 analysis)
   - `DOCS_STATUS_OCTOBER_2025.md` (master reality check)

**Action**: **SYNTHESIZE** into 2-3 comprehensive docs:
- `NAN_PREVENTION_COMPLETE.md` (all NaN strategies + 3-tier protection)
- `GRADIENT_MONITORING_GUIDE.md` (realistic expectations + when to intervene)
- `ARCHITECTURE_V3_STABILITY.md` (eigendecomposition fix + PR1-5 summary)

---

### Category 2: Warmup Schedules (CONSOLIDATE)

**Overlapping docs**:
- `archive-2/WARMUP_IMPLEMENTATION_ANALYSIS.md` (complete implementation)
- `archive-2/WARMUP_SCHEDULES_IMPLEMENTATION.md` (redundant)

**Action**: **SYNTHESIZE** into:
- `WARMUP_SCHEDULES_GUIDE.md` (config + usage + results)

---

### Category 3: Incident Reports (KEEP AS-IS)

**Specific incidents** (historical value, keep separate):
- `archive/PR708_APPLICATION.md` (Mamba int32 kernel fix)
- `archive/MODAL_XID31_RECURRENCE.md` (Triton cache issue)
- `archive/MODAL_A100_CUDA_FAILURE_ANALYSIS.md` (CUDA failures)
- `archive/PYTORCH_2.5_UPGRADE_INCIDENT.md` (upgrade issues)

**Action**: **KEEP** as-is in SYNTHESIZED (valuable historical reference)

---

### Category 4: Upgrade & Development History (CONSOLIDATE)

**Overlapping docs**:
- `archive/MAMBA_UPGRADE_ANALYSIS.md`
- `archive/STACK_UPGRADE_PLAN_V3.md`
- `archive/UPGRADE_PLAN.md`
- `archive/POST-DEPENDENCY-UPGRADE-ARCHITECTURAL-REVIEW.md`

**Action**: **SYNTHESIZE** into:
- `UPGRADE_HISTORY_V3.md` (PyTorch 2.5 + Mamba 2.2.5 + lessons learned)

---

### Category 5: Testing & Configuration (CONSOLIDATE)

**Overlapping docs**:
- `archive/TEST_SUITE_CONFIG_SSOT.md`
- `archive/TEST_SUITE_FIX_PLAN.md`

**Action**: **SYNTHESIZE** into:
- `TESTING_CONFIGURATION.md` (test suite setup + known issues)

---

### Category 6: Training Insights (CONSOLIDATE)

**Overlapping docs**:
- `archive/LOCAL_VS_CLOUD_TRAINING.md`

**Action**: **SYNTHESIZE** into:
- `TRAINING_PLATFORM_COMPARISON.md` (Local RTX 4090 vs Modal A100)

---

### Category 7: Meta/Process Docs (KEEP AS-IS)

**Documentation about documentation**:
- `archive-2/DOCS_UPDATE_SUMMARY_OCT1.md` (today's changes)
- `archive-2/DOCS_STATUS_OCTOBER_2025.md` (reality check)

**Action**: **KEEP** as-is (useful for understanding doc evolution)

---

## 🎯 SYNTHESIZED DOCS TO CREATE

### RECENT-WORK-SYNTHESIZED/ Structure

```
docs/RECENT-WORK-SYNTHESIZED/
├── 01-nan-and-stability/
│   ├── NAN_PREVENTION_COMPLETE.md          (NEW - consolidate NaN docs)
│   ├── GRADIENT_MONITORING_GUIDE.md        (NEW - realistic expectations)
│   └── ARCHITECTURE_V3_STABILITY.md        (NEW - eigendecomp + PR1-5)
│
├── 02-warmup-schedules/
│   └── WARMUP_SCHEDULES_GUIDE.md           (NEW - config + usage)
│
├── 03-incidents/
│   ├── PR708_APPLICATION.md                (KEEP - Mamba kernel fix)
│   ├── MODAL_XID31_RECURRENCE.md           (KEEP - Triton cache)
│   ├── MODAL_A100_CUDA_FAILURE_ANALYSIS.md (KEEP - CUDA failures)
│   └── PYTORCH_2.5_UPGRADE_INCIDENT.md     (KEEP - upgrade issues)
│
├── 04-development/
│   ├── UPGRADE_HISTORY_V3.md               (NEW - consolidate upgrades)
│   ├── TESTING_CONFIGURATION.md            (NEW - test suite)
│   └── TRAINING_PLATFORM_COMPARISON.md     (NEW - local vs cloud)
│
└── 05-meta/
    ├── DOCS_UPDATE_SUMMARY_OCT1.md         (KEEP - today's work)
    └── DOCS_STATUS_OCTOBER_2025.md         (KEEP - reality check)
```

---

## 📝 SYNTHESIS DETAILS

### 01-nan-and-stability/NAN_PREVENTION_COMPLETE.md

**Sources**:
- `10-major-NAN-refactor/NAN_CANONICAL.md` (canonical strategies)
- `archive/NAN-PROTECTION-REFERENCE.md` (3-tier protection)
- `10-major-NAN-refactor/PR1_FINAL_STATUS.md` (boundary normalization)
- `10-major-NAN-refactor/PR2_FINAL_STATUS.md` (edge stream bounding)
- `10-major-NAN-refactor/PR3_ADJACENCY_CONDITIONING_PLAN.md` (adjacency)
- `10-major-NAN-refactor/PR5_DEFINITIVE_CLEANUP.md` (final cleanup)

**Content**:
- 3-tier NaN protection (environment + component + output)
- PR-1: Boundary Normalization (COMPLETE)
- PR-2: Edge Stream Bounding (COMPLETE)
- PR-3: Adjacency Conditioning (COMPLETE)
- PR-5: Edge Similarity Margin (COMPLETE)
- Current status: v3.4.1 (zero NaN/Inf at batch 166)
- Troubleshooting guide

### 01-nan-and-stability/GRADIENT_MONITORING_GUIDE.md

**Sources**:
- `10-major-NAN-refactor/GRADIENT_BEHAVIOR_GUIDE.md` (original guide)
- `archive-2/GRADIENT_STABILITY_ANALYSIS_OCT1.md` (batch 59 analysis)
- `archive-2/DOCS_STATUS_OCTOBER_2025.md` (realistic expectations)

**Content**:
- Realistic gradient expectations by phase:
  - Early training (0-200): P95 20-60
  - Warmup (200-1000): P95 10-30
  - Stable (1000+): P95 5-20
- BiMamba+GNN vs transformer differences
- Current training data (batch 166: P95=12.64)
- When to intervene vs wait
- Clipping frequency expectations
- What "large grad norm" messages mean

### 01-nan-and-stability/ARCHITECTURE_V3_STABILITY.md

**Sources**:
- `10-major-NAN-refactor/ARCHITECTURAL_INSTABILITY_FIX.md` (original fix)
- `archive-2/ARCHITECTURAL_STABILITY_INVESTIGATION.md` (Sept 30)
- `archive-2/DEEP_INVESTIGATION_FINDINGS.md` (comprehensive audit)

**Content**:
- v3.3.1: Eigendecomposition fix (Sept 30)
  - Why eigenvectors must be detached
  - PyTorch eigendecomposition backward instability
  - Implementation (gnn_pyg.py:205)
- v3.4.0: Pre-norm Mamba
  - Switch from post-norm to pre-norm
  - Alignment with reference Mamba2
- v3.4.1: Warmup schedules
  - Optional gradient stabilization
  - Adjacency temperature + focal gamma
- Training validation (batch 166 results)
- False alarms debunked (P95 < 1.0 speculation)

### 02-warmup-schedules/WARMUP_SCHEDULES_GUIDE.md

**Sources**:
- `archive-2/WARMUP_IMPLEMENTATION_ANALYSIS.md` (complete implementation)
- `archive-2/WARMUP_SCHEDULES_IMPLEMENTATION.md` (redundant, skip)

**Content**:
- What warmup schedules are
- Why they help (smoother early training)
- Config schema (WarmupScheduleConfig)
- Adjacency temperature schedule (τ: 2.0 → 1.0)
- Focal gamma schedule (γ: 1.0 → 2.0)
- Usage examples (local + Modal)
- Implementation details (model state management)
- Training results (batch 166: 52% P95 decrease)
- When to enable vs disable

### 03-incidents/ (All KEEP AS-IS)

No changes - these are valuable historical records.

### 04-development/UPGRADE_HISTORY_V3.md

**Sources**:
- `archive/PYTORCH_2.5_UPGRADE_INCIDENT.md`
- `archive/MAMBA_UPGRADE_ANALYSIS.md`
- `archive/STACK_UPGRADE_PLAN_V3.md`
- `archive/UPGRADE_PLAN.md`
- `archive/POST-DEPENDENCY-UPGRADE-ARCHITECTURAL-REVIEW.md`

**Content**:
- PyTorch 2.4.1 → 2.5.0 upgrade (Sept 2025)
  - Why: Mamba 2.2.5 compatibility
  - Issues: Mixed precision bugs, gradient instability
  - Solutions: Disable AMP on local, enable on Modal
- Mamba 2.0.4 → 2.2.5 upgrade
  - Why: A100 int32 kernel fix (PR #708)
  - Issues: XID 31 crashes
  - Solutions: Triton cache workaround
- Lessons learned
- Current stable stack (v3.4.1)

### 04-development/TESTING_CONFIGURATION.md

**Sources**:
- `archive/TEST_SUITE_CONFIG_SSOT.md`
- `archive/TEST_SUITE_FIX_PLAN.md`

**Content**:
- Test suite structure
- GPU vs CPU tests
- Modal CI/CD setup
- Known issues and workarounds
- Environment variables for testing
- Coverage requirements

### 04-development/TRAINING_PLATFORM_COMPARISON.md

**Sources**:
- `archive/LOCAL_VS_CLOUD_TRAINING.md`

**Content**:
- RTX 4090 (local) specs and settings
- A100-80GB (Modal) specs and settings
- Performance comparison
- Cost analysis
- When to use which platform

### 05-meta/ (All KEEP AS-IS)

No changes - these document our documentation process.

---

## ✅ SYNTHESIS CHECKLIST

### Phase 1: Create Core Stability Docs
- [ ] Create `01-nan-and-stability/NAN_PREVENTION_COMPLETE.md`
- [ ] Create `01-nan-and-stability/GRADIENT_MONITORING_GUIDE.md`
- [ ] Create `01-nan-and-stability/ARCHITECTURE_V3_STABILITY.md`

### Phase 2: Create Warmup Guide
- [ ] Create `02-warmup-schedules/WARMUP_SCHEDULES_GUIDE.md`

### Phase 3: Copy Incident Reports
- [ ] Copy `PR708_APPLICATION.md` to `03-incidents/`
- [ ] Copy `MODAL_XID31_RECURRENCE.md` to `03-incidents/`
- [ ] Copy `MODAL_A100_CUDA_FAILURE_ANALYSIS.md` to `03-incidents/`
- [ ] Copy `PYTORCH_2.5_UPGRADE_INCIDENT.md` to `03-incidents/`

### Phase 4: Create Development Docs
- [ ] Create `04-development/UPGRADE_HISTORY_V3.md`
- [ ] Create `04-development/TESTING_CONFIGURATION.md`
- [ ] Create `04-development/TRAINING_PLATFORM_COMPARISON.md`

### Phase 5: Copy Meta Docs
- [ ] Copy `DOCS_UPDATE_SUMMARY_OCT1.md` to `05-meta/`
- [ ] Copy `DOCS_STATUS_OCTOBER_2025.md` to `05-meta/`

### Phase 6: Verification
- [ ] Every COMBINED doc accounted for (extracted or kept)
- [ ] All info preserved in SYNTHESIZED
- [ ] No redundancy in SYNTHESIZED
- [ ] All docs match v3.4.1 codebase
- [ ] Code examples work
- [ ] Config examples valid

### Phase 7: Cleanup (AFTER VERIFICATION)
- [ ] Delete `docs/RECENT-WORK-COMBINED/` (all info extracted)
- [ ] Keep `docs/RECENT-WORK-SYNTHESIZED/` (clean consolidated docs)

---

## 🚀 EXECUTION

**START NOW**: Create SYNTHESIZED docs one by one, verifying accuracy against codebase.

**NEXT**: Once SYNTHESIZED complete → Integrate into docs/00-09/

**FINAL**: Clean, accurate, non-redundant documentation! 🎉

---

**Ready to begin synthesis?** Let's start with the most critical: NaN prevention + gradient monitoring!
