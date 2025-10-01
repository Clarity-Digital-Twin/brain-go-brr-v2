# Documentation Consolidation Plan - October 1, 2025

**Purpose**: Extract all valuable information from archived docs and integrate into main docs (00-10) to reflect v3.4.1 codebase accurately.

**Status**: In Progress
**Critical Rule**: ❌ **DO NOT DELETE ARCHIVES** until all information extracted and verified

---

## 📊 DOCUMENTATION INVENTORY

**Total Docs**: 90 markdown files
- Main docs (00-10): 70 files
- Archive: 13 files
- Archive-2: 7 files

---

## 📋 ARCHIVE AUDIT

### docs/archive/ (Historical Incident Reports)

These docs chronicle past issues that have been resolved. Need to extract:
- **Lessons learned** → Integrate into troubleshooting/operations
- **Solutions that worked** → Document in relevant sections
- **Configuration insights** → Add to config guides

| File | Size | Status | Extract To |
|------|------|--------|------------|
| LOCAL_VS_CLOUD_TRAINING.md | ? | Audit | docs/05-training/local.md + modal.md |
| MAMBA_UPGRADE_ANALYSIS.md | ? | Audit | docs/09-development/versioning-and-roadmap.md |
| MODAL_A100_CUDA_FAILURE_ANALYSIS.md | ? | Audit | docs/08-operations/troubleshooting.md |
| MODAL_XID31_RECURRENCE.md | ? | Audit | docs/08-operations/troubleshooting.md |
| NAN-PROTECTION-REFERENCE.md | ? | Audit | docs/08-operations/nan-prevention-complete.md |
| NAN-REGRESSION-POST-DEPENDENCY-UPGRADE.md | ? | Audit | docs/08-operations/nan-troubleshooting.md |
| POST-DEPENDENCY-UPGRADE-ARCHITECTURAL-REVIEW.md | ? | Audit | docs/09-development/versioning-and-roadmap.md |
| PR708_APPLICATION.md | ? | Audit | docs/08-operations/troubleshooting.md |
| PYTORCH_2.5_UPGRADE_INCIDENT.md | ? | Audit | docs/01-installation/gpu-stack.md |
| STACK_UPGRADE_PLAN_V3.md | ? | Audit | docs/09-development/versioning-and-roadmap.md |
| TEST_SUITE_CONFIG_SSOT.md | ? | Audit | docs/09-development/testing.md |
| TEST_SUITE_FIX_PLAN.md | ? | Audit | docs/09-development/testing.md |
| UPGRADE_PLAN.md | ? | Audit | docs/09-development/versioning-and-roadmap.md |

### docs/archive-2/ (October 2025 Investigation Docs)

These were just moved from root - they're the investigation docs we updated earlier today!

| File | Size | Status | Extract To |
|------|------|--------|------------|
| ARCHITECTURAL_STABILITY_INVESTIGATION.md | 24K | Updated Today | Keep as historical reference |
| DEEP_INVESTIGATION_FINDINGS.md | 31K | Updated Today | Keep as historical reference |
| DOCS_STATUS_OCTOBER_2025.md | 18K | Master Doc | Extract to relevant sections |
| DOCS_UPDATE_SUMMARY_OCT1.md | 9K | Change Log | Keep as historical |
| GRADIENT_STABILITY_ANALYSIS_OCT1.md | 26K | Updated Today | Extract gradient expectations |
| WARMUP_IMPLEMENTATION_ANALYSIS.md | 22K | Complete Status | Extract to training docs |
| WARMUP_SCHEDULES_IMPLEMENTATION.md | 17K | Redundant | Merge or skip |

---

## 🎯 MAIN DOCS STRUCTURE AUDIT

### Current State by Section

#### 00-overview/ ✅
- architecture-summary.md
- overview.md
- performance-targets.md
**Status**: Likely needs v3.4.1 updates

#### 01-installation/ ✅
- env-setup.md
- gpu-stack.md
- preflight-checks.md
**Needs**: PyTorch 2.5.0 specifics from archives

#### 02-data/ ✅
- CACHE_MANIFEST_ARCHITECTURE.md
- cache-layout.md
- overview.md
- preprocessing.md
**Status**: Probably current

#### 03-configuration/ ✅
- config-schema.md
- env-vars.md
- local-configs.md
- modal-configs.md
- schema-validation.md
**Needs**: Warmup schedule config from archive-2

#### 04-model/ ✅
- edge-features-and-adjacency.md
- gnn.md
- laplacian-pe.md
- mamba.md
- postprocess.md
- tcn.md
- time-frequency-hybrid.md
- v3-architecture.md
**Needs**:
- v3.4.0 pre-norm Mamba
- v3.4.1 warmup schedules
- v3.3.1 eigendecomposition fix
- Gradient expectations from archive-2

#### 05-training/ ✅
- checkpoint-strategy.md
- local.md
- modal-deployment.md
- modal.md
- monitoring.md
- resume.md
- smoke-tests.md
**Needs**:
- Warmup schedule usage
- Gradient monitoring expectations
- Local vs Modal insights from archive

#### 06-evaluation/ ✅
- metrics-and-taes.md
**Status**: Probably current

#### 07-cli-tools/ ✅
- cli-usage.md
- makefile-commands.md
**Status**: Check if current

#### 08-operations/ 🚨 HIGH PRIORITY
- modal-volume-architecture.md
- nan-logits-dynamic-pe.md
- nan-prevention-complete.md
- nan-troubleshooting.md
- performance-optimization.md
- pr5-edge-similarity-clamping.md
- streaming.md
- troubleshooting.md
- v3-nan-explosion-resolution.md
- wsl2-notes.md
**Needs**: MASSIVE consolidation from archive/ NaN docs

#### 09-development/ ✅
- bug-tracker.md
- coding-standards.md
- technical-debt.md
- testing.md
- versioning-and-roadmap.md
**Needs**: Upgrade history from archive/

#### 10-major-NAN-refactor/ ✅ (Historical Record)
- ARCHITECTURAL_INSTABILITY_FIX.md
- GRADIENT_BEHAVIOR_GUIDE.md
- NAN_CANONICAL.md
- PR1-5 docs
**Status**: These are complete historical records, keep as-is

---

## 🔧 EXTRACTION STRATEGY

### Phase 1: Critical Information Extraction (NOW)
1. **Gradient Expectations** (archive-2)
   - Extract realistic P95 targets from DOCS_STATUS_OCTOBER_2025.md
   - Add to docs/05-training/monitoring.md
   - Add to docs/04-model/v3-architecture.md

2. **Warmup Schedules** (archive-2)
   - Extract implementation details from WARMUP_IMPLEMENTATION_ANALYSIS.md
   - Add to docs/03-configuration/schema-validation.md
   - Add to docs/05-training/local.md + modal.md

3. **NaN Prevention** (archive + 08-operations)
   - Consolidate all NaN docs into nan-prevention-complete.md
   - Extract solutions from archive/ incidents
   - Update with v3.4.1 status

### Phase 2: Troubleshooting Consolidation
4. **Modal Issues** (archive)
   - Extract XID 31 solutions
   - Extract CUDA failure workarounds
   - Add to docs/08-operations/troubleshooting.md

5. **Upgrade History** (archive)
   - Extract PyTorch 2.5.0 lessons
   - Extract Mamba upgrade insights
   - Add to docs/09-development/versioning-and-roadmap.md

### Phase 3: Architecture Updates
6. **v3.4.1 Features** (codebase + archive-2)
   - Pre-norm Mamba (v3.4.0)
   - Warmup schedules (v3.4.1)
   - Eigendecomposition fix (v3.3.1)
   - Update docs/04-model/v3-architecture.md

7. **Training Insights** (archive + archive-2)
   - Local vs Modal differences
   - Gradient behavior by phase
   - Add to docs/05-training/monitoring.md

### Phase 4: Verification
8. **Code-to-Docs Alignment**
   - Verify each doc matches actual code
   - Check config examples match schemas
   - Validate command examples work

---

## 📝 EXTRACTION TRACKING

### Archive → Main Docs Mapping

**From archive/NAN-PROTECTION-REFERENCE.md**:
- [ ] Extract 3-tier protection strategy → docs/08-operations/nan-prevention-complete.md
- [ ] Extract BGB_SANITIZE_GRADS usage → docs/03-configuration/env-vars.md
- [ ] Extract gradient sanitization → docs/05-training/monitoring.md

**From archive/MODAL_XID31_RECURRENCE.md**:
- [ ] Extract Triton cache workaround → docs/08-operations/troubleshooting.md
- [ ] Extract int32 kernel issue → docs/01-installation/gpu-stack.md

**From archive/PYTORCH_2.5_UPGRADE_INCIDENT.md**:
- [ ] Extract version requirements → docs/01-installation/gpu-stack.md
- [ ] Extract compatibility notes → docs/09-development/versioning-and-roadmap.md

**From archive-2/DOCS_STATUS_OCTOBER_2025.md**:
- [ ] Extract gradient expectations → docs/05-training/monitoring.md
- [ ] Extract phase-specific targets → docs/04-model/v3-architecture.md
- [ ] Extract implementation status → docs/00-overview/architecture-summary.md

**From archive-2/WARMUP_IMPLEMENTATION_ANALYSIS.md**:
- [ ] Extract config schema → docs/03-configuration/schema-validation.md
- [ ] Extract usage guide → docs/05-training/local.md + modal.md
- [ ] Extract implementation details → docs/04-model/mamba.md

**From archive-2/GRADIENT_STABILITY_ANALYSIS_OCT1.md**:
- [ ] Extract early training expectations → docs/05-training/monitoring.md
- [ ] Extract clipping frequency guidance → docs/08-operations/troubleshooting.md

---

## 🎯 DELIVERABLES

### Updated Main Docs (Priority Order)

1. **docs/05-training/monitoring.md** (CRITICAL)
   - Realistic gradient expectations by phase
   - Warmup schedule effects
   - When to intervene vs wait
   - Batch 166 checkpoint data

2. **docs/04-model/v3-architecture.md** (CRITICAL)
   - v3.3.1 eigendecomposition fix
   - v3.4.0 pre-norm Mamba
   - v3.4.1 warmup schedules
   - Current parameter count (31M)

3. **docs/08-operations/nan-prevention-complete.md** (HIGH)
   - Consolidate all NaN prevention strategies
   - 3-tier protection detailed
   - Historical incidents and solutions
   - Current status (zero NaN/Inf at batch 166)

4. **docs/08-operations/troubleshooting.md** (HIGH)
   - Modal XID 31 solutions
   - CUDA failure workarounds
   - Common training issues

5. **docs/03-configuration/schema-validation.md** (MEDIUM)
   - Add WarmupScheduleConfig details
   - Add usage examples
   - Link to training guides

6. **docs/01-installation/gpu-stack.md** (MEDIUM)
   - PyTorch 2.5.0 + CUDA 12.4 requirements
   - Mamba 2.2.5 specifics
   - Known issues and workarounds

7. **docs/09-development/versioning-and-roadmap.md** (LOW)
   - Upgrade history (PyTorch 2.5, Mamba 2.2.5)
   - Lessons learned
   - Future considerations

### New Docs (If Needed)

- **docs/05-training/gradient-expectations.md** (maybe)
  - Detailed phase-by-phase expectations
  - BiMamba+GNN vs transformer comparisons
  - When norms are concerning vs normal

---

## ✅ SUCCESS CRITERIA

**Documentation is complete when**:
1. ✅ All main docs (00-10) reflect v3.4.1 codebase accurately
2. ✅ Every valuable insight from archives is preserved in main docs
3. ✅ No contradictions between docs
4. ✅ All code examples work
5. ✅ All config examples match schemas
6. ✅ New users can understand system without archives
7. ✅ Troubleshooting covers all known issues

**Then and only then**: Archives can be safely deleted (with git history preserving them)

---

## 🚀 EXECUTION PLAN

### Today (Oct 1)
- [x] Create this consolidation plan
- [ ] Extract gradient expectations → monitoring.md
- [ ] Extract warmup schedules → configuration + training docs
- [ ] Update v3-architecture.md with v3.4.1 features

### This Week
- [ ] Consolidate NaN docs → nan-prevention-complete.md
- [ ] Update troubleshooting with Modal solutions
- [ ] Update gpu-stack with PyTorch 2.5 specifics
- [ ] Verify all main docs accuracy

### Verification
- [ ] Run all commands in docs to verify they work
- [ ] Check config examples against schemas
- [ ] Have another person review (or fresh AI session)
- [ ] Compare against actual codebase line-by-line

---

**Status**: Ready to begin extraction
**Next**: Start with critical path (gradient expectations + warmup)
