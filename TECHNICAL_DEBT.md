# Technical Debt

**Date**: October 11, 2025
**Status**: 🟡 **1 IMPORTANT ITEM** - P2 config separation needed before FLA training
**Version**: v3.11.0 (StatefulDataLoader & Mid-Epoch Resume)
**Training Impact**: P2 must be fixed before FLA training starts, P3 items are polish only

---

## Executive Summary

| Priority | Count | Training Impact | Status |
|----------|-------|-----------------|--------|
| **P0 BLOCKER** | 0 | None | ✅ **CLEAR** |
| **P1 URGENT** | 0 | None | ✅ **CLEAR** |
| **P2 MEDIUM** | 1 | Must fix before FLA training | ⚠️ **ACTION REQUIRED** |
| **P3 LOW** | 2 | None (polish only) | 📝 **OPTIONAL** |

**CRITICAL**: P2-1 must be fixed before starting FLA training to prevent checkpoint contamination!

---

## ⚠️ P2: ACTION REQUIRED

### P2-1: YAML Config Output Directory Conflict (CRITICAL BEFORE FLA TRAINING)

**Status**: ⚠️ **MUST FIX BEFORE FLA TRAINING**

**Current Problem**:
Both BiMamba2 and FLA configs use the **SAME output directory**:

```yaml
# configs/modal/train_bimamba.yaml
experiment:
  name: v3_full_training
  output_dir: /results/v3_full_training  # ← BiMamba2 checkpoints here

# configs/modal/train_fla.yaml
experiment:
  name: v3_full_training_fla
  output_dir: /results/v3_full_training  # ← SAME PATH! ❌
```

**Why This Is Critical**:
1. **BiMamba2 currently training** in `/results/v3_full_training` with checkpoints saved
2. If we accidentally resume with FLA config → tries to load BiMamba2 checkpoints into FLA model → **checkpoint contamination**
3. Already happened once today (killed immediately, no damage)
4. Must separate directories before FLA training begins

**Contamination Risk**:
- ❌ Different model architectures (BiMamba2 vs GatedDeltaNet)
- ❌ Checkpoints are NOT interchangeable between architectures
- ❌ Loading wrong checkpoint → silent failures or training corruption
- ❌ Wasted compute if contamination goes unnoticed

**Solution Strategy**:
1. **DO NOT touch BiMamba2 config** - it's currently training, checkpoints already there
2. **Change FLA config ONLY** before FLA training starts:
   ```yaml
   # configs/modal/train_fla.yaml
   experiment:
     name: v3_full_training_fla
     output_dir: /results/v3_fla_training  # ← NEW separate directory
   ```
3. Similarly update local configs:
   ```yaml
   # configs/local/train_fla.yaml
   experiment:
     output_dir: results/v3_fla_training  # ← Separate from results/v3_full_training
   ```

**When To Fix**:
- ⏰ **Before starting FLA training** (not urgent while only BiMamba2 runs)
- ✅ Safe to fix now or anytime before FLA training
- 📋 Simple config edit, no code changes needed

**Verification After Fix**:
```bash
# Check configs have different output directories
grep -A5 "experiment:" configs/modal/train_bimamba.yaml configs/modal/train_fla.yaml
grep -A5 "experiment:" configs/local/train_bimamba.yaml configs/local/train_fla.yaml

# Expected:
# train_bimamba.yaml: output_dir: /results/v3_full_training (or results/v3_full_training)
# train_fla.yaml:     output_dir: /results/v3_fla_training (or results/v3_fla_training)
```

**Priority**: P2 - Medium urgency, **MUST fix before FLA training**, zero urgency while only BiMamba2 runs

**Files to Update**:
- `configs/modal/train_fla.yaml` - Change output_dir to `/results/v3_fla_training`
- `configs/local/train_fla.yaml` - Change output_dir to `results/v3_fla_training`
- (BiMamba2 configs stay unchanged - already in use)

---

## 📝 P3: OPEN (Documentation/Polish) - No Training Impact

### P3-1: Pydantic Field Warning Still Appearing in Modal Logs

**Status**: 📝 **OPEN FOR FUTURE INVESTIGATION**

**Evidence from Modal Logs** (October 11, 2025):
```
UserWarning: The 'repr' attribute with value False was provided to the `Field()`
function, which has no effect in the context it was used.

UserWarning: The 'frozen' attribute with value True was provided to the `Field()`
function, which has no effect in the context it was used.
```

**Impact**:
- ✅ Zero impact on training correctness or performance
- ⚠️ Cosmetic log noise only
- ⚠️ May indicate opportunity for further code cleanup

**Files Referenced**:
- `docs/archive_v3/PYDANTIC_WARNING_ANALYSIS.md` - Previous investigation results
- `src/brain_brr/config/schemas.py` - Config schema definitions (previously cleaned)

**Future Work**:
- Further trace remaining warnings to source in Pydantic v2 schema generation
- Investigate if additional `Annotated[Type | None, Field(...)]` patterns needed
- Consider if this is truly fixable or a Pydantic v2 quirk we must accept

**Priority**: P3 - Cosmetic only, no functional impact

---

### P3-2: Missing .gitkeep Files for Directory Structure Documentation

**Status**: 📝 **OPEN FOR OSS CONTRIBUTOR CLARITY**

**Current Situation**:
- Local training requires specific directory structure for data files
- Directories exist locally but not documented in git for OSS contributors
- Contributors may be confused about where to place downloaded TUSZ data

**Directories That Need .gitkeep** (for structure documentation):
```
cache/
cache/tusz/
cache/tusz_mmap/
data_ext4/
data_ext4/tusz/
```

**Purpose**:
- Document directory structure in git without committing actual data files
- Help OSS contributors understand local setup requirements
- Show where to place TUSZ data for local training

**Future Work**:
- Add `.gitkeep` to required cache/data directories
- Update `.gitignore` to ensure data files still excluded
- Add README.md in each directory explaining expected contents

**Priority**: P3 - OSS contributor experience only, doesn't block internal development

---

## 🎉 Current Status

**Production Readiness**:
- ✅ **P0 Blockers**: 0 issues - training stable
- ✅ **P1 Urgent**: 0 issues - all critical fixes deployed
- ⚠️ **P2 Medium**: 1 issue - YAML config separation needed before FLA training
- 📝 **P3 Low**: 2 issues - documentation/polish only, zero training impact

**Training Status**:
- ✅ BiMamba2 baseline training LIVE on Modal A100 (Epoch 3, Batch 354/1284)
- ✅ Exact mid-epoch resume working with StatefulDataLoader
- ✅ All checkpoint fixes deployed and verified
- ⚠️ Must fix P2-1 before starting FLA training

**Version History**:
- **v3.11.0** (Oct 11): StatefulDataLoader integration, exact mid-epoch resume
- **v3.10.0** (Oct 10): Auto-restart + three checkpoint fixes
- **v3.9.1** (Oct 9): Validation OOM fix
- **v3.9.0** (Oct 8): Bulletproof checkpoints + timeout guard
- Historical resolved issues archived in `docs/archive_v3/TECHNICAL_DEBT_RESOLVED.md`

---

## Quality Maintenance Policy

**Before Every Major Training Run**:
```bash
make q        # Ensure zero lint/format/type errors
make test     # Ensure all tests pass
```

**Before FLA Training**:
```bash
# Fix P2-1: Update FLA config output directories
vim configs/modal/train_fla.yaml   # Change output_dir
vim configs/local/train_fla.yaml   # Change output_dir
grep "output_dir" configs/**/*fla.yaml  # Verify changes
```

---

**Status**: 🟡 **1 IMPORTANT ITEM** - P2 config separation needed before FLA training
**Current Version**: v3.11.0 (StatefulDataLoader & Mid-Epoch Resume)
**Training Status**: BiMamba2 baseline training LIVE, P2 must be fixed before FLA training
**Next Action**: Fix P2-1 (YAML configs) → Monitor BiMamba2 → Launch FLA training → Compare! 🚀
