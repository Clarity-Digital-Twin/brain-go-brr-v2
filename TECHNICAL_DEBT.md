# Technical Debt

**Date**: October 11, 2025
**Status**: ✅ **ZERO P0/P1/P2 DEBT** - Only 2 P3 polish items remain
**Version**: v3.11.0 (StatefulDataLoader & Mid-Epoch Resume)
**Training Impact**: NONE - all critical items resolved, P3 items are polish only

---

## Executive Summary

| Priority | Count | Training Impact | Status |
|----------|-------|-----------------|--------|
| **P0 BLOCKER** | 0 | None | ✅ **CLEAR** |
| **P1 URGENT** | 0 | None | ✅ **CLEAR** |
| **P2 MEDIUM** | 0 | None | ✅ **CLEAR** |
| **P3 LOW** | 2 | None (polish only) | 📝 **OPTIONAL** |

**All critical training blockers resolved!** Ready for both BiMamba2 and FLA full training.

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
- ✅ **P2 Medium**: 0 issues - all medium-priority items resolved
- 📝 **P3 Low**: 2 issues - documentation/polish only, zero training impact

**Training Status**:
- ✅ BiMamba2 baseline training LIVE on Modal A100 (Epoch 3, ongoing)
- ✅ Exact mid-epoch resume working with StatefulDataLoader
- ✅ All checkpoint fixes deployed and verified
- ✅ FLA config separation complete - ready for FLA training when BiMamba2 completes

**Version History**:
- **v3.11.0** (Oct 11): StatefulDataLoader integration, exact mid-epoch resume, YAML config separation (P2-1 RESOLVED)
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

**Config Verification** (optional check):
```bash
# Verify configs have separate output directories
grep -A5 "experiment:" configs/modal/train_bimamba.yaml configs/modal/train_fla.yaml
grep -A5 "experiment:" configs/local/train_bimamba.yaml configs/local/train_fla.yaml
```

---

**Status**: ✅ **ZERO P0/P1/P2 DEBT** - Only 2 P3 polish items remain
**Current Version**: v3.11.0 (StatefulDataLoader & Mid-Epoch Resume)
**Training Status**: BiMamba2 baseline training LIVE, all critical issues resolved
**Next Action**: Monitor BiMamba2 → Launch FLA training → Compare! 🚀
