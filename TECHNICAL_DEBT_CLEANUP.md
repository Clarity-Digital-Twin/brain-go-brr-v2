# Technical Debt and Cleanup Plan

**Created**: September 28, 2025
**Status**: DRAFT - Pending Senior Review
**Purpose**: Document all technical debt, cruft, and inconsistencies for systematic cleanup of v3.2.1

## Executive Summary

After the v3.2.0 release and documentation updates, a comprehensive audit revealed several areas of technical debt and inconsistencies. This document catalogs all findings for review before making changes.

## 🔴 Critical Issues (Must Fix)

### 1. Dead Code - ClampRetirementConfig
**Location**: `src/brain_brr/config/schemas.py` lines 254-288
- `ClampRetirementConfig` class still defined but never used in any configs
- `ModelConfig.clamp_retirement` field references it (lines 285-288)
- `src/brain_brr/models/detector.py` lines 93-94, 555-565 still check for it
- **Impact**: Dead code that confuses readers
- **Action**: Remove entirely from schemas.py and detector.py

### 2. Module Version Inconsistencies
**Locations**:
```
src/__init__.py:1: "Brain-Go-Brr v2"
src/brain_brr/cli/__init__.py:1: "Brain-Go-Brr v2"
src/brain_brr/utils/__init__.py:1: "Brain-Go-Brr v2"
src/brain_brr/cli/cli.py:20: "Brain-Go-Brr v2"
src/brain_brr/train/wandb_integration.py:1: "Brain-Go-Brr v2"
```
- **Impact**: Confusing version references
- **Action**: Update all to "Brain-Go-Brr V3"

## ⚠️ Medium Priority Issues

### 3. Brittle File Count Checks
**Location**: `deploy/modal/app.py`
- Lines 212-213: `print(f"Train files: {train_count} (expected: 4667)")`
- Lines 711, 715: Hard-coded checks for exactly 4667/1832 files
- **Impact**: Will fail if dataset changes slightly
- **Action**: Change to ranges (e.g., 4600-4700, 1800-1900)

### 4. Legacy V2 References in Code
**Locations**:
- `src/brain_brr/models/detector.py:605`: Comment about "V2 heuristic path"
- `src/brain_brr/models/gnn_pyg.py:368`: Comment about "v2 compatibility"
- `src/brain_brr/data/io.py:298`: Reference to "v2.0.3 data"
- **Impact**: Misleading comments
- **Action**: Update or remove references

### 5. Val vs Dev Naming Inconsistency
**Internal Code** (OK to keep):
- `src/brain_brr/train/loop.py`: Uses `val_loader`, `val_dataset` variables
- This is fine for internal variables

**External Facing** (Should fix):
- `src/brain_brr/cli/cli.py:206`: CLI accepts "val" as split option
- `configs/`: All cleaned up ✅
- `docs/`: All cleaned up ✅

### 9. Debug Print Statements
**Location**: `src/brain_brr/train/loop.py`
- Lines 264, 654, 660, 668, 672, 718, 752, 1157, 1515: DEBUG print statements
- **Impact**: Verbose output in production
- **Action**: Convert to proper logging or remove

## 📝 Documentation Issues

### 6. PR-1 through PR-4 References Throughout Code
**Locations**: Found in multiple files
- `src/brain_brr/models/detector.py`: 15+ references
- `src/brain_brr/models/mamba.py`: 4 references
- `src/brain_brr/models/tcn.py`: 3 references
- `src/brain_brr/models/gnn_pyg.py`: 5 references
- `src/brain_brr/config/schemas.py`: 8 references

**Analysis**: These document what each PR implemented
- **Recommendation**: KEEP but add header comment explaining they're historical markers

### 7. Test Files for PR Features
**Files**:
```
tests/unit/models/test_pr1_normalization.py
tests/unit/models/test_pr2_bounded_edge.py
tests/unit/models/test_pr3_adjacency_conditioning.py
tests/unit/models/test_pr4_clamp_retirement.py
```
- **Status**: These test implemented features, not removed features
- **Action**: KEEP all except possibly test_pr4_clamp_retirement.py

### 8. Historical Planning Documents
**Location**: `docs/10-final-refactor/`
- All PR planning documents have historical notes ✅
- **Action**: Consider moving to `docs/99-archive/` subdirectory

## 🔍 Dead Code Detection Strategy

### Recommended Tools (2025 Research Update)
1. **vulture** - Lightweight dead code detection (still the primary tool)
2. **coverage.py** - More reliable than vulture but requires running all code paths
3. **pyflakes** - Finds unused imports and variables efficiently
4. **ruff** - Modern, fast linter with dead code rules (F401, F841)
5. **pycln** - Auto-remove unused imports
6. **Prospector** - Bundles multiple tools including vulture
7. **Codacy/Qodana** - CI/CD platforms for continuous monitoring
8. **Smart TS XL** - AI-driven static analysis (new in 2025)

### Detection Commands
```bash
# Install tools
pip install vulture pyflakes pycln

# Find dead code
vulture src/ --min-confidence 90

# Find unused imports
pyflakes src/

# Use ruff for comprehensive checks
ruff check src/ --select F401,F841  # unused imports, unused variables

# Check test coverage
pytest --cov=src --cov-report=html
```

## 🧹 Cleanup Priority Order

### Phase 1: Critical Fixes (Do First)
1. Remove `ClampRetirementConfig` and all references
2. Update module docstrings from v2 to V3
3. Fix brittle file count checks in Modal

### Phase 2: Code Cleanup
1. Update/remove legacy V2 comments
2. Standardize val→dev in CLI
3. Run dead code detection tools
4. Remove any identified dead code

### Phase 3: Documentation
1. Move historical docs to archive folder
2. Add explanatory headers to PR comments in code
3. Update any remaining stale references

## ✅ Validation Checklist

Before making changes:
- [ ] Run full test suite: `make test`
- [ ] Run quality checks: `make q`
- [ ] Test GPU components: `make test-gpu`
- [ ] Verify smoke test works: `make s`
- [ ] Check Modal deployment still works

After each phase:
- [ ] Commit with clear message
- [ ] Run tests again
- [ ] Document what was changed

## 🚨 Risk Assessment

**Low Risk**:
- Updating docstrings
- Fixing comments
- Moving documentation files

**Medium Risk**:
- Removing ClampRetirementConfig (unused but referenced)
- Changing Modal file count checks

**High Risk**:
- None identified

## 📊 Metrics

Current state:
- Total Python files: ~100
- Total lines of code: ~10,000
- Test coverage: TBD
- Dead code instances: TBD (run vulture)
- DEBUG print statements: 8+ instances in train/loop.py
- TODO/FIXME/HACK comments: 0 found
- NOTE comments: 3 found (acceptable)

## 🤝 Review Process

1. **Internal Review**: This document
2. **External AI Agent Review**: Get second opinion on cleanup plan
3. **Senior Engineer Review**: Final approval before execution
4. **Phased Execution**: Implement in priority order
5. **Validation**: Test after each phase

## 📎 Appendix: Full Search Results

### ClampRetirement References
```
src/brain_brr/config/schemas.py:254-288 (class definition)
src/brain_brr/models/detector.py:93-94, 555-565
tests/unit/models/test_pr4_clamp_retirement.py (entire file)
```

### V2 References
```
src/__init__.py:1
src/brain_brr/cli/__init__.py:1
src/brain_brr/utils/__init__.py:1
src/brain_brr/cli/cli.py:20
src/brain_brr/train/wandb_integration.py:1
src/brain_brr/models/detector.py:605
src/brain_brr/models/gnn_pyg.py:368
src/brain_brr/data/io.py:298
```

### File Count References
```
deploy/modal/app.py:212-213, 711, 715
Various docs (acceptable for reference)
```

---

**Next Steps**:
1. Review this document with senior engineer
2. Get external AI agent validation
3. Execute cleanup in phases
4. Document all changes in CHANGELOG.md