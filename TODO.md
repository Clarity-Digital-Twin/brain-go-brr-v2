# TODO - Active Tasks

**Last Updated:** 2025-10-09
**Status:** 🟢 **ZERO ACTIVE TASKS** - All known work completed

---

## Current Status

**Active Work**: None - codebase is ready for production training

**In Progress**:
- Modal full training (100 epochs, v3.9.1 validation OOM fix baseline, W&B run: 983c1fbf706b4d0f8870cc0331dc6201)

**Next Steps**:
1. Monitor Modal training completion (~100 hours total)
2. Analyze results and validate TAES metrics
3. Optional post-training optimizations (P4/P5)

---

## Recently Completed (October 7, 2025)

### v3.8.3 - Manifest Naming Cleanup Complete
- [x] Backed up existing manifests (train + dev)
- [x] Verified cache integrity (4667 train + 1832 dev NPY files)
- [x] Updated 11 code locations to eliminate `.replace("_windows", "")` workarounds
- [x] Regenerated train manifest: 303,990 windows from 4438 NPY files
- [x] Regenerated dev manifest: 148,224 windows
- [x] Verified 100% NPY naming, 0 NPZ references
- [x] Smoke test validation passed
- [x] Version bumped to v3.8.3
- [x] All documentation updated
- [x] Git commit, tag, and push completed
- [x] **ZERO P0/P1/P2/P3 TECHNICAL DEBT ACHIEVED**

### Previous Milestones (October 6, 2025)

#### v3.8.2 - Zero Warnings
- [x] NumPy copy-on-read tensors across all 3 dataset classes
- [x] AMP scheduler guard for accurate LR schedule
- [x] Verified 0 PyTorch warnings in Modal logs

#### v3.8.1 - Complete Tensor Safety
- [x] EEGWindowDataset hardened with proper .clone() calls

#### v3.8.0 - NPZ Cache Cleanup
- [x] Clean 3 stray NPZ files from Modal cache
- [x] Fix datasets.py NPZ creation bug
- [x] Fix all type annotations (3 files)
- [x] Extract duplicate `_load_cache_for_worker` (120 lines eliminated)

### Quality Verification
- [x] Run `make q` (lint + format + mypy + config validation)
- [x] Run `make test` (104 tests, 83.80% coverage)

---

## Optional Future Work (Post-Training)

**Performance Optimization** (only if profiling shows need):
- Profile `.item()` calls - optimize if >1% GPU sync time
- Consider detector.py refactor if readability degrades

**No action required** - these are ideas only, not active tasks.

---

## Quality Maintenance

**Before Each Training Run**:
```bash
make q        # Ensure zero lint/format/type errors
make test     # Ensure all tests pass
```

**Policy**: Maintain zero active TODO items. New work should be completed or explicitly deferred with justification.

---

Keep this file minimal - only active tasks belong here.
