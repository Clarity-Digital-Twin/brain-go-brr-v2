# TODO - Active Tasks

**Last Updated:** 2025-10-06 (14:30 UTC)
**Status:** 🟢 **ZERO ACTIVE TASKS** - All known work completed

---

## Current Status

**Active Work**: None - codebase is ready for production training

**In Progress**:
- Modal smoke test running (ap-39MmeGlcwE1KgLEibaq8Cg)

**Next Steps**:
1. Monitor smoke test completion (~10 min)
2. Launch full Modal training if smoke test passes
3. Optional post-training optimizations (P4/P5)

---

## Recently Completed (October 6, 2025)

### P0 Blockers - NPZ Cache Contamination
- [x] Clean 3 stray NPZ files from Modal cache
- [x] Fix datasets.py NPZ creation bug
- [x] Update cache validation to check NPY files
- [x] Fix test regression (cache_dir=None support)

### P2 Code Quality
- [x] Fix all type annotations (3 files)
- [x] Extract duplicate `_load_cache_for_worker` (120 lines eliminated)
- [x] Update NPZ references in comments
- [x] Fix clean_cache() paths

### Quality Verification
- [x] Run `make q` (lint + format + mypy + config validation)
- [x] Run `make test` (104 tests, 83.80% coverage)
- [x] Archive old debt documentation to `docs/archive_v1/`

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
