# TODO - Active Tasks

**Last Updated:** 2025-11-01
**Status:** 🟡 **WAITING** - Baseline early stop, then launch Exp4

---

## Current Status

**Active Work**: FLA Baseline running (epoch 30+, plateaued at 0.257)

**In Progress**:
- FLA Baseline: Epoch 30+, patience 13/20, expected early stop at epoch 36 (~Nov 4-5)
- BiMamba2: PAUSED at epoch 6 (Modal, budget control)

**Next Steps**:
1. ⏳ Monitor baseline early stop (~epoch 36, Nov 4-5)
2. 🚀 Launch Exp4 (SGDR) immediately after baseline stops
3. 📊 Monitor Exp4 epochs 1-15 for early signal (should hit 0.28+ if hypothesis correct)
4. 🎯 Analyze Exp4 results, update experimental plan
5. 💡 Decide next action based on Exp4 outcome (see TRAINING.md decision tree)

---

## Recently Completed (November 1, 2025)

### v4.2.0 - SGDR Scheduler + Experimental Pipeline
- [x] Implemented SGDR (cosine restarts) scheduler
- [x] Fixed 4 critical bugs in initial SGDR implementation (agent validation)
- [x] Added test coverage: `test_scheduler_cosine_restarts`
- [x] Created Exp4 config: `train_fla_exp4_cyclic.yaml`
- [x] Validated all configs (100% passing)
- [x] Completed Exp1 (Regularization): FAILED (-2.7% vs baseline)
- [x] Updated documentation: EXPERIMENTAL_PLAN.md (506→121 lines), README.md, STATUS.md, TRAINING.md
- [x] Baseline reached epoch 30+, plateaued at 0.257
- [x] Confirmed best: 0.284 @ epoch 9
- [x] All quality checks passing (lint, format, mypy, tests)

### v4.1.0 - NEDC Evaluation Pipeline (October 20, 2025)
- [x] NEDC evaluation integration (nedc-bench wrapper)
- [x] CLI integration: `python -m src evaluate --checkpoint <path> --nedc`
- [x] Eval cache building: `python -m src build-cache eval`
- [x] 277 lines of tests (231 unit + 46 integration)
- [x] Documentation optimization (CLAUDE.md: 537→372 lines)

### Previous Milestones
- [x] v4.0.0: FLA stack production + WSL2 SIGBUS fix
- [x] v3.11.0: StatefulDataLoader & mid-epoch resume
- [x] v3.10.0: Auto-restart training + checkpoint fix
- [x] v3.9.1: Validation OOM fix (disk-backed storage)
- [x] v3.8.3: Manifest naming cleanup
- [x] **ZERO P0/P1/P2/P3 TECHNICAL DEBT MAINTAINED**

### Quality Verification (Nov 1, 2025)
- [x] Run `make q` (lint + format + mypy + config validation) → ✅ PASS
- [x] Run `make test` (592 tests passing) → ✅ PASS
- [x] Config validation (12 configs) → ✅ 11 valid (exp3 has pre-existing issue)

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
