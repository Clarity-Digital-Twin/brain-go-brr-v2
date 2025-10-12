# Archive: Historical Documentation

**Status**: ✅ All issues RESOLVED
**Archive Date**: October 12, 2025
**Context**: Documents from v3.9.0-v3.10.0 development

## Contents

This archive contains incident reports and bug investigations from the v3.9.x release series. All issues documented here have been **resolved** and fixes are integrated into the codebase (see `docs/09-development/bug-tracker.md` and `docs/09-development/technical-debt.md` for the live status).

### Files

1. **BUG_HUNT_2025-10-09.md** - Comprehensive bug audit after config separation
   - Status: ✅ All P0/P1/P2/P3 issues fixed
   - Key fixes: Config references updated, tests fixed, docs aligned

2. **CONFIG_ARCHITECTURE_SEPARATION.md** - Strategy for BiMamba2/FLA config split
   - Status: ✅ Implemented in v3.9.2
   - Result: Clean separation with `train_bimamba.yaml` and `train_fla.yaml`

3. **MATHEMATICAL_FORMULATIONS_AND_PRIORITIES.md** - Comprehensive reference
   - Status: ✅ Content extracted to main docs
   - See: docs/06-evaluation/metrics-and-taes.md, docs/04-model/

4. **REMAINING_ISSUES_INVESTIGATION.md** - P2.2 and P2.3 analysis
   - Status: ✅ Both resolved or working-as-designed
   - P2.2: MinimalTCN is production (not pytorch-tcn)
   - P2.3: Migration tests enabled and passing

## Why Archived?

These documents capture the **journey** of stabilizing v3.9.x but are no longer active development concerns. They're preserved for:
- Historical context
- Incident learning
- Future reference

## Current Documentation

For up-to-date information, see:
- Main docs: `/docs/README.md`
- Current status: `STATUS.md`
- Release notes: `RELEASE_NOTES.md`
