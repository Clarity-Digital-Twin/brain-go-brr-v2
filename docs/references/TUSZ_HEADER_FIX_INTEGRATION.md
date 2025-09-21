# TUSZ Header Fix Integration — Summary

Last updated: 2025-09-21

Where it hooks in
- EDF load path in `src/brain_brr/data/io.py` attempts standard read, then repairs known header issue and retries.

Impact
- No changes to model code; only data loading robustness improved
- Prevents data loss by skipping malformed-but-fixable EDFs

Testing
- Unit-level tests cover the parsing/utilities; integration tests recommended on a small real subset.

