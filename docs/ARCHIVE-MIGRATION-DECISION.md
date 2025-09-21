# Archive Migration Decision

## Summary
After scanning `/docs_archive/`, I found **critical bug documentation** that wasn't captured in the new `/docs/` structure. I've now added `CRITICAL-ISSUES-RESOLVED.md` to preserve this institutional knowledge.

## What We Found in Archives

### Critical Missing Documentation:
1. **Zero-Seizure Cache Catastrophe** - 254GB wasted, $60+ burned
2. **Focal Loss Double-Counting Bug** - Training instability
3. **PyTorch Multiprocessing Hangs** - WSL2-specific issues
4. **Dataset Label Duration Bug** - 3.9M samples vs 15360 mismatch
5. **Mamba-SSM Installation Issues** - PyTorch version hell

### Already Documented:
- ✅ mysz seizure type crisis (in `tusz-mysz-crisis.md`)
- ✅ Channel naming issues (in `tusz-channels.md`)
- ✅ EDF header repairs (in `tusz-edf-repair.md`)
- ✅ Mamba fallback (mentioned in multiple docs)

## Recommendation: DELETE ARCHIVES

**Reasoning:**
1. We've extracted all critical issues into `CRITICAL-ISSUES-RESOLVED.md`
2. The archive has 15+ bug files with overlapping/redundant content
3. Most issues are marked RESOLVED in `RESOLUTION_STATUS.md`
4. Keeping archives creates confusion about what's current
5. Better to maintain one source of truth in `/docs/`

## Action Plan:
```bash
# We've already captured critical content, so safe to:
rm -rf /home/jj/proj/brain-go-brr-v2/docs_archive

# Current /docs/ now contains:
# - All TUSZ critical documentation
# - CRITICAL-ISSUES-RESOLVED.md with bug history
# - Clean, organized structure
# - No redundancy or confusion
```

## What We're Preserving:
- Zero-seizure cache disaster lessons
- Focal loss configuration gotchas
- WSL2 multiprocessing workarounds
- All mysz seizure type fixes
- Mamba CUDA fallback knowledge

## What We're Discarding:
- Duplicate bug reports (5+ versions of same issues)
- Obsolete phase documentation (replaced by components/)
- Old architecture discussions (superseded)
- Test logs and temporary notes

The new `/docs/` structure with `CRITICAL-ISSUES-RESOLVED.md` captures everything important while being 10x cleaner and more maintainable.