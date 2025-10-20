# Active Technical Debt (v3.9.0)

**⚠️ HISTORICAL SNAPSHOT** - Captured state as of v3.9.0 (October 8, 2025)

**Status at v3.9.0**: 🟢 **0 P0/P1/P2/P3 issues** (all debt resolved!)

**Current Status**: See `docs/09-development/technical-debt.md` for latest debt tracking

---

This document captures the technical debt state at v3.9.0 when zero P0-P3 issues remained. It's preserved for historical reference showing the debt resolution journey from v3.0.0 → v3.9.0.

---

# SNAPSHOT: v3.9.0 Technical Debt State

---

## P3: Minor Cleanup (Non-Blocking)

**🎉 ALL RESOLVED** as of v3.9.0

---

## P4: Nice-to-Have (Optional)

### P4-1: NumPy Copy Pattern Optimization

**Status**: 🟢 Optional
**Impact**: Micro-performance (~1-2% potential gain)
**Effort**: ~30 minutes

**Current Pattern** (v3.9.0):
```python
torch.from_numpy(np.array(window, copy=True, dtype=np.float32, order="C"))
```

**Alternative Pattern**:
```python
torch.as_tensor(window).contiguous().to(torch.float32)
```

**Trade-offs**:
- Current: Explicit copy on NumPy side (clear intent)
- Alternative: PyTorch handles copy (slightly more efficient)
- Both eliminate read-only warnings equally well

**Decision**: Keep current pattern (clarity > micro-perf)

---

## P0/P1/P2/P3: Resolved Issues

**🎉 ALL RESOLVED**:
- ✅ **P3-1**: Manifest naming mismatch → Fixed in v3.8.3 (eliminated 11 `.replace()` workarounds)
- ✅ **P1**: Read-only tensor warnings → Fixed in v3.8.2
- ✅ **P2**: GradScaler + scheduler interaction → Fixed in v3.8.2
- ✅ **P0**: ValidationDataset NPY mapping → Fixed in v3.8.1
- ✅ **P1**: Edge similarity clamping → Fixed in v3.3.0
- ✅ **P0**: Gradient explosion → Fixed in v3.3.1

See `CHANGELOG.md` for full resolution history.

---

## How to Add New Debt

1. **Classify priority**:
   - P0: Breaks training/inference (URGENT)
   - P1: Degraded quality/warnings (HIGH)
   - P2: Maintenance burden (MEDIUM)
   - P3: Minor cleanup (LOW)
   - P4: Nice-to-have (OPTIONAL)

2. **Document clearly**:
   - Problem statement
   - Root cause
   - Impact assessment
   - Fix plan
   - When to address

3. **Link to code**: File paths + line numbers

4. **Update status**: When resolved, move to bottom section
