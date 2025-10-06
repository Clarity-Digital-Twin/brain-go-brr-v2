# Active Technical Debt

**Last Updated**: 2025-10-06
**Status**: 🟢 **0 P0/P1/P2 issues** (production-ready)

---

## P3: Minor Cleanup (Non-Blocking)

### P3-1: Manifest Naming Convention Mismatch

**Status**: 🟡 Active (low priority)
**Impact**: Cognitive overhead, no runtime issues
**Effort**: ~2 hours (regenerate manifests + cleanup)

**Problem**:
- Manifests use NPZ-style naming: `"file_windows.npz"`
- Actual cache files are NPY format: `file_data.npy` + `file_labels.npy`
- Code has to strip `"_windows"` suffix in 4+ locations to convert names

**Locations Affected**:
```python
# src/brain_brr/data/cache_utils.py:95
stem = cache_path.stem.replace("_windows", "")

# src/brain_brr/data/datasets.py (3 locations)
stem = cache_path.stem.replace("_windows", "")
file_id = cache_file.stem.replace("_windows", "")
```

**Why It Exists**:
- Migrated NPZ → NPY for memory efficiency (v3.7.0)
- Kept manifest NPZ-style naming for compatibility
- Avoided regenerating 6499 manifests during hot-fix cycle

**Fix Plan**:
1. Regenerate manifests with NPY naming: `"file_data.npy"` entries
2. Remove all `.replace("_windows", "")` logic
3. Update `build_manifest()` to use NPY stems directly

**When to Fix**:
- After v3.8.2 Modal training completes
- During next maintenance window
- NOT urgent - current code works fine

**Benefit**: Cleaner code, less confusion, easier onboarding

---

## P4: Nice-to-Have (Optional)

### P4-1: NumPy Copy Pattern Optimization

**Status**: 🟢 Optional
**Impact**: Micro-performance (~1-2% potential gain)
**Effort**: ~30 minutes

**Current Pattern** (v3.8.2):
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

## P0/P1/P2: Critical/High/Medium

**🎉 ALL RESOLVED** as of v3.8.2:
- ✅ Read-only tensor warnings (P1) → Fixed in v3.8.2
- ✅ GradScaler + scheduler interaction (P2) → Fixed in v3.8.2
- ✅ ValidationDataset NPY mapping (P0) → Fixed in v3.8.1
- ✅ Edge similarity clamping (P1) → Fixed in v3.3.0
- ✅ Gradient explosion (P0) → Fixed in v3.3.1

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
