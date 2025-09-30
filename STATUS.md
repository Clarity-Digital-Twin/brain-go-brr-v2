# Brain-Go-Brr V3.3.0 - Current Status

**Last Updated**: 2025-09-30 12:17 UTC
**Branch**: `fix/upgrade-mamba`
**Version**: v3.3.0 (PyTorch 2.5.0 + mamba-ssm 2.2.5 + PR #708 patch)

---

## 🧪 TESTING IN PROGRESS

### Current Training Runs

| Environment | Config | Status | Progress | Started |
|-------------|--------|--------|----------|---------|
| **Modal A100** | smoke.yaml | 🟡 RUNNING | Batch 50/114 (~44%) | 11:15 UTC |
| **Local RTX 4090** | train.yaml | 🟡 STARTING | Building dev index | 12:17 UTC |

**Purpose**: Validate PR #708 fix for XID 31 MMU Fault on A100

---

## 🔧 Recent Changes (2025-09-30)

### Dependency Stack Upgrade ✅
- **PyTorch**: 2.2.2 → 2.5.0+cu124
- **mamba-ssm**: 2.2.2 → 2.2.5
- **causal-conv1d**: 1.4.0 → 1.5.2
- **torch-geometric**: 2.6.1 (unchanged)
- **CUDA**: 12.1 → 12.4

### PR #708 Patch Applied ✅
- **What**: Manual application of unmerged mamba-ssm PR #708
- **Why**: Fixes int32 pointer overflow in Triton kernels (causes XID 31)
- **Where**: Local `.venv` + Modal image
- **Status**: Testing in progress (not yet validated)

### Test Suite Refactor ✅
- **Commits**: 6cf555f, 13b50d3, 6e71240, 9fcb823, 3ac753b
- **Changes**: 27 files (+3,154/-107 lines)
- **Improvements**:
  - ✅ 65 GPU tests properly marked with `@pytest.mark.gpu`
  - ✅ 0 hardcoded batch sizes (all use `test_batch_size` fixture)
  - ✅ Single cleanup fixture (no duplication)
  - ✅ Configurable GPU memory (BGB_TEST_GPU_FRACTION)
  - ✅ Comprehensive documentation (MARKER_POLICY.md, GPU_ADJUSTMENTS.md)
- **Tests**: 445 passing (380 CPU-safe, 65 GPU, 18 performance)

---

## 🚨 Known Issues

### XID 31 on Modal A100 (INVESTIGATING)
- **Symptom**: GPU crashes with "XID 31 MMU Fault" during first large batch
- **Root Cause**: mamba-ssm CUDA kernel pointer overflow (int32 → int64)
- **Fix**: PR #708 (unmerged) applied manually
- **Status**: 🟡 Testing in progress (Modal smoke at 44%)
- **Evidence**: Issue #732 (A6000) shows identical pattern

### Local Training (RTX 4090)
- **Status**: ✅ STABLE - no XID 31 (Ada architecture uses different kernels)
- **Config**: 100 epochs, 4667 train files, 1832 dev files, batch_size=12
- **Current**: Building dev dataset index (11/1832 files)

---

## 📊 What We Know

### Test Results Matrix

| Test | Environment | Result | Evidence |
|------|-------------|--------|----------|
| Smoke (50 files, batch=64) | Modal A100 | 🟡 RUNNING | Currently at 44% |
| Full train (4667 files, batch=64) | Modal A100 | ❌ XID 31 | Failed at preflight (2025-09-30 07:50) |
| Smoke (3 files, batch=12) | Local RTX 4090 | ✅ PASS | Completed multiple times |
| Full train (4667 files, batch=12) | Local RTX 4090 | 🟡 STARTING | Just started (12:17 UTC) |

### Key Findings

1. **mamba-ssm 2.2.5 did NOT fix XID 31** - bug still present
2. **PR #708 is the correct fix** - targets exact issue (int32 overflow)
3. **Smoke test passes** - smaller dataset falls below bug threshold
4. **Full training fails** - large batch after cold GPU start triggers bug
5. **RTX 4090 works** - different architecture (Ada vs Ampere)

---

## 🎯 Success Criteria

### For PR #708 Validation

- [ ] Modal smoke test completes without XID 31 (~30 min remaining)
- [ ] Local full training runs stably (first batch critical, ~1 hour)
- [ ] Loss converges normally (no NaN/inf)
- [ ] No GPU faults in logs (`nvidia-smi`, `dmesg`)

### For Full Training Approval

- [ ] Modal smoke test validates PR #708
- [ ] Run Modal full training with 4667 files
- [ ] Complete at least 10 epochs without crash
- [ ] Compare metrics to baseline (if available)

---

## 📚 Documentation Status

### Test Suite ✅
- `tests/TEST_SUITE_CURRENT_STATE.md` - As-built documentation
- `tests/MARKER_POLICY.md` - GPU marker enforcement policy
- `tests/GPU_ADJUSTMENTS.md` - RTX 4090 adjustments + env vars
- `docs/archive/TEST_SUITE_*` - Historical planning docs (ARCHIVED)

### Dependencies 🟡
- `MAMBA_UPGRADE_ANALYSIS.md` - Upgrade decision context (ARCHIVED)
- `MODAL_XID31_RECURRENCE.md` - Current investigation (TESTING IN PROGRESS)
- `PR708_APPLICATION.md` - PR #708 application guide (TESTING IN PROGRESS)
- `docs/archive/STACK_UPGRADE_PLAN_V3.md` - Stack upgrade details (COMPLETED)

### Architecture ✅
- `CLAUDE.md` - Project overview + quick commands
- `INSTALLATION.md` - Setup guide (updated for PyTorch 2.5.0)
- `README.md` - Project introduction
- `CHANGELOG.md` - Version history

---

## ⚠️ What's NOT Proven Yet

1. **PR #708 efficacy** - Applied but not validated (testing in progress)
2. **Modal A100 stability** - Smoke test must complete first
3. **Full training success** - Need 10+ epochs without crash
4. **Mamba CUDA fix** - PR #708 may not be sufficient (fallback plan needed)

---

## 🔮 Next Steps

### Immediate (Next 1 hour)
1. Monitor Modal smoke test completion (~30 min remaining)
2. Monitor local training startup (~1 hour to first batch)
3. Check for any XID 31 errors in logs

### If Modal Smoke Passes (✅)
1. **RUN MODAL FULL TRAINING** - 4667 files, batch=64
2. Monitor for first 10 epochs
3. If stable → continue to 100 epochs
4. Update docs with "PR #708 validated"

### If Modal Smoke Fails (❌)
1. Analyze failure mode (same XID 31 or different?)
2. Consider alternative fixes:
   - Reduce batch size (64 → 32)
   - Disable mamba-ssm fallback to Conv1d
   - Try L40S GPU (Ada architecture like RTX 4090)
3. Report to mamba-ssm maintainers

### If Local Training Fails (❌)
1. PR #708 patch corrupted or incompatible
2. Re-apply patch carefully
3. Run local smoke test (3 files) to isolate
4. Check `.venv` integrity

---

## 📞 Critical Files to Monitor

### During Testing
- `tmux attach -t train` - Local training logs
- Modal App Logs - Modal smoke test progress
- `nvidia-smi` - GPU health check
- `dmesg | grep XID` - Hardware error logs

### After Testing
- `results/smoke/` - Modal smoke outputs
- `results/train/` - Local training outputs
- Git branches - Ensure all synced (fix/upgrade-mamba, development, main)

---

## 💡 Key Insights

1. **Mamba-SSM is fragile** - CUDA kernel bugs persist across versions
2. **First-batch bug** - Cold GPU + large batch = pointer overflow
3. **Hardware matters** - A100/A6000 (Ampere) fails, RTX 4090 (Ada) works
4. **PR #708 is unmerged** - We're running a patched fork
5. **Test suite is solid** - 445 tests, comprehensive coverage

---

**Status**: 🟡 HOPEFUL - PR #708 should fix XID 31, but not yet validated

**Risk Level**: 🟠 MEDIUM - If Modal smoke fails, need alternative approach

**Confidence**: 🎯 70% - Evidence strongly supports PR #708, but empirical validation pending