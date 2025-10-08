# Implementation Plan: Bulletproof Resume System

**Date**: October 7, 2025
**Version**: v3.8.3 → v3.9.0
**Author**: Claude Code
**Philosophy**: Clean Code (Robert C. Martin) - Modular, Testable, Defensive

---

## 🎯 Goals

1. **Fix P0 bugs** (blocking production)
2. **Build bulletproof resume** (survive Modal 24h timeout)
3. **Zero technical debt** (clean, tested, documented)
4. **Production-ready** (graceful degradation, observability)

---

## 📋 Issues to Fix

### P0 (Blocking - Must Fix)
1. **Metric key mismatch bug** - `sensitivity_at_10fa` vs `sensitivity_at_10.0fa` → logs show 0.0000
2. **Non-atomic checkpoints** - Modal kill during save → corrupt .pt files
3. **Incomplete state capture** - Missing scaler/RNG → non-deterministic resume

### P1 (Quality of Life - Should Fix)
4. **No timeout guard** - Lose last hour of progress to hard kill
5. **W&B fragmentation** - New run ID on each resume → broken curves

### P2 (Future Enhancement - Deferred)
6. **True mid-batch resume** - Save/restore DataLoader sampler state for exact batch resumption
   - **Current**: Resume at epoch boundary (lose max 1 epoch ≈ 12h)
   - **Future**: Save sampler state, skip to exact batch_idx (complex, low ROI)
   - **Decision**: Acceptable to lose <12h once per 24h timeout (can add later if needed)

---

## 🏗️ Architecture Design

### Principles
- **Single Responsibility**: Each module does ONE thing well
- **Dependency Inversion**: Core logic doesn't depend on infrastructure
- **Defensive Programming**: Validate inputs, handle errors, fail safely
- **Testability**: Every function has clear inputs/outputs, mockable dependencies

### Module Structure

```
src/brain_brr/train/
├── checkpoints.py          # Checkpoint I/O (atomic saves, full state)
├── metrics_utils.py        # NEW - Metric key normalization
├── timeout_guard.py        # NEW - Wall-clock timeout handling
├── loop.py                 # Training loop (orchestration only)
├── train_step.py           # Training step (unchanged)
└── val_step.py             # Validation step (unchanged)

src/brain_brr/logging/
├── wandb_logger.py         # W&B integration (add run persistence)
└── console.py              # Console logging (unchanged)

tests/unit/train/
├── test_checkpoints.py     # NEW - Atomic saves, state capture
├── test_metrics_utils.py   # NEW - Key normalization
├── test_timeout_guard.py   # NEW - Timeout detection
└── test_integration.py     # NEW - End-to-end resume flow
```

---

## 📐 Detailed Design

### 1. Atomic Checkpoint System

**File**: `src/brain_brr/train/checkpoints.py`

**Requirements**:
- ✅ Atomic writes (temp file + fsync + rename)
- ✅ Full state capture (model, optimizer, scheduler, scaler, RNG)
- ✅ Backward compatible (can load old checkpoints)
- ✅ Versioned state dict (detect schema changes)
- ✅ Corruption detection (optional checksum)

**API Design**:
```python
@dataclass
class CheckpointState:
    """Complete training state for deterministic resume."""
    model_state_dict: dict
    optimizer_state_dict: dict
    epoch: int
    metric: float
    scheduler_state_dict: dict | None = None
    scaler_state_dict: dict | None = None
    rng_state: dict | None = None
    extra: dict | None = None
    version: str = "3.9.0"  # Schema version

def save_checkpoint_atomic(
    state: CheckpointState,
    path: Path,
    validate: bool = True,
) -> None:
    """Save checkpoint atomically with optional validation."""

def load_checkpoint_safe(
    path: Path,
    device: str = "cpu",
    validate: bool = True,
) -> CheckpointState:
    """Load checkpoint with validation and error handling."""
```

**Implementation Strategy**:
1. Create `CheckpointState` dataclass for type safety
2. Implement `atomic_save()` helper (temp + fsync + rename)
3. Update `save_checkpoint()` to use atomic save + full state
4. Update `load_checkpoint()` with validation + backward compat
5. Add optional checksum validation (SHA256)

---

### 2. Metric Key Normalization

**File**: `src/brain_brr/train/metrics_utils.py` (NEW)

**Requirements**:
- ✅ Normalize all FA-related metric keys consistently
- ✅ Handle both integer and float FA rates (10 vs 10.0)
- ✅ Preserve non-FA metrics unchanged
- ✅ Idempotent (multiple normalizations = same result)

**API Design**:
```python
def normalize_metric_key(key: str) -> str:
    """Normalize metric key for consistent lookup.

    Args:
        key: Metric key (e.g., "sensitivity_at_10.0fa")

    Returns:
        Normalized key (e.g., "sensitivity_at_10fa")

    Examples:
        >>> normalize_metric_key("sensitivity_at_10.0fa")
        'sensitivity_at_10fa'
        >>> normalize_metric_key("sensitivity_at_1fa")
        'sensitivity_at_1fa'
        >>> normalize_metric_key("auroc")
        'auroc'
    """

def normalize_metrics_dict(metrics: dict[str, float]) -> dict[str, float]:
    """Normalize all metric keys in dictionary.

    Creates BOTH normalized and original keys for backward compatibility.
    """
```

**Implementation Strategy**:
1. Extract key normalization logic to standalone module
2. Apply normalization in `loop.py` before metric lookup
3. Add tests covering edge cases (int vs float, non-FA metrics)
4. Document expected key formats in `constants.py`

---

### 3. Graceful Timeout Guard

**File**: `src/brain_brr/train/timeout_guard.py` (NEW)

**Requirements**:
- ✅ Detect approaching wall-clock timeout
- ✅ Trigger graceful exit with checkpoint save
- ✅ Configurable safety margin (default 10 min)
- ✅ Optional timeout callback (for cleanup)

**API Design**:
```python
class TimeoutGuard:
    """Monitor wall-clock time and trigger graceful exit."""

    def __init__(
        self,
        limit_seconds: int | None,
        safety_margin_seconds: int = 600,
        on_timeout: Callable[[], None] | None = None,
    ):
        """Initialize timeout guard.

        Args:
            limit_seconds: Wall-clock timeout (None = no limit)
            safety_margin_seconds: Exit N seconds before timeout
            on_timeout: Optional callback when timeout imminent
        """
        self.limit = limit_seconds
        self.margin = safety_margin_seconds
        self.start_time = time.time()
        self.callback = on_timeout

    def check(self) -> bool:
        """Check if timeout is imminent.

        Returns:
            True if should exit gracefully
        """
        if self.limit is None:
            return False
        elapsed = time.time() - self.start_time
        imminent = elapsed >= (self.limit - self.margin)
        if imminent and self.callback:
            self.callback()
        return imminent

    def remaining_seconds(self) -> float | None:
        """Get remaining time before timeout."""
```

**Implementation Strategy**:
1. Create `TimeoutGuard` class with clean API
2. Integrate into `loop.py` training loop
3. Add logging when timeout detected
4. Support optional cleanup callback
5. Test with mock time functions

---

### 4. W&B Run Persistence

**File**: `src/brain_brr/logging/wandb_logger.py`

**Requirements**:
- ✅ Persist W&B run ID to checkpoint dir
- ✅ Resume existing run on restart
- ✅ Handle missing/corrupt run ID gracefully
- ✅ Log run ID for debugging

**API Design**:
```python
def init_wandb_with_resume(
    config: TrainingConfig,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> Run:
    """Initialize W&B with automatic run persistence.

    Args:
        config: Training configuration
        checkpoint_dir: Directory for .wandb_run_id file
        resume: Whether to resume existing run

    Returns:
        W&B run object
    """
```

**Implementation Strategy**:
1. Save run ID to `.wandb_run_id` after init
2. Load run ID on resume if file exists
3. Pass `id=` and `resume="allow"` to `wandb.init()`
4. Add error handling for corrupt/missing run ID
5. Test with mock W&B API

---

## 🔨 Implementation Phases

### Phase 1: Core Fixes (P0 - 1 hour)

**Goal**: Fix blocking bugs, make resume bulletproof

**Tasks**:
1. ✅ Create `metrics_utils.py` with key normalization (15 min)
2. ✅ Update `checkpoints.py` with atomic saves + full state (30 min)
3. ✅ Update `loop.py` to use normalized metrics (5 min)
4. ✅ Update `train_step.py` to pass scaler to checkpoint save (5 min)
5. ✅ Test atomic save under kill signal (10 min)

**Success Criteria**:
- Logs show correct sensitivity values (not 0.0000)
- Checkpoint survives process kill during save
- Resume restores identical RNG sequence

---

### Phase 2: Robustness (P1 - 1.5 hours)

**Goal**: Graceful degradation, observability

**Tasks**:
1. ✅ Create `timeout_guard.py` with TimeoutGuard class (30 min)
2. ✅ Integrate timeout guard into `loop.py` (15 min)
3. ✅ Add W&B run persistence to `wandb_logger.py` (30 min)
4. ✅ Add logging for all state transitions (15 min)

**Success Criteria**:
- Training exits gracefully 10 min before Modal timeout
- W&B curves are continuous across resumes
- All state transitions logged clearly

---

### Phase 3: Testing (1 hour)

**Goal**: Comprehensive test coverage

**Tasks**:
1. ✅ Unit tests for `metrics_utils.py` (15 min)
2. ✅ Unit tests for `checkpoints.py` (20 min)
3. ✅ Unit tests for `timeout_guard.py` (15 min)
4. ✅ Integration test for full resume flow (10 min)

**Success Criteria**:
- 100% coverage for new modules
- Integration test passes end-to-end
- All edge cases covered

---

### Phase 4: Documentation (30 min)

**Goal**: Clear migration guide, runbook

**Tasks**:
1. ✅ Update `CLAUDE.md` with new modules (10 min)
2. ✅ Create migration guide for existing checkpoints (10 min)
3. ✅ Update Modal deployment docs (10 min)

**Success Criteria**:
- Developers understand how to use new system
- Migration path is clear
- Modal setup documented

---

## 🧪 Testing Strategy

### Unit Tests

**`test_metrics_utils.py`**:
```python
def test_normalize_metric_key_with_decimal():
    assert normalize_metric_key("sensitivity_at_10.0fa") == "sensitivity_at_10fa"

def test_normalize_metric_key_without_decimal():
    assert normalize_metric_key("sensitivity_at_10fa") == "sensitivity_at_10fa"

def test_normalize_metric_key_non_fa_metric():
    assert normalize_metric_key("auroc") == "auroc"

def test_normalize_metrics_dict_creates_both_keys():
    result = normalize_metrics_dict({"sensitivity_at_10.0fa": 0.5})
    assert "sensitivity_at_10fa" in result
    assert "sensitivity_at_10.0fa" in result  # Backward compat
```

**`test_checkpoints.py`**:
```python
def test_atomic_save_survives_kill(tmp_path):
    # Save checkpoint, kill during write, verify file intact

def test_load_checkpoint_validates_version():
    # Load checkpoint with old schema, verify backward compat

def test_checkpoint_captures_full_state():
    # Verify scaler + RNG states are saved/restored
```

**`test_timeout_guard.py`**:
```python
def test_timeout_guard_triggers_at_margin(monkeypatch):
    # Mock time, verify guard triggers at correct moment

def test_timeout_guard_calls_callback():
    # Verify cleanup callback is called
```

### Integration Tests

**`test_integration.py`**:
```python
def test_full_resume_flow(tmp_path):
    """End-to-end: train → kill → resume → verify identical state."""
    # 1. Train for N batches
    # 2. Save checkpoint with full state
    # 3. Simulate Modal kill
    # 4. Resume from checkpoint
    # 5. Verify next batch is identical (RNG restored)
```

---

## 🚀 Deployment Plan

### Pre-deployment Checklist
- [ ] All unit tests pass (`make test`)
- [ ] Integration test passes
- [ ] Linting passes (`make q`)
- [ ] Type checking passes (`mypy`)
- [ ] Smoke test runs successfully
- [ ] Documentation updated

### Deployment Steps

1. **Merge to development branch**
   ```bash
   git checkout development
   git pull origin development
   git checkout -b fix/bulletproof-resume
   ```

2. **Run full test suite**
   ```bash
   make test
   make q
   ```

3. **Smoke test locally**
   ```bash
   make s  # 3 files, 1 epoch
   # Verify logs show correct sensitivity values
   ```

4. **Deploy to Modal**
   ```bash
   # Test with 50-file smoke test first
   modal run --detach deploy/modal/app.py --action train \
     --config configs/modal/smoke.yaml

   # Monitor logs for correct metrics
   modal app logs <app-id>
   ```

5. **Full training run**
   ```bash
   modal run --detach deploy/modal/app.py --action train \
     --config configs/modal/train.yaml
   ```

6. **Test resume after timeout**
   ```bash
   # Wait for Modal to kill at 24h
   # Then resume:
   modal run --detach deploy/modal/app.py --action train \
     --config configs/modal/train.yaml --resume true
   ```

---

## 🛡️ Rollback Plan

### If Critical Bug Found

1. **Stop Modal training immediately**
   ```bash
   modal app stop <app-id>
   ```

2. **Revert to previous version**
   ```bash
   git checkout v3.8.3
   ```

3. **Verify old checkpoints still work**
   ```bash
   # Load checkpoint with old code
   python scripts/verify_checkpoint.py checkpoints/mid_epoch_002_000334.pt
   ```

4. **Document issue in GitHub**
   - Create issue with reproduction steps
   - Tag as P0
   - Assign to team

### Backward Compatibility

- **New code MUST load old checkpoints** (v3.8.3)
- **New checkpoints SHOULD work with old code** (optional fields)
- **Schema version in checkpoint** (detect incompatibilities)

---

## 📊 Success Metrics

### Immediate (After Phase 1)
- ✅ Logs show `New best sensitivity_at_10fa: 0.1643` (not 0.0000)
- ✅ Checkpoint file survives kill signal
- ✅ Resume produces identical batch sequence

### Short-term (After Phase 2)
- ✅ Training exits gracefully at 23h 50min
- ✅ W&B shows continuous curves across resumes
- ✅ Zero checkpoint corruption incidents

### Long-term (After 100 epochs)
- ✅ Training completes all 100 epochs
- ✅ Total manual intervention <2 hours
- ✅ Final metrics match or exceed baseline

---

## 📚 Migration Guide

### For Existing Checkpoints

**Old checkpoints (v3.8.3) are compatible** - no migration needed.

**To use new features**:
1. Resume from old checkpoint once
2. New checkpoint will include full state (scaler + RNG)
3. Subsequent resumes will be deterministic

### For New Training Runs

**No changes required** - new system is transparent:
1. Start training normally
2. When Modal times out, resume with `--resume true`
3. Training picks up exactly where it stopped

### For Developers

**Using new checkpoint API**:
```python
from src.brain_brr.train.checkpoints import CheckpointState, save_checkpoint_atomic

# Create state
state = CheckpointState(
    model_state_dict=model.state_dict(),
    optimizer_state_dict=optimizer.state_dict(),
    epoch=epoch,
    metric=metric,
    scaler_state_dict=scaler.state_dict() if scaler else None,
)

# Save atomically
save_checkpoint_atomic(state, checkpoint_path)
```

**Using timeout guard**:
```python
from src.brain_brr.train.timeout_guard import TimeoutGuard

guard = TimeoutGuard(limit_seconds=23*3600, safety_margin_seconds=600)

for epoch in range(epochs):
    if guard.check():
        logger.warning("Timeout imminent, exiting gracefully")
        save_checkpoint(...)
        break
```

---

## 🔍 Observability

### Logging Strategy

**All state transitions logged**:
- Checkpoint saves (with size, duration)
- Checkpoint loads (with validation status)
- Timeout warnings (with remaining time)
- W&B run ID persistence
- Metric key normalization

**Example logs**:
```
[CHECKPOINT] Saving atomic checkpoint: mid_epoch_002_000334.pt (125MB)
[CHECKPOINT] fsync complete, atomic rename successful (took 3.2s)
[METRICS] Normalized 4 metric keys: sensitivity_at_10.0fa → sensitivity_at_10fa
[WANDB] Persisted run ID: xyz123abc to .wandb_run_id
[TIMEOUT] Wall-clock: 23h 50m / 24h 00m - exiting gracefully
[CHECKPOINT] Saved timeout_exit.pt, safe to resume
```

### Monitoring Dashboards

**W&B Dashboard**:
- Training loss (continuous across resumes)
- Validation metrics (continuous)
- Checkpoint save frequency
- Resume events (annotated)

**Modal Logs**:
- Timeout warnings
- Checkpoint corruption incidents
- Resume success/failure rate

---

## 📝 Checklist

### Phase 1: Core Fixes (P0)
- [ ] Create `src/brain_brr/train/metrics_utils.py`
- [ ] Implement `normalize_metric_key()` function
- [ ] Implement `normalize_metrics_dict()` function
- [ ] Update `checkpoints.py` with `atomic_save()` helper
- [ ] Update `checkpoints.py` with `CheckpointState` dataclass
- [ ] Update `save_checkpoint()` to capture scaler + RNG
- [ ] Update `load_checkpoint()` with validation
- [ ] Update `loop.py` to normalize metrics before lookup
- [ ] Update `train_step.py` to pass scaler to save
- [ ] Write unit tests for `metrics_utils.py`
- [ ] Write unit tests for `checkpoints.py`
- [ ] Run smoke test, verify logs show correct values

### Phase 2: Robustness (P1)
- [ ] Create `src/brain_brr/train/timeout_guard.py`
- [ ] Implement `TimeoutGuard` class
- [ ] Integrate timeout guard into `loop.py`
- [ ] Update `wandb_logger.py` with run persistence
- [ ] Write unit tests for `timeout_guard.py`
- [ ] Write integration test for full resume flow
- [ ] Test graceful exit on Modal (kill at 23h 50m)

### Phase 3: Documentation
- [ ] Update `CLAUDE.md` with new modules
- [ ] Create migration guide
- [ ] Update Modal deployment docs
- [ ] Add troubleshooting section
- [ ] Update `ARCHITECTURE_EVOLUTION.md`

### Phase 4: Deployment
- [ ] Run full test suite locally
- [ ] Smoke test on Modal (50 files)
- [ ] Monitor first resume event
- [ ] Full 100-epoch run
- [ ] Post-deployment validation

---

## 🎓 Lessons Learned (Pre-emptive)

### What Could Go Wrong

1. **Old checkpoints break** → Solution: Backward compatibility testing
2. **Timeout guard misfires** → Solution: Configurable safety margin
3. **W&B run ID corruption** → Solution: Graceful fallback to new run
4. **Metric normalization breaks something** → Solution: Preserve original keys

### Defensive Programming

- **Always validate inputs** (checkpoint paths, metric keys)
- **Always handle errors** (corrupt files, missing keys)
- **Always log state transitions** (debugging resume issues)
- **Always provide fallbacks** (missing scaler → warning, not crash)

---

## 🚀 Let's Ship It!

**Total time estimate**: 3.5 hours
- Phase 1 (P0): 1 hour
- Phase 2 (P1): 1.5 hours
- Phase 3 (Testing): 1 hour

**Ready to start?** We'll go phase by phase, test thoroughly, and ship production-ready code.

**First step**: Implement Phase 1 (Core Fixes) - want to proceed? 💪
