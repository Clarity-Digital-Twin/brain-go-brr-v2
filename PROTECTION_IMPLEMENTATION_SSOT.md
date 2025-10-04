# Protection System - Single Source of Truth Implementation Plan

**Date**: October 4, 2025
**Status**: 🔴 **CRITICAL - IMPLEMENTATION REQUIRED**
**Severity**: Documentation claims features that don't exist
**Impact**: False sense of security, wasted debugging time, confused users

---

## Executive Summary

**THE PROBLEM**: We documented and promoted a "3-tier NaN protection system" with 4 protection flags (`BGB_SANITIZE_GRADS`, `BGB_SANITIZE_INPUTS`, `BGB_SKIP_OPT_STEP_ON_NAN`, `BGB_SAFE_CLAMP`) that are **COMPLETELY UNUSED** in the codebase. Hundreds of documentation references claim these are "REQUIRED", "CRITICAL", "MANDATORY" — but they do nothing.

**THE GOOD NEWS**: Training works fine because:
1. PyTorch's `torch.nn.utils.clip_grad_norm_()` provides real gradient protection
2. Architectural safeguards (LayerNorm, clamping) are hardcoded and working
3. Preprocessing clips outliers effectively

**THE DECISION**: We will implement a **DeepMind-grade, industrial-strength protection system** with:
- Clear separation of concerns
- Minimal performance overhead
- Comprehensive testing
- No false advertising

---

## Part 1: Current State Analysis

### What Documentation Claims

From `docs/08-operations/nan-prevention-complete.md` and 100+ other files:

**"3-Tier NaN Protection System"**:
1. **Environment Variables** - `BGB_SANITIZE_GRADS=1`, `BGB_SANITIZE_INPUTS=1`, `BGB_SAFE_CLAMP=1`
2. **Component-Level** - Architectural safeguards (LayerNorm, residual scaling)
3. **Output-Level** - Final sanitization before loss

**Claims**:
- "`BGB_SANITIZE_GRADS=1` is **REQUIRED** for PyTorch 2.5.0 training"
- "Training **fails completely** without `BGB_SANITIZE_GRADS=1`"
- Modal automatically sets these "for NaN protection"
- "Defense-in-depth" strategy

### What Code Actually Does

**Environment Variables** (`src/brain_brr/utils/env.py`):
- ✅ Defined: `_SANITIZE_GRADS`, `_SANITIZE_INPUTS`, `_SKIP_OPT_STEP_ON_NAN`, `_SAFE_CLAMP`
- ❌ **NEVER USED**: Zero references in training code
- ✅ Modal sets them: `deploy/modal/app.py:713-714`
- ❌ **NO EFFECT**: Flags are read, then ignored

**Actual Protection** (`src/brain_brr/train/train_step.py:194,199`):
```python
# THIS is what actually works:
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
# gradient_clip = 0.5 (from config, always applied)
```

**Architectural Safeguards** (ACTUALLY WORKING):
- LayerNorm at 5 boundaries ✅ (via config, always enabled)
- Edge feature clamping ✅ (hardcoded, not controlled by flags)
- Preprocessing outlier clipping ✅ (hardcoded to ±10σ)
- Detached eigenvectors ✅ (v3.3.1 fix, hardcoded)

---

## Part 2: Industry Best Practices

### What Google DeepMind Does

**Source**: [TensorFlow Best Practices](https://www.tensorflow.org/guide/effective_tf2), [JAX Training Loop](https://github.com/google/jax/blob/main/docs/notebooks/neural_network_with_tfds_data.ipynb)

**Gradient Protection**:
1. **Always clip gradients** (global norm, not per-parameter)
2. **Log pre-clip norm** for monitoring (not post-clip)
3. **Optional gradient sanitization** for research/debugging only
4. **Never skip optimizer steps** (breaks learning rate schedules)

**Activation Protection**:
1. **Use LayerNorm/BatchNorm liberally** (every 2-3 layers)
2. **Avoid manual clamping** (indicates architectural problem)
3. **Use mixed precision correctly** (torch.amp.autocast, not manual FP16)

**Environment Variables**:
1. **Minimal use** (only for debugging, not core training)
2. **Default to safe** (flags should enable risky behavior, not safety)
3. **Clear naming** (`DEBUG_*`, `EXPERIMENTAL_*`)

### What PyTorch Recommends

**Source**: [PyTorch Training Loop](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html), [AMP Documentation](https://pytorch.org/docs/stable/amp.html)

**Gradient Handling**:
```python
# STANDARD PATTERN (no sanitization):
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()

# WITH MIXED PRECISION:
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # BEFORE clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

**Key Points**:
- `clip_grad_norm_()` **always** safe to use
- Returns **pre-clip norm** for monitoring
- No sanitization needed if clipping is applied
- Mixed precision scaler handles inf/nan gracefully

---

## Part 3: Decision Matrix

### Question 1: Should we implement gradient sanitization?

**Arguments FOR**:
- ✅ Adds defense-in-depth
- ✅ Useful for debugging (log where NaNs occur)
- ✅ Documentation already promises it
- ✅ Zero performance overhead if disabled
- ✅ Can help with extreme FP16 overflows

**Arguments AGAINST**:
- ❌ PyTorch clipping already sufficient
- ❌ Adds complexity for marginal benefit
- ❌ Not industry standard practice
- ❌ Masks underlying bugs

**VERDICT**: ✅ **IMPLEMENT** (but make it opt-in, disabled by default)

**Reasoning**:
- Useful for debugging and research
- Can be disabled in production
- Helps justify documentation claims
- **BUT**: Change default to `0` (off), make docs say it's optional for debugging

---

### Question 2: Should we implement input sanitization?

**Arguments FOR**:
- ✅ Could catch bad data from preprocessing
- ✅ Documentation promises it

**Arguments AGAINST**:
- ❌ Preprocessing already clips outliers to ±10σ
- ❌ Model has LayerNorm at all boundaries
- ❌ Would mask data quality issues
- ❌ Performance overhead (extra tensor checks every batch)
- ❌ Not standard practice (data should be clean)

**VERDICT**: ❌ **REMOVE**

**Reasoning**:
- Preprocessing should guarantee clean data
- Architectural safeguards already handle outliers
- Sanitizing inputs masks data quality problems
- **Action**: Remove flag, update docs

---

### Question 3: Should we implement skip-optimizer-step-on-NaN?

**Arguments FOR**:
- ✅ Prevents NaN propagation to parameters
- ✅ Some frameworks do this (TensorFlow)

**Arguments AGAINST**:
- ❌ Breaks learning rate schedules (scheduler steps anyway)
- ❌ Causes loss curve discontinuities
- ❌ PyTorch gradient clipping prevents NaN grads
- ❌ If loss is NaN, whole batch is problematic anyway
- ❌ Not PyTorch standard practice

**VERDICT**: ❌ **REMOVE**

**Reasoning**:
- Current behavior (log NaN but continue) is correct
- Skipping breaks training dynamics
- Gradient clipping prevents NaN parameters anyway
- **Action**: Remove flag, update docs

---

### Question 4: Should we implement safe-clamp?

**Arguments FOR**:
- ✅ Extra safety layer
- ✅ Could help debugging

**Arguments AGAINST**:
- ❌ LayerNorm already handles activation magnitudes
- ❌ Hardcoded clamping exists where needed (edge features)
- ❌ Configurable clamping adds complexity
- ❌ If activations are exploding, architecture is broken
- ❌ Performance overhead (clamp every activation)

**VERDICT**: ❌ **REMOVE**

**Reasoning**:
- LayerNorm is the correct solution for activation magnitudes
- Clamping should be surgical (edge features), not global
- **Action**: Remove flag, keep hardcoded clamping where it exists

---

## Part 4: Final Decision Summary

| Flag | Decision | Reason | Default | Use Case |
|------|----------|--------|---------|----------|
| `BGB_SANITIZE_GRADS` | ✅ **IMPLEMENT** | Debugging tool, defense-in-depth | **`0` (OFF)** | Research, debugging FP16 issues |
| `BGB_SANITIZE_INPUTS` | ❌ **REMOVE** | Masks data quality issues | N/A | None - use better preprocessing |
| `BGB_SKIP_OPT_STEP_ON_NAN` | ❌ **REMOVE** | Breaks training dynamics | N/A | None - log and continue is correct |
| `BGB_SAFE_CLAMP` | ❌ **REMOVE** | LayerNorm is better solution | N/A | None - use architectural fixes |

**Key Changes**:
1. **`BGB_SANITIZE_GRADS`** changes from "REQUIRED" to "optional debugging tool"
2. **Remove 3 flags entirely** - clean up env.py
3. **Fix ALL documentation** - remove false claims
4. **Update Modal setup** - clarify what actually protects training

---

## Part 5: Industrial-Strength Implementation

### Design Principles

1. **Separation of Concerns**:
   - Gradient monitoring: separate from sanitization
   - Sanitization: optional layer, not core protection
   - Core protection: gradient clipping (always on)

2. **Performance**:
   - Zero overhead when disabled (compiled away)
   - Minimal overhead when enabled (single pass over grads)
   - No redundant checks

3. **Observability**:
   - Log pre-clip gradient norms (current behavior)
   - Log post-clip gradient norms (new)
   - Warn when sanitization occurs (rare event)

4. **Testability**:
   - Unit tests for sanitization logic
   - Integration tests with injected NaN
   - Performance benchmarks

### Implementation: Gradient Sanitization

**Location**: `src/brain_brr/train/train_step.py`

**Current Code** (lines 191-200):
```python
if scaler.is_enabled():
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    scaler.step(optimizer)
    scaler.update()
else:
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    optimizer.step()
```

**New Code** (DeepMind-grade):
```python
# Backward pass
if scaler.is_enabled():
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
else:
    loss.backward()

# Gradient sanitization (optional, for debugging)
if env.sanitize_grads():
    sanitized_count = _sanitize_gradients(model, logger, batch_idx)
    if sanitized_count > 0:
        logger.warning(
            f"[GRAD_SANITIZE] Replaced {sanitized_count} non-finite gradients "
            f"at batch {batch_idx} (consider investigating root cause)"
        )

# Gradient clipping (ALWAYS applied)
pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

# Optimizer step
if scaler.is_enabled():
    scaler.step(optimizer)
    scaler.update()
else:
    optimizer.step()

# Monitoring (improved)
gradient_norms.append(float(pre_clip_norm))
if batch_idx % LOG_EVERY_N_STEPS == 0 and pre_clip_norm > gradient_clip * 2:
    # Calculate post-clip norm for comparison
    post_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    logger.debug(
        f"[GRAD_CLIP] Batch {batch_idx}: "
        f"pre={pre_clip_norm:.2f} → post={post_clip_norm:.2f} "
        f"(clipped {(pre_clip_norm - post_clip_norm) / pre_clip_norm * 100:.1f}%)"
    )
```

**Helper Function** (new, in same file):
```python
def _sanitize_gradients(
    model: nn.Module,
    logger: logging.Logger,
    batch_idx: int,
) -> int:
    """Sanitize non-finite gradients to zero.

    Args:
        model: Model with gradients to sanitize
        logger: Logger for warnings
        batch_idx: Current batch number

    Returns:
        Number of parameters with sanitized gradients

    Note:
        This is a DEBUGGING TOOL, not core protection.
        Gradient clipping is the primary protection mechanism.
    """
    sanitized_count = 0

    for param in model.parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                # Count how many values were non-finite
                n_nonfinite = (~torch.isfinite(param.grad)).sum().item()
                sanitized_count += 1

                # Replace with zeros (in-place)
                param.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

                # Detailed logging (first occurrence only)
                if sanitized_count == 1:
                    logger.debug(
                        f"[GRAD_SANITIZE] First occurrence at batch {batch_idx}: "
                        f"param_shape={param.shape}, "
                        f"non_finite_count={n_nonfinite}"
                    )

    return sanitized_count
```

### Implementation: Improved Gradient Logging

**Problem**: Current logs show `Mean=inf`, which is scary but harmless (pre-clip value).

**Solution**: Log both pre-clip and post-clip norms.

**Code** (add to logging section):
```python
# Every LOG_EVERY_N_STEPS batches, log gradient stats
if batch_idx > 0 and batch_idx % LOG_EVERY_N_STEPS == 0:
    if len(gradient_norms) > 0:
        recent_norms = gradient_norms[-LOG_EVERY_N_STEPS:]

        # Filter out inf values for statistics
        finite_norms = [n for n in recent_norms if np.isfinite(n)]

        if len(finite_norms) > 0:
            logger.info(
                f"[GRADIENTS] Last {len(recent_norms)} batches: "
                f"Mean={np.mean(finite_norms):.2f} | "
                f"P50={np.median(finite_norms):.2f} | "
                f"P95={np.percentile(finite_norms, 95):.2f} | "
                f"Max={np.max(finite_norms):.2f}"
            )

            # Warn about inf gradients
            n_inf = len(recent_norms) - len(finite_norms)
            if n_inf > 0:
                logger.warning(
                    f"[GRADIENTS] {n_inf}/{len(recent_norms)} batches had inf pre-clip norm "
                    f"(normal with FP16, clipping handles it)"
                )
        else:
            logger.warning(
                f"[GRADIENTS] All {len(recent_norms)} batches had inf pre-clip norm "
                f"(verify gradient clipping is working)"
            )
```

---

## Part 6: Testing Strategy

### Unit Tests

**File**: `tests/unit/train/test_gradient_sanitization.py` (new)

```python
def test_sanitize_gradients_replaces_nan():
    """Verify NaN gradients are replaced with zeros."""
    model = SimpleModel()

    # Inject NaN gradient
    for param in model.parameters():
        param.grad = torch.tensor([1.0, float('nan'), 3.0])
        break

    count = _sanitize_gradients(model, logger, batch_idx=0)

    assert count == 1
    assert torch.isfinite(param.grad).all()
    assert param.grad[1] == 0.0  # NaN replaced


def test_sanitize_gradients_replaces_inf():
    """Verify inf gradients are replaced with zeros."""
    model = SimpleModel()

    # Inject inf gradient
    for param in model.parameters():
        param.grad = torch.tensor([1.0, float('inf'), -float('inf')])
        break

    count = _sanitize_gradients(model, logger, batch_idx=0)

    assert count == 1
    assert param.grad[1] == 0.0
    assert param.grad[2] == 0.0


def test_sanitize_gradients_noop_when_finite():
    """Verify no changes when all gradients are finite."""
    model = SimpleModel()

    for param in model.parameters():
        param.grad = torch.randn_like(param)

    original_grads = [p.grad.clone() for p in model.parameters()]
    count = _sanitize_gradients(model, logger, batch_idx=0)

    assert count == 0
    for orig, param in zip(original_grads, model.parameters()):
        assert torch.equal(orig, param.grad)
```

### Integration Tests

**File**: `tests/integration/test_training_edge_cases.py` (add)

```python
@pytest.mark.gpu
def test_training_with_injected_nan_gradients():
    """Verify training continues with NaN gradients when sanitization enabled."""
    # Enable sanitization
    os.environ["BGB_SANITIZE_GRADS"] = "1"

    # Train with injected NaN
    model = create_test_model()

    # Hook to inject NaN at batch 5
    def inject_nan_hook(module, grad_input, grad_output):
        if trainer.current_batch == 5:
            return (grad_output[0] * float('nan'),)

    model.register_full_backward_hook(inject_nan_hook)

    # Should complete without crashing
    metrics = train_one_epoch(model, dataloader, config)

    # Verify training continued
    assert metrics["batches_completed"] > 5
    assert "nan" not in str(metrics["final_loss"]).lower()
```

### Performance Benchmarks

**File**: `tests/performance/test_gradient_overhead.py` (new)

```python
@pytest.mark.gpu
def test_sanitization_overhead():
    """Measure performance impact of gradient sanitization."""
    model = create_large_model()  # 31M parameters

    times_without = []
    times_with = []

    # Benchmark without sanitization
    os.environ["BGB_SANITIZE_GRADS"] = "0"
    for _ in range(100):
        t0 = time.time()
        train_one_batch(model, batch)
        times_without.append(time.time() - t0)

    # Benchmark with sanitization
    os.environ["BGB_SANITIZE_GRADS"] = "1"
    for _ in range(100):
        t0 = time.time()
        train_one_batch(model, batch)
        times_with.append(time.time() - t0)

    overhead = np.mean(times_with) - np.mean(times_without)
    overhead_pct = (overhead / np.mean(times_without)) * 100

    # Overhead should be < 5%
    assert overhead_pct < 5.0, f"Sanitization overhead too high: {overhead_pct:.1f}%"
```

---

## Part 7: Documentation Updates

### Files to Update (CRITICAL)

**ALL of these need fixing**:

1. `docs/08-operations/nan-prevention-complete.md` - Remove "REQUIRED" claims
2. `docs/03-configuration/env-vars.md` - Update flag descriptions
3. `docs/getting-started/quickstart.md` - Remove mandatory export
4. `docs/getting-started/first-run.md` - Update troubleshooting
5. `CLAUDE.md` - Update environment variables section
6. `README.md` - Fix quickstart commands
7. `deploy/modal/app.py` - Update environment setup comments

### New Documentation

**File**: `docs/08-operations/gradient-protection-guide.md` (new)

```markdown
# Gradient Protection Guide

## What Actually Protects Your Gradients

### Primary Protection: Gradient Clipping (ALWAYS ON)

```yaml
training:
  gradient_clip: 0.5  # Scales gradients with norm > 0.5
```

This is PyTorch standard practice and is **always applied**, regardless of environment variables.

### Optional: Gradient Sanitization (DEBUGGING ONLY)

```bash
export BGB_SANITIZE_GRADS=1  # Replace NaN/Inf with zeros
```

**When to use**:
- Debugging gradient explosions
- Research with experimental architectures
- Investigating FP16 overflow with mixed precision

**When NOT to use**:
- Production training (adds overhead for minimal benefit)
- If gradient clipping is working (it is)
- If you just want stability (clipping provides it)

**Default**: `0` (disabled)

## Why You Don't Need Sanitization

Gradient clipping already handles inf/nan gradients:
1. Calculates total gradient norm (can be inf)
2. Scales ALL gradients proportionally
3. Result: finite gradients even if some were inf
4. PyTorch handles this automatically

## What To Do If You See "Mean=inf"

**This is normal with FP16 mixed precision!**

- The logged value is **pre-clip** gradient norm
- Actual parameter updates use **post-clip** (finite) gradients
- If loss is decreasing, training is working correctly
- No action needed

## Related Documentation

- Industry best practices: [PyTorch Training Loop](https://pytorch.org/tutorials)
- Mixed precision: [PyTorch AMP](https://pytorch.org/docs/stable/amp.html)
```

---

## Part 8: Implementation Checklist

### Phase 1: Code Changes (1-2 hours)

- [ ] **Add `_sanitize_gradients()` helper** to `train_step.py`
- [ ] **Integrate sanitization** (after backward, before clip)
- [ ] **Improve gradient logging** (pre-clip vs post-clip)
- [ ] **Remove unused flags** from `env.py`:
  - [ ] Remove `BGB_SANITIZE_INPUTS`
  - [ ] Remove `BGB_SKIP_OPT_STEP_ON_NAN`
  - [ ] Remove `BGB_SAFE_CLAMP`
  - [ ] Keep `BGB_SANITIZE_GRADS` (change default to `0`)

### Phase 2: Testing (2-3 hours)

- [ ] **Write unit tests** for sanitization logic
- [ ] **Write integration tests** with injected NaN
- [ ] **Write performance benchmarks** for overhead
- [ ] **Run full test suite** (ensure no regressions)

### Phase 3: Documentation (3-4 hours)

- [ ] **Update `env-vars.md`** - fix all 4 flags
- [ ] **Update `nan-prevention-complete.md`** - remove "REQUIRED" claims
- [ ] **Update `quickstart.md`** - remove mandatory exports
- [ ] **Update `first-run.md`** - fix troubleshooting
- [ ] **Update `CLAUDE.md`** - correct environment variables
- [ ] **Update `README.md`** - fix quickstart
- [ ] **Create `gradient-protection-guide.md`** - new authoritative doc
- [ ] **Update Modal app.py** - fix comments about protection

### Phase 4: Verification (1 hour)

- [ ] **Search all docs** for remaining false claims
  ```bash
  rg "SANITIZE_GRADS.*REQUIRED|SANITIZE_GRADS.*CRITICAL" docs/ archived_docs/
  ```
- [ ] **Verify Modal setup** is updated
- [ ] **Run smoke test** with sanitization enabled
- [ ] **Run smoke test** with sanitization disabled (default)

### Phase 5: Cleanup (1 hour)

- [ ] **Archive old docs** that reference removed flags
- [ ] **Update TODO.md** with this work
- [ ] **Git commit** with detailed message
- [ ] **Create PR** for review

---

## Part 9: Rollout Plan

### Step 1: Implement and Test Locally

1. Make code changes
2. Run full test suite
3. Run local smoke test (both modes)
4. Verify no regressions

### Step 2: Update Documentation

1. Fix all docs (see checklist)
2. Create new protection guide
3. Archive outdated docs
4. Update CLAUDE.md

### Step 3: Test on Modal

1. Deploy updated code
2. Run smoke test (sanitization disabled)
3. Run smoke test (sanitization enabled)
4. Verify logs are clear

### Step 4: Full Training Validation

1. Start local training (sanitization disabled)
2. Start Modal training (sanitization disabled)
3. Monitor for 100 batches each
4. Verify:
   - Loss decreasing
   - No NaN warnings
   - Gradient norms finite
   - No "Sanitized NaN gradients" messages

### Step 5: AI Consensus Review

1. Present this document to another AI agent
2. Get consensus on approach
3. Address any concerns
4. Finalize implementation plan

---

## Part 10: Success Criteria

Training is considered **correctly protected** when:

1. ✅ **Code matches documentation** (no false claims)
2. ✅ **Gradient clipping always applied** (primary protection)
3. ✅ **Gradient sanitization works** (when enabled)
4. ✅ **Zero performance regression** (sanitization disabled by default)
5. ✅ **Clear logging** (pre-clip vs post-clip norms)
6. ✅ **All tests passing** (unit + integration + performance)
7. ✅ **Documentation accurate** (no "REQUIRED" for optional features)
8. ✅ **Modal setup correct** (no false security claims)

---

## Appendix A: Why This Happened

**Root Cause Analysis**:

1. **Documentation-Driven Development**: Docs written before code
2. **Copy-Paste Documentation**: Same text across v1/v2/v3 archives
3. **No Integration Tests**: Flags never tested end-to-end
4. **False Correlation**: Training worked, so flags "must be working"
5. **Complexity Hiding Bugs**: Large codebase, hard to trace execution

**How to Prevent**:

1. **Test-Driven Development**: Write tests FIRST, then code
2. **Documentation Reviews**: Verify code matches docs
3. **Integration Tests**: Test full training pipeline
4. **Regular Audits**: Check environment variable usage
5. **Simplicity**: Fewer flags = less to go wrong

---

## Appendix B: Comparison with Current State

| Aspect | Current (Broken) | After Implementation |
|--------|------------------|---------------------|
| **`BGB_SANITIZE_GRADS`** | Defined, never used | Implemented, optional (default: off) |
| **`BGB_SANITIZE_INPUTS`** | Defined, never used | **REMOVED** |
| **`BGB_SKIP_OPT_STEP_ON_NAN`** | Defined, never used | **REMOVED** |
| **`BGB_SAFE_CLAMP`** | Defined, never used | **REMOVED** |
| **Gradient Clipping** | Works (but logging confusing) | Works (with clear logging) |
| **Documentation** | False claims ("REQUIRED") | Accurate (optional for debugging) |
| **Modal Setup** | Claims "NaN protection enabled" | Clarifies what actually protects |
| **Test Coverage** | No tests for sanitization | 100% coverage with unit+integration |

---

**FINAL RECOMMENDATION**:

✅ **IMPLEMENT THIS PLAN**

This gives us:
1. Industrial-strength protection (DeepMind-grade)
2. No false advertising (docs match code)
3. Debugging tools when needed (sanitization available)
4. Zero regression (sanitization off by default)
5. Clear path forward (detailed checklist)

**Estimated Total Time**: 7-10 hours of focused work

**Risk**: LOW (all changes are additive or cleanup)

**Benefit**: HIGH (eliminate false security, enable real debugging)

---

**Next Step**: Get AI consensus on this plan, then proceed with implementation.
