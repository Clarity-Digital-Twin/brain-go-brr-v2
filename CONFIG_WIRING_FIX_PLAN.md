# Config Wiring Fix Plan - FINAL CORRECTED VERSION

**Date**: 2025-10-04
**Status**: 🔴 ALL TRAINING STOPPED - Awaiting approval
**Verification**: Full codebase review + external validation

---

## What's ACTUALLY Broken (Triple-Verified)

### 1. Focal Gamma Warmup ❌

**Function exists but NEVER called**:
- Definition: `src/brain_brr/train/warmup.py:11-43`
- Usage: `grep -rn "get_focal_gamma" src/` shows ONLY definition, zero calls
- Impact: Focal gamma fixed at 2.0 from batch 0 → full 4x gradient amplification immediately

**Adjacency warmup WORKS** ✅:
- `src/brain_brr/models/adjacency.py:21-55` implements `get_adj_temperature()`
- Called at `adjacency.py:102` inside `condition_adjacency()`
- Verified working: tau ramps 2.0→1.0 over warmup

**Fix**: Import and call `get_focal_gamma()` in train_step.py:224

---

### 2. Mid-Epoch Checkpoints ❌ (CORRECTED - I WAS WRONG)

**Code uses WRONG field + env vars**:

`src/brain_brr/train/loop.py:177-186`:
```python
mid_epoch_minutes=(
    float(env.mid_epoch_minutes() or 0)  # ← ENV VAR
    if config.training.resume and env.mid_epoch_minutes() is not None
    else getattr(
        config.experiment,
        "mid_epoch_checkpoint_minutes",  # ← Field doesn't exist!
        10.0 if config.training.resume else None,
    )
),
mid_epoch_keep=int(env.mid_epoch_keep()),  # ← ENV VAR
```

**Schema defines**: `training.mid_checkpoint_interval_s` (schemas.py:508)
**Code reads**: `experiment.mid_epoch_checkpoint_minutes` (doesn't exist)
**Fallback**: Env vars `BGB_MID_EPOCH_MINUTES` and `BGB_MID_EPOCH_KEEP`

**Result**: Config fields `mid_checkpoint_interval_s` and `mid_epoch_keep` are NEVER used

**Fix**: Use correct config fields, remove env var dependency

---

### 3. Gradient Accumulation ❌

**Config field defined**: `training.gradient_accumulation_steps` (schemas.py:516-518)
**Code usage**: `grep -rn "gradient_accumulation_steps" src/brain_brr/train/` → **NO RESULTS**

**train_step.py:215-260**:
```python
optimizer.zero_grad(set_to_none=True)  # EVERY batch
loss.backward()  # EVERY batch
optimizer.step()  # EVERY batch
```

**No accumulation logic exists**

---

### 4. Optimizer/Scheduler Enums ❌

**Schema promises**:
```python
optimizer: Literal["adamw", "adam", "sgd"]  # schemas.py:495
scheduler.type: Literal["cosine", "linear", "constant"]  # schemas.py:402
```

**Runtime crashes**:
```python
# optimizer_factory.py:28-60
if config.optimizer == "adamw":
    return AdamW(...)
else:
    raise ValueError(f"Unknown optimizer: {config.optimizer}")  # adam/sgd crash

# optimizer_factory.py:75-99
if config.type == "cosine":
    return LambdaLR(...)
else:
    raise ValueError(f"Unknown scheduler: {config.type}")  # linear/constant crash
```

---

### 5. Preprocessing Configs ❌

**Config fields defined**: `preprocessing.bandpass`, `preprocessing.notch_freq`
**Code usage**: `grep -rn "preprocessing.bandpass" src/` → **NO RESULTS**

**preprocess.py:10-52** uses hardcoded defaults:
```python
def preprocess_eeg(
    bandpass: tuple[float, float] = (0.5, 120.0),  # Hardcoded
    notch_freq: int = 60,  # Hardcoded
):
```

**Dataset hardcoded**: `loop.py:382` always loads TUSZ, never checks `config.data.dataset`

---

### 6. Logging/Eval Toggles ❌

**Config fields defined**: `save_model`, `save_best_only`, `save_predictions`, `save_plots`
**Code usage**: `grep -rn "save_model\|save_predictions" src/brain_brr/train/` → **NO RESULTS**

**loop.py:272-323** always saves checkpoints (no config checks)

---

## Implementation Plan

### Phase 1: Focal Gamma Warmup (2 hours)

**File**: `src/brain_brr/train/train_step.py`

**Add import**:
```python
from src.brain_brr.train.warmup import get_focal_gamma
```

**Replace line 224**:
```python
# BEFORE:
focal_weight = at * ((1 - pt) ** focal_gamma)

# AFTER:
current_gamma = get_focal_gamma(global_step, warmup_schedule, target_gamma=focal_gamma)
focal_weight = at * ((1 - pt) ** current_gamma)
```

**Add logging**:
```python
if warmup_schedule and warmup_schedule.enabled and warmup_schedule.focal_gamma_enabled:
    if batch_idx % 100 == 0:
        logger.info(f"[WARMUP] Batch {batch_idx} focal_gamma={current_gamma:.3f}")
```

---

### Phase 2: Mid-Epoch Checkpoints (2 hours)

**File**: `src/brain_brr/train/loop.py:177-186`

**BEFORE (broken)**:
```python
mid_epoch_minutes=(
    float(env.mid_epoch_minutes() or 0)
    if config.training.resume and env.mid_epoch_minutes() is not None
    else getattr(config.experiment, "mid_epoch_checkpoint_minutes", ...)
),
mid_epoch_keep=int(env.mid_epoch_keep()),
```

**AFTER (fixed)**:
```python
mid_epoch_minutes=(
    config.training.mid_checkpoint_interval_s / 60.0
    if config.training.mid_checkpoint_interval_s
    else None
),
mid_epoch_keep=config.training.mid_epoch_keep or 3,
```

**Deprecate env vars** (`src/brain_brr/utils/env.py`):
```python
def mid_epoch_minutes() -> float | None:
    """DEPRECATED: Use config.training.mid_checkpoint_interval_s instead."""
    val = os.getenv("BGB_MID_EPOCH_MINUTES")
    if val:
        logger.warning(
            "BGB_MID_EPOCH_MINUTES is DEPRECATED. "
            "Use 'training.mid_checkpoint_interval_s' in config."
        )
    return float(val) if val else None
```

---

### Phase 3: Gradient Accumulation (4 hours)

**File**: `src/brain_brr/train/train_step.py`

**Add parameter**:
```python
def train_epoch(
    # ... existing ...
    gradient_accumulation_steps: int = 1,
) -> float | tuple[float, int]:
```

**Add accumulation logic**:
```python
accumulation_counter = 0

for batch_idx, batch in enumerate(dataloader):
    # ... batch prep ...

    loss = loss / gradient_accumulation_steps

    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()

    accumulation_counter += 1

    if accumulation_counter >= gradient_accumulation_steps:
        if env.sanitize_grads():
            _sanitize_gradients(model, logger, batch_idx)

        pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        accumulation_counter = 0

        if scheduler is not None:
            scheduler.step()
            global_step += 1
```

**Thread config** (`loop.py`):
```python
result = train_epoch(
    # ... existing ...
    gradient_accumulation_steps=config.training.gradient_accumulation_steps,
)
```

---

### Phase 4: Schema Cleanup (1 hour)

**File**: `src/brain_brr/config/schemas.py`

**Restrict enums**:
```python
# Line 495
optimizer: Literal["adamw"] = Field(default="adamw", description="Only AdamW implemented")

# Line 402
type: Literal["cosine"] = Field(default="cosine", description="Only cosine implemented")
```

**Add dataset validator**:
```python
@field_validator("dataset")
@classmethod
def check_dataset(cls, v: str) -> str:
    if v != "tuh_eeg":
        raise ValueError("Only 'tuh_eeg' implemented (CHB-MIT coming v3.7)")
    return v
```

---

### Phase 5: Preprocessing Wiring (4 hours)

**Thread config values**:

**loop.py:382**:
```python
if config.data.dataset == "tuh_eeg":
    from src.brain_brr.data.tusz_splits import load_tusz_for_training
    splits = load_tusz_for_training(data_root, use_eval=False, verbose=True)
elif config.data.dataset == "chb_mit":
    raise NotImplementedError("CHB-MIT coming v3.7")
else:
    raise ValueError(f"Unknown dataset: {config.data.dataset}")
```

**Wherever preprocess_eeg called**:
```python
preprocessed = preprocess_eeg(
    signal=raw_signal,
    fs=sampling_rate,
    bandpass=tuple(config.preprocessing.bandpass),
    notch_freq=config.preprocessing.notch_freq,
)
```

---

### Phase 6: Logging/Eval Toggles (2 hours)

**loop.py:272-323**:
```python
if config.experiment.save_model:
    if current_metric == early_stopping.best_score:
        if not config.experiment.save_best_only or is_new_best:
            save_checkpoint(...)

    if checkpoint_interval > 0:
        save_checkpoint(...)

    save_checkpoint(...)  # Always save last for resume
```

**Validation section**:
```python
if config.evaluation.save_predictions:
    np.save(output_dir / "predictions.npy", all_preds)

if config.evaluation.save_plots:
    plot_roc_curve(...)
```

---

## Summary

| Issue | Status | Fix Time | Verified By |
|-------|--------|----------|-------------|
| Focal gamma warmup | ❌ NOT implemented | 2h | grep shows no calls |
| Adjacency warmup | ✅ WORKS | 0h | Called at adjacency.py:102 |
| Mid-epoch checkpoints | ❌ Uses env vars + wrong field | 2h | loop.py:177-186 |
| Gradient accumulation | ❌ NOT implemented | 4h | No code references |
| Optimizer/scheduler enums | ❌ Schema lies | 1h | Crashes on other values |
| Preprocessing configs | ❌ Hardcoded | 4h | No config usage |
| Logging toggles | ❌ Never checked | 2h | No code references |

**Total**: 15 hours (7 issues, 1 already works)

**Warmup status**: 50% implemented (adjacency works, focal doesn't)

---

**Status**: TRIPLE-VERIFIED (code review + grep + external validation) - Ready for approval
