# Technical Debt: train/loop.py

**File**: `src/brain_brr/train/loop.py`
**Lines**: 1694
**Created**: 2025-01-28
**Priority**: MEDIUM - Refactor after logging migration complete

## Overview

The training loop orchestrator is functional but has accumulated procedural complexity. While it follows SOLID principles at a high level, several functions exceed recommended size limits and mix concerns.

## Debt Items

### 1. Oversized Functions

#### `train_epoch()` (~400 lines, 752-1152)
**Issue**: Monolithic training loop with mixed concerns
**Impact**: Hard to test, debug, and extend
**Refactor targets**:
```python
# Extract these concerns into separate functions:
- _process_batch(model, windows, labels, loss_fn, device) -> tuple[loss, logits]
- _update_gradients(loss, optimizer, scaler, gradient_clip, sanitize)
- _log_batch_progress(batch_idx, loss, metrics, progress_bar)
- _save_mid_epoch_checkpoint(model, optimizer, epoch, batch_idx, checkpoint_dir)
- _check_nan_convergence(consecutive_nans, max_consecutive) -> bool
```

#### `main()` (~400 lines, 1412-1694)
**Issue**: Dataset creation logic mixed with training orchestration
**Impact**: Violates Single Responsibility Principle
**Refactor targets**:
```python
# Move to data module:
- create_train_datasets(config) -> tuple[Dataset, Dataset]
- setup_data_loaders(train_dataset, val_dataset, config) -> tuple[DataLoader, DataLoader]
- validate_split_policy(config, data_root) -> tuple[train_files, val_files]
- build_cache_manifest(cache_dir, dataset) -> Path
```

### 2. Duplicated Patterns

#### tqdm Setup (3x duplication)
**Lines**: 816-835, 1012-1031, 1236-1255
**Issue**: Identical tqdm initialization logic repeated
```python
# Extract to utility:
def create_progress_bar(iterator, desc: str, use_tqdm: bool = True):
    """Create safe tqdm progress bar for Modal compatibility."""
    if not use_tqdm or env.disable_tqdm():
        return iterator
    try:
        return tqdm(iterator, desc=desc, leave=False,
                   file=sys.stderr, ascii=True, ncols=80)
    except Exception:
        return iterator
```

#### Heartbeat Logging (2x duplication)
**Lines**: 911-919, 1281-1289
**Issue**: Identical heartbeat pattern in train and validate
```python
# Extract to utility:
def log_heartbeat_if_due(last_time: float, interval: float,
                         batch_idx: int, total: int, **metrics):
    """Log heartbeat for long-running operations."""
    if time.time() - last_time > interval:
        logger.info(f"[HEARTBEAT] Batch {batch_idx}/{total} | " +
                   " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        return time.time()
    return last_time
```

#### Dataset Sampling Logic (2x duplication)
**Lines**: 607-677 (train_epoch), 1581-1626 (main)
**Issue**: Similar dataset statistics sampling

### 3. Complex Conditional Logic

#### Loss Function Setup
**Lines**: 690-719
**Issue**: Nested conditionals for focal/BCE selection
```python
# Simplify with factory:
def create_loss_function(config: TrainingConfig, pos_weight: float):
    """Factory for loss functions."""
    if config.loss == "focal":
        return FocalLoss(alpha=config.focal_alpha,
                        gamma=config.focal_gamma,
                        pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

#### Gradient Sanitization
**Lines**: 861-893, 895-927
**Issue**: Repeated NaN handling logic with/without AMP

### 4. Magic Numbers and Constants

```python
# Should be configuration/constants:
- heartbeat_interval = 300  # Line 605
- sample_size = 500  # Line 144
- max_consecutive_nans = 50  # Line 793
- warmup_steps calculation  # Line 344
- checkpoint file patterns  # Lines 972, 1349
```

### 5. Error Handling Inconsistencies

- Some exceptions caught and logged, others re-raised
- Inconsistent cleanup patterns (tqdm, file handles)
- Mix of `logger.error()`, `logger.critical()`, and `raise`

## Proposed Refactoring Plan

### Phase 1: Extract Utilities (1 day)
1. Create `src/brain_brr/train/utils.py`:
   - Progress bar creation
   - Heartbeat logging
   - Checkpoint management
   - Dataset statistics

### Phase 2: Split Dataset Logic (1 day)
1. Move to `src/brain_brr/data/loaders.py`:
   - Dataset creation from config
   - Split policy handling
   - Manifest building
   - Sampler creation

### Phase 3: Decompose train_epoch (2 days)
1. Create `src/brain_brr/train/batch_processor.py`:
   - Batch processing
   - Gradient updates
   - NaN handling
   - Loss computation

### Phase 4: Simplify main() (1 day)
1. Reduce to orchestration only:
   - Load config
   - Create datasets (via factory)
   - Create model
   - Call train()
   - Save results

## Benefits of Refactoring

1. **Testability**: Can unit test individual components
2. **Maintainability**: Easier to modify specific behaviors
3. **Reusability**: Utilities available for evaluation/inference
4. **Clarity**: Clear separation of concerns
5. **Performance**: Easier to optimize isolated functions

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking training | HIGH | Extensive testing, gradual refactor |
| Performance regression | MEDIUM | Benchmark before/after |
| Lost functionality | LOW | Keep old code as reference |

## Recommended Timeline

**When**: After Phase 3 of logging migration
**Duration**: 5 days
**Priority**: MEDIUM - Current code works, but limits velocity

## Notes

- The file is NOT a disaster - it's typical ML training code
- SOLID principles are generally followed at module level
- Main issue is function size and concern mixing
- Refactoring should preserve all current functionality
- Consider keeping verbose logging during refactor for debugging

## Decision

**Recommendation**: Defer refactoring until after logging migration and edge similarity fixes are stable. The code works and has good error handling. Refactoring now would distract from critical production issues.

**Alternative**: Leave as-is if team velocity is acceptable. Many successful ML projects have similar training loops.