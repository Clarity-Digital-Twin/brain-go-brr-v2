# Known Issues (P0 Focus)

This file tracks critical (P0) issues discovered through deep audit. Do not fix here to avoid collisions with parallel workstreams; use this for triage and coordination.

## Summary Statistics
- Total P0 Issues: 11 (NEW: 7 discovered in audit)
- Total P1 Issues: 4 (NEW: 1 discovered in audit)
- Total P2 Issues: 5 (NEW: 2 discovered in audit)
- Critical Blockers: 6 (training/evaluation completely broken)

## P0 ISSUES - CRITICAL RUNTIME FAILURES

### EXISTING P0 ISSUES (Already Documented)

- Issue: Pipeline CLI requires labels but datasets created without labels
  - Severity: P0 (crash at runtime)
  - Location: src/experiment/pipeline.py: train() and train_epoch(); src/experiment/data.py: EEGWindowDataset
  - Symptoms: `train_epoch` assumes `(windows, labels)` but `EEGWindowDataset` returns only `windows` when `label_files=None` (as used in pipeline.main). Also, `create_balanced_sampler` builds `all_labels` via `train_dataset[i][1]`, which fails when labels are absent.
  - Repro:
    - `python -m src.experiment.pipeline --config configs/production.yaml` (with no label files provisioned)
  - Impact: Training CLI unusable on unlabeled data; sampler creation crashes; first-batch label extraction crashes.
  - Workaround: Provide label files and adapt dataset instantiation; or disable balanced sampling and avoid sampler path (not recommended for production).
  - Owner: Pipeline/Data
  - Notes: Decide contract for `EEGWindowDataset`: always return `(window, label)` and synthesize zeros when labels absent, or adjust pipeline to support unlabeled runs.

- Issue: Evaluation entry point in pyproject points to non-existent function
  - Severity: P0 (packaging/CLI break)
  - Location: pyproject.toml: [project.scripts] `evaluate = "src.experiment.evaluate:main"`; but `src/experiment/evaluate.py` has no `main`.
  - Symptoms: Installed script `evaluate` fails to run/import.
  - Repro: `uv run evaluate` (after install)
  - Impact: Broken CLI for evaluation when installed as a package.
  - Workaround: Use `python -m src.cli evaluate ...` instead.
  - Owner: Packaging/CLI
  - Notes: Either add `main()` in evaluate.py or point script to `src.cli:main` with a subcommand.

### NEW P0 ISSUES (Discovered in Deep Audit)

- Issue: Dataset returns inconsistent types causing train_epoch to crash
  - Severity: P0 (type mismatch crash)
  - Location: src/experiment/data.py:349-357, pipeline.py:184
  - Symptoms: EEGWindowDataset.__getitem__ returns either `window` alone OR `(window, label)` tuple depending on self._labels state. train_epoch expects ALWAYS `(windows, labels)` tuple.
  - Repro: Run training with dataset that has no labels
  - Impact: TypeError crash: "cannot unpack non-sequence" in train_epoch line 184
  - Fix Plan: ALWAYS return tuple `(window, label)`. When no labels, return zero tensor of correct shape.
  - Code Fix:
    ```python
    # In EEGWindowDataset.__getitem__
    if self._labels:
        label = torch.from_numpy(self._labels[idx])
    else:
        # Return zeros of correct shape when no labels
        label = torch.zeros(self._windows[idx].shape[-1], dtype=torch.float32)
    return window, label
    ```

- Issue: Config schema mismatch - production.yaml incompatible with schemas
  - Severity: P0 (config validation failure)
  - Location: configs/production.yaml vs src/experiment/schemas.py
  - Symptoms: Multiple field mismatches:
    1. `batch_size` in yaml under `data:` but not in DataConfig schema
    2. `min_duration` in yaml but schema expects `duration.min_duration_s`
    3. Missing required nested configs (hysteresis, morphology, duration, events)
  - Repro: `Config.from_yaml(Path("configs/production.yaml"))`
  - Impact: Pydantic validation error on config load
  - Fix Plan: Either update yaml to match schema OR update schema to be backward compatible
  - Code Fix: Add missing fields to schemas and update yaml structure

- Issue: Train function missing output_dir in config
  - Severity: P0 (AttributeError crash)
  - Location: src/experiment/pipeline.py:476
  - Symptoms: `output_dir = Path(config.experiment.output_dir)` but ExperimentConfig has no `output_dir` field
  - Repro: Run train() with any config
  - Impact: AttributeError: 'ExperimentConfig' object has no attribute 'output_dir'
  - Fix Plan: Add output_dir to ExperimentConfig schema
  - Code Fix:
    ```python
    class ExperimentConfig(BaseModel):
        output_dir: Path = Field(default=Path("outputs"), description="Output directory")
    ```

- Issue: Missing PostprocessingConfig fields cause AttributeError
  - Severity: P0 (runtime crash)
  - Location: src/experiment/evaluate.py:170-174
  - Symptoms: Code tries to access `post_cfg.events.tau_merge` and `post_cfg.events.confidence_method` but PostprocessingConfig may not have `events` attribute
  - Repro: Call batch_probs_to_events with minimal PostprocessingConfig
  - Impact: AttributeError when accessing nested config fields
  - Fix Plan: Add proper hasattr checks OR ensure PostprocessingConfig always has events field
  - Code Fix: Already partially fixed with hasattr checks, but schema should enforce structure

- Issue: Window stitching not implemented but required for correct FA/24h
  - Severity: P0 (metric correctness)
  - Location: src/experiment/evaluate.py:253, postprocess.py (missing stitch_windows)
  - Symptoms: `total_hours = labels.numel() / (fs * 3600)` counts overlapped time multiple times
  - Impact: FA/24h rates are 6x inflated (60s windows with 10s stride = 6x overlap)
  - Fix Plan: Implement stitch_windows in postprocess.py and use actual recording duration
  - Code Fix:
    ```python
    # Need to track actual recording duration, not sample count
    # Implement window stitching before FA calculation
    ```

- Issue: Missing TrainingConfig.batch_size field
  - Severity: P0 (AttributeError)
  - Location: src/experiment/pipeline.py:639, schemas.py TrainingConfig
  - Symptoms: main() tries to use `config.training.batch_size` but TrainingConfig has no such field
  - Repro: Run pipeline.main()
  - Impact: AttributeError: 'TrainingConfig' object has no attribute 'batch_size'
  - Fix Plan: Add batch_size to TrainingConfig OR use config.data.batch_size
  - Code Fix: Change to use `config.data.batch_size` since DataConfig should own this

- Issue: FA/24h computed on overlapped windows without stitching
  - Severity: P0 (metric correctness)
  - Location: src/experiment/evaluate.py: sensitivity_at_fa_rates(), find_threshold_for_fa_eventized(); uses `labels.numel()` for time and per-window events without stitching.
  - Symptoms: `total_hours = labels.numel() / (fs * 3600)` double-counts time when windows overlap; predicted events are computed per-window and not stitched across boundaries.
  - Impact: FA/24h and threshold search are inaccurate; downstream sensitivity@FA and TAES become unreliable.
  - Workaround: For now, evaluate on non-overlapping segments only; or pre-stitch probabilities to full timelines before calling evaluation.
  - Owner: Evaluation/Post-processing
  - Notes: Blocked on Phase 4 stitching APIs. Replace with `stitch_windows()` and compute actual per-record duration.

## P1 ISSUES - LOGIC ERRORS AND AMBIGUITIES

### EXISTING P1 ISSUES

- Issue: Eventization semantics unclear; unused variable suggests mismatch
  - Severity: P1 (logic ambiguity, not a crash)
  - Location: src/experiment/evaluate.py: batch_probs_to_events()
  - Symptoms: Variable `binary = (prob_np > threshold)` computed but unused; hysteresis gates directly on raw probs (τ_on/τ_off). It mixes a global threshold and hysteresis, but only hysteresis affects the final mask.
  - Impact: Threshold parameter may not influence outputs as expected; harder to reason about FA/threshold search.
  - Workaround: Treat `threshold` as unused and rely on hysteresis only; or remove hysteresis temporarily.
  - Owner: Evaluation/Post-processing
  - Notes: To be resolved when Phase 4 APIs land (apply_hysteresis vs global thresholding).

### NEW P1 ISSUE (Discovered in Audit)

- Issue: Threshold parameter completely ignored in batch_probs_to_events
  - Severity: P1 (silent logic error)
  - Location: src/experiment/evaluate.py:150-182
  - Symptoms: threshold parameter passed but never used; postprocess_predictions doesn't accept threshold
  - Impact: FA threshold search may not work as intended; all thresholds produce same results
  - Fix Plan: Either remove threshold param OR implement proper global thresholding before hysteresis
  - Code Fix: Pass threshold to postprocess_predictions or apply before calling it

- Issue: Memory-heavy window materialization in dataset
  - Severity: P1 (scaling blocker)
  - Location: src/experiment/data.py: EEGWindowDataset
  - Symptoms: All windows are precomputed and kept in memory.
  - Impact: OOM for large corpora; not viable for full TUH.
  - Workaround: Keep for unit tests and small runs; switch to streaming/dynamic windowing for large runs.
  - Owner: Data/Infra
  - Notes: Planned improvement; not urgent for unit/integration tests.

## P2 ISSUES - PERFORMANCE AND MINOR PROBLEMS

### EXISTING P2 ISSUES

- Issue: BCE with logits reconstructed from probabilities
  - Severity: P2 (numerical suboptimality)
  - Location: src/experiment/pipeline.py: train_epoch()
  - Symptoms: Model outputs Sigmoid probabilities; training reconstructs logits via `torch.logit(probs.clamp(...))` to use `BCEWithLogitsLoss`.
  - Impact: Extra non-linear step; may reduce numerical headroom at extremes.
  - Workaround: Keep as-is; stable with clamp.
  - Owner: Training
  - Notes: Consider changing detection head to output raw logits and apply Sigmoid only at inference.

- Issue: Balanced sampler label aggregation cost
  - Severity: P2 (performance)
  - Location: src/experiment/pipeline.py: main(), create_balanced_sampler()
  - Symptoms: Builds `all_labels` by iterating full dataset, materializing labels for sampler.
  - Impact: Slow startup and high memory on large sets.
  - Workaround: Disable balanced sampling or compute window-level label presence offline.
  - Owner: Pipeline

### NEW P2 ISSUES (Discovered in Audit)

- Issue: Config.from_yaml not implemented but used in main()
  - Severity: P2 (missing method)
  - Location: src/experiment/pipeline.py:607
  - Symptoms: `Config.from_yaml(Path(args.config))` but Config class has no from_yaml method
  - Impact: AttributeError when running CLI
  - Fix Plan: Implement from_yaml classmethod in Config
  - Code Fix:
    ```python
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    ```

- Issue: Missing Config root class that combines all sub-configs
  - Severity: P2 (missing schema)
  - Location: src/experiment/schemas.py
  - Symptoms: No Config class that combines DataConfig, ModelConfig, etc.
  - Impact: Can't validate full config structure
  - Fix Plan: Add Config class that composes all sub-configs
  - Code Fix:
    ```python
    class Config(BaseModel):
        data: DataConfig
        preprocessing: PreprocessingConfig
        model: ModelConfig
        postprocessing: PostprocessingConfig
        training: TrainingConfig
        evaluation: EvaluationConfig
        experiment: ExperimentConfig
    ```

- Issue: Unused global threshold vs hysteresis definitions in docs/code
  - Severity: P2 (documentation drift)
  - Location: README.md, PHASE4_POSTPROCESSING.md vs evaluate.py
  - Impact: Confusion over intended thresholding pipeline.
  - Owner: Docs/Eval

# CRITICAL FIX PRIORITIES

## Must Fix Before ANY Training/Evaluation (P0 Blockers):
1. EEGWindowDataset type inconsistency - blocks all training
2. Config schema mismatches - blocks config loading
3. Missing Config.from_yaml - blocks CLI usage
4. Missing TrainingConfig.batch_size - blocks training
5. Missing output_dir in ExperimentConfig - blocks checkpoint saving
6. Window stitching for correct FA/24h - blocks accurate evaluation

## Fix Plan Execution Order:
1. **Phase 1: Type Safety** (Fix dataset return types, ensure consistent tuple returns)
2. **Phase 2: Schema Alignment** (Add missing Config class, align yaml with schemas)
3. **Phase 3: Window Stitching** (Implement proper stitching for overlapped windows)
4. **Phase 4: Threshold Logic** (Clarify threshold vs hysteresis semantics)

# Coordination Notes
- Phase 4 implementation (postprocess/events/export) will address stitching, morphology, and eventization; evaluation should be refactored to consume those APIs.
- Mamba CUDA/CPU dispatch appears robust; no P0 found after gating and kernel-width coercion.

# Code Snippets for Critical Fixes

## Fix 1: EEGWindowDataset Type Consistency
```python
# In src/experiment/data.py:349-357
def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.from_numpy(self._windows[idx])
    if self.transform is not None:
        window = self.transform(window)

    if self._labels:
        label = torch.from_numpy(self._labels[idx])
    else:
        # ALWAYS return tuple with zero labels when none exist
        label = torch.zeros(window.shape[-1], dtype=torch.float32)

    return window, label  # ALWAYS return tuple
```

## Fix 2: Add Missing Config Class
```python
# In src/experiment/schemas.py
class Config(BaseModel):
    """Root configuration combining all sub-configs."""
    data: DataConfig = Field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    postprocessing: PostprocessingConfig = Field(default_factory=PostprocessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

## Fix 3: Add batch_size to Proper Location
```python
# Move batch_size from yaml data: section to training: section
# OR add to DataConfig schema as it relates to data loading
class DataConfig(BaseModel):
    batch_size: int = Field(default=32, ge=1, le=512, description="Batch size for training")
```

# Triage Owners
- Pipeline/Data: unlabeled dataset handling; sampler assumptions
- Packaging/CLI: evaluate entrypoint mismatch
- Evaluation/Post-processing: FA/24h time accounting; stitching; eventization semantics
- Training: logits vs probabilities

