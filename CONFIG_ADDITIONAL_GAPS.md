# Additional Config/Env Wiring Gaps (beyond Phase 1 plan)

Verified October 4, 2025. These items are **not** covered in the current fix plan and remain unwired today.

## 1. Preprocessing knobs are cosmetic (outside bandpass/notch)
- Schema: `src/brain_brr/config/schemas.py:85-104`
  - `montage`, `normalize`, `use_mne` exposed to users.
- Code usage: `rg "config.preprocessing" src` returns nothing.
- Data loaders (`src/brain_brr/data/datasets.py:140`) call `preprocess_recording(...)` without passing config fields.
- `preprocess_recording` defaults in `src/brain_brr/data/preprocess.py:11-52` ignore montage/normalize/use_mne entirely.
- **Impact**: Users cannot disable MNE, skip normalization, or select montage despite docs promising it.

## 2. Data limits (`max_samples`, `max_hours`) are dead fields
- Schema: `src/brain_brr/config/schemas.py:70-75`.
- Search: `rg "max_samples" src` and `rg "max_hours" src` show only schema definitions.
- Datasets never receive or enforce these limits.
- **Impact**: Config suggests you can cap data for debugging; nothing happens.

## 3. Logging config section unused
- Schema: `src/brain_brr/config/schemas.py:575-584` defines `log_every_n_steps`, `log_gradients`, `log_weights`.
- Search: `rg "config.logging" src` → no references.
- Actual logging uses environment defaults (`src/brain_brr/utils/logging_config.py:15-110`).
- **Impact**: Logging section in YAML has no effect; users must rely on env vars.

## 4. Experiment log level ignored
- Schema: `src/brain_brr/config/schemas.py:561-563` adds `experiment.log_level`.
- Search: `rg "experiment.log_level" src` → no hits.
- Logging configuration reads `BGB_LOG_LEVEL` env (`utils/logging_config.py:32`), not config.

## 5. Evaluation metrics list unused
- Schema: `src/brain_brr/config/schemas.py:528-534` exposes `evaluation.metrics`.
- Search: `rg "config.evaluation.metrics" src` → no hits.
- Validation pipeline (`train/loop.py:200-248`) only uses `fa_rates` and hardcoded metrics.
- **Impact**: Cannot customize metric set despite docs implying support.

## 6. Montage flag ignored in EDF loader
- Config advertises `preprocessing.montage`/`use_mne`.
- EDF loader `load_edf_file` (`src/brain_brr/data/io.py:44-205`) has an `apply_montage` parameter but nothing threads config into it.
- **Impact**: Users can’t disable montage or switch to alternate approach when data already aligned.

## 7. Warmup residual scaling flags unused
- Schema: `src/brain_brr/config/schemas.py:462-470` (residual_scale_* fields).
- Search: `rg "residual_scale_enabled" src` → only schema.
- No code adjusts residual gain during warmup.

These gaps should be triaged after the initial wiring plan. Confirm priorities with stakeholders before implementation.
