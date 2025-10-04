# Implementation Gap Audit (2025-02-14)

## Overview
- Reviewed core training, data, and configuration paths after recent stability work.
- Several config surfaces still act as no-ops, creating mismatch between documentation and runtime behaviour.
- The items below focus on concrete code paths that need either wiring or deprecation to avoid reward-hacking style regressions.

## Findings

1. Training config fields for mid-epoch safety nets and gradient accumulation are ignored.
   - Evidence: `src/brain_brr/config/schemas.py:508-518` defines `mid_checkpoint_interval_s`, `mid_epoch_keep`, and `gradient_accumulation_steps`.
   - Runtime behaviour: `src/brain_brr/train/train_step.py:93-104` pulls checkpoint cadence exclusively from `BGB_MID_EPOCH_*` env vars, and `src/brain_brr/train/train_step.py:210-276` calls `optimizer.step()` every batch with no accumulation logic.
   - Impact: Modal configs advertising `gradient_accumulation_steps: 2` or 30-minute checkpointing simply do nothing.
   - Suggested fix: either implement true accumulation + config-driven checkpoint cadence, or prune these fields from the schema and configs.

2. Warmup schedule options are half-implemented.
   - Evidence: `src/brain_brr/config/schemas.py:452-470` exposes focal-loss and residual scaling warmup flags; `src/brain_brr/train/warmup.py:11-43` implements `get_focal_gamma` but nothing calls it, and no residual scaling hook exists.
   - Impact: Setting `warmup_schedule.focal_gamma_enabled` or `residual_scale_enabled` provides a false sense of protection; focal gamma stays fixed and residuals never scale.
   - Suggested fix: thread `get_focal_gamma` into `train/train_step.py` when `loss == "focal"`, and either remove or implement residual scaling in the model builders.

3. Optimizer/scheduler enum values are misleading.
   - Evidence: `src/brain_brr/config/schemas.py:494-499` allows `optimizer: "adamw" | "adam" | "sgd"`, but `src/brain_brr/train/optimizer_factory.py:22-60` raises `ValueError` unless the value is `adamw`. Similarly, `SchedulerConfig` permits `"linear"` and `"constant"` while `create_scheduler` in the same file only supports `"cosine"`.
   - Impact: Using the advertised enum values triggers runtime errors or silently falls back to defaults depending on call site.
   - Suggested fix: either restrict the schema to `"adamw"`/`"cosine"` or implement the additional optimizers and schedulers.

4. Experiment, logging, and evaluation toggles do not control anything.
   - Evidence: `src/brain_brr/config/schemas.py:550-581` defines `save_model`, `save_best_only`, `log_level`, and logging toggles, yet the training loop always saves checkpoints (`src/brain_brr/train/loop.py:272-320`) and logging cadence is driven by `src/brain_brr/constants.py:195-199`. `EvaluationConfig.save_predictions/save_plots` (`src/brain_brr/config/schemas.py:530-538`) are never read—the validation step only consumes `fa_rates` (`src/brain_brr/train/loop.py:200-248`).
   - Impact: Operators cannot turn off checkpointing, adjust log spam, or request prediction dumps despite configs claiming support.
   - Suggested fix: plumb these flags into the checkpoint writer, logging setup, and validation pipeline or delete them from user-facing configs.

5. Data and preprocessing configs are effectively cosmetic.
   - Evidence: `src/brain_brr/config/schemas.py:43-97` exposes dataset selection (`"tuh_eeg" | "chb_mit"`), sample/hour limits, and preprocessing knobs, but the training entrypoint unconditionally loads TUSZ splits (`src/brain_brr/train/loop.py:380-547`) and the preprocessing stack uses hard-coded defaults (`src/brain_brr/data/preprocess.py:10-52`).
   - Impact: Switching `data.dataset`, `max_samples`, or the bandpass/notch settings in YAML has no effect, which can mask configuration drift or validation experiments.
   - Suggested fix: thread these values through the dataset/IO stack or pare the schema back to the parameters we actually respect.

## Next steps
- Decide which of the above levers are genuinely needed; remove any legacy/no-op fields to keep the schema honest.
- For the remaining items, schedule wiring work (gradient accumulation, focal gamma warmup, logging controls) before the next training cycle so production configs match reality.
- Update documentation once the code reflects the intended behaviour to break the reward-hacking loop.
