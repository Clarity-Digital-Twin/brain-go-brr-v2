# P0124 Codebase Debt Audit

**Project**: Brain-Go-Brr v3.5.0
**Date**: October 3, 2025
**Scope**: Post-constants-refactor follow-up debt (no code changes yet)

---

## High-Priority Debt (Fix before production baseline)

- **Time conversion literals still scattered**  
  - `src/brain_brr/train/val_step.py:70`, `:156`; `src/brain_brr/eval/helpers/false_alarm.py:104`, `:174`; `src/brain_brr/eval/metrics.py:172`, `:364`, `:370`, `:502`, `:597`, `:613`, `:615` still hardcode `24.0`/`3600.0`.  
  - Replace with `HOURS_PER_DAY` / `SECONDS_PER_HOUR` to complete centralization and avoid silent drift if clinical definitions change.

- **Binary-search loop still hardcodes 10 iterations**  
  - `src/brain_brr/train/val_step.py:139` uses `range(10)`; `src/brain_brr/eval/helpers/false_alarm.py:33` defaults `max_iters: int = 10`.  
  - Swap to `THRESHOLD_SEARCH_MAX_ITERS` so evaluation and validation stay in lockstep if we tune precision later.

- **Config schemas not yet backed by constants**  
  - `src/brain_brr/config/schemas.py:27-58` still embed literals such as `256`, `60`, `10`, `0.5`, `120.0`, `60`.  
  - Use `Field(default=SAMPLING_RATE)` (and similar) so schema defaults remain a single source of truth.  
  - Consider replacing `Literal[256]` with `Literal[SAMPLING_RATE]` to keep typing guarantees while pulling from constants.

- **YAML configs require structured consistency checks**  
  - `configs/local/*.yaml`, `configs/modal/*.yaml` duplicate clinical constants (τ_on/off, morphology kernels, durations, FA targets).  
  - Add a `make config-audit` script that asserts each YAML value matches `constants.py` (read via Python) to avoid drift. Comments currently document intent but are not enforced.

## Medium-Priority Debt (Plan for v3.5.x)

- **Outdated config documentation**  
  - `configs/CONFIG_CONSISTENCY_CHECK.md` still reflects v3.2.0 defaults (batch_size=4, gradient_clip=0.1). Update or replace with generated docs once constants are authoritative.

- **Duplicate FA-sweep logic**  
  - `src/brain_brr/train/val_step.py:135-187` reimplements `find_threshold_for_fa_target`; the two paths can diverge again. Factor validation to call the helper directly once constants are centralized and dependency cycles are resolved.

- **Inline imports inside tight loops**  
  - `src/brain_brr/train/val_step.py:149`, `:171` import `batch_probs_to_events`/`batch_mask_to_events` inside per-record loops. Python caches modules but moving imports to module scope will make intent clearer and avoid linter suppressions.

## Low-Priority Debt (Track but not blocking)

- **Schema epsilon bounds**  
  - `src/brain_brr/config/schemas.py:474` still uses literal `ge=1e-6`. Tie to `EPSILON_NUMERICAL` for consistency with the new policy.

- **Metric-name string formatting**  
  - `src/brain_brr/train/val_step.py:177` formats `f"sensitivity_at_{fa}fa"` manually. Consider using `METRIC_SENSITIVITY` helpers or enums to avoid future typos.

---

## Training Readiness Check

- ✅ P0 bug (threshold bounds) is fixed.
- ⚠️ Constants rollout is only 80% complete: validation/eval paths still ignore several new constants, and configs can drift without automation.

**Recommendation**: Finish the High-Priority items above before kicking off expensive training. They require light edits but ensure evaluation, validation, and configs truly share a single source of truth.

