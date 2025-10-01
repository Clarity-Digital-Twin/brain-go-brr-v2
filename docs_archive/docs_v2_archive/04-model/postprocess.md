# Post-processing

File: `src/brain_brr/post/postprocess.py`

- Hysteresis τ_on=0.86, τ_off=0.78
- Morphological Opening(11), Closing(31)
- Valid duration: 3–600s; merge events within 2s

Config schema

- `HysteresisConfig`: `tau_on > tau_off` enforced; `min_onset_samples`, `min_offset_samples` available.
- `MorphologyConfig`: `opening_kernel`, `closing_kernel` must be odd.
- `DurationConfig`: `max_duration_s ≥ min_duration_s`.
- `EventsConfig`: `tau_merge` and confidence methods (mean, peak, percentile).

Usage

- Training/validation uses these configs to convert per‑sample probabilities to events and compute TAES.
