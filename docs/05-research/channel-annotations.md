# Channel-Annotation Integration Plan (Channel-Head + Aggregator)

Objective
- Add an interpretable, channel-aware path inspired by “Channel-Annotated Deep Learning for Enhanced Interpretability in EEG-based Seizure Detection” and fuse it with our Bi‑Mamba‑2 + U‑Net + ResCNN pipeline.
- Preserve current public APIs; all new functionality must be optional via config flags.
- Deliver channel heatmaps and channel-localization scores while keeping O(N) global modeling.

Key Ideas From Paper
- Two-block design:
  - Block 1: Per-channel seizure likelihood from single‑channel 1D‑CNN → shallow Transformer; 1 s segments; interpretable heatmap.
  - Block 2: 3‑layer MLP aggregates channel probabilities per segment; DeepSHAP ranks channel contributions.
- Preprocessing: 10–20 system; resample 250 Hz; bandpass 0.1–60 Hz; notch 50/60; bipolar montage (22 pairs) matching TUSZ annotation format.
- Architecture specifics: 1D‑CNN kernels [3, 5, 10], max‑pool between blocks; ViT depth=2, 12 heads; ensemble MLP for final decision.
- Montage (22 bipolar pairs):
  - FP1-F7, F7-T3, T3-T5, T5-O1, FP2-F8, F8-T4, T4-T6, T6-O2, A1-T3, T3-C3, C3-CZ, CZ-C4, C4-T4, T4-A2, FP1-F3, F3-C3, C3-P3, P3-O1, FP2-F4, F4-C4, C4-P4, P4-O2.

Mapping To Our Repository
- Primary model remains SeizureDetector (Bi‑Mamba‑2 + U‑Net + ResCNN) producing per‑sample logits, then post‑processed via hysteresis + morphology + duration.
- We add a channel‑head producing per‑channel probabilities and an optional aggregator MLP that fuses channel probabilities into a segment/timestep probability that gates or blends with the main detector output.
- All additions are behind config flags; defaults preserve current behaviors and tests.

Planned Additions
1) Data: bipolar derivation + per‑channel labels
- File: `src/brain_brr/data/io.py`
  - Add a function `compute_bipolar_pairs(raw: mne.io.BaseRaw, pairs: list[tuple[str,str]]) -> np.ndarray` to compute 22 bipolar channels from raw referential montage when available.
  - Add default `TUSZ_BIPOLAR_PAIRS` constant (22 tuples) and helpers to resolve synonyms (A1/A2 naming variants).
- File: `src/brain_brr/data/datasets.py`
  - Extend `_process_file` to optionally compute and cache bipolar signals alongside our canonical 19‑ch referential array.
  - When TUSZ channel annotations are present (CSV_BI), create 1 s channel‑wise labels aligned to the 22 pairs for Block 1 supervision.
- File: `src/brain_brr/data/windows.py`
  - Add utility to downsample sequences to 1 s grid at 256 Hz: `reshape_seconds(x, fs=256) -> (T_seconds, ...)` for efficient Block 1 targets.

2) Config: feature flags + hyperparameters
- File: `src/brain_brr/config/schemas.py`
  - Add `ChannelHeadConfig`:
    - `enabled: bool = False`
    - `use_bipolar_22: bool = True`
    - `cnn_kernels: list[int] = [3,5,10]`
    - `cnn_channels: int = 64` (per‑channel branch width)
    - `transformer_depth: int = 2`, `transformer_heads: int = 12`
    - `output_stride_seconds: int = 1` (1 s resolution)
  - Add `AggregatorConfig`:
    - `enabled: bool = False`
    - `hidden: list[int] = [64, 32]`, `dropout: float = 0.1`
    - `input_channels: int = 22`, `topk_channels: int = 8` (for reporting)
  - Add both under `ModelConfig` as optional sections.
  - Preprocessing ablation knobs: allow `bandpass=(0.1, 60.0)` and dataset‑specific `notch_freq` (TUSZ=60, AU/EU=50).

3) Models: channel‑head and aggregator modules
- New: `src/brain_brr/models/channel_head.py`
  - `ChannelHead1D`: inputs (B, C_bipolar, T); depthwise or grouped 1D‑CNN with kernels [3,5,10], max‑pool; optional shallow transformer encoder (depth=2, 12 heads) over temporal tokens; output per‑second logits (B, C_bipolar, T_sec).
  - Efficient variant: 1D stride‑256 convs to produce 1 sample per second at 256 Hz without heavy tokenization.
- New: `src/brain_brr/models/aggregator.py`
  - `ChannelAggregatorMLP`: inputs per‑second channel probabilities (B, T_sec, 22) → per‑second logit (B, T_sec); 3‑layer MLP with dropout.
  - Expose top‑k contributing channels per timestep via input‑gradient or SHAP hooks.
- File: `src/brain_brr/models/detector.py`
  - Optionally instantiate ChannelHead1D and ChannelAggregatorMLP when enabled in config.
  - Fusion strategies (configurable):
    - Gate: `p_final = sigmoid(main_logits) * sigmoid(agg_logits_upsampled)`
    - Blend: `p_final = α*sigmoid(main_logits) + (1-α)*sigmoid(agg_logits_upsampled)` with learned or fixed α.
  - Upsample aggregator outputs from 1 s grid to sample grid (nearest or linear) before fusion.

4) Training: multi‑task loss and sampling
- File: `src/brain_brr/train/loop.py`
  - When channel‑head enabled and channel labels available (TUSZ): train with multi‑task loss:
    - `L_total = L_window + λ_ch * L_channel`
    - `L_window`: current per‑sample BCE (after stitching to windows as today)
    - `L_channel`: BCE on per‑second channel logits vs channel‑wise 1/0 labels
    - `λ_ch` configurable (default 0.3–0.5)
  - Balanced sampling unchanged; add optional “channel‑positive emphasis” sampler if needed.

5) Evaluation: channel localization and reporting
- New: `src/brain_brr/eval/channel_eval.py`
  - Compute channel‑localization sensitivity: for predicted segments/events, compare top‑k channels vs TUSZ annotated channels per segment/event.
  - Report sensitivity by seizure type where labels permit.
- File: `src/brain_brr/eval/metrics.py`
  - Add hooks to include channel‑localization metrics in evaluation summary when enabled.
- File: `src/brain_brr/events/events.py`
  - Extend event objects to optionally carry channel heatmaps aggregated over event duration (mean/max across per‑second maps).

6) Post‑processing & exports
- File: `src/brain_brr/post/postprocess.py`
  - No change to core hysteresis. Provide optional smoothing of aggregator output on 1 s grid before fusion (moving average 3–5 s) to stabilize.
- New: `src/brain_brr/utils/visualize.py`
  - Rendering utilities for channel heatmaps per event and per 60 s window (PNG for reports).
- File: `src/cli.py`
  - Extend `evaluate` to optionally save channel heatmaps, top‑k channels, and CSV_BI with channel scores.

7) Interpretability (DeepSHAP or Captum fallback)
- New: `src/brain_brr/eval/explain.py`
  - If SHAP available: DeepSHAP over `ChannelAggregatorMLP` to rank channels by contribution per timestep/segment.
  - CPU‑friendly fallback: Captum Integrated Gradients or InputXGradient on the MLP.
  - Aggregate over events to report per‑event top‑k channel list and sensitivity against TUSZ channel annotations.

8) Configs & experiments
- New example configs in `configs/` (not enabled by default):
  - `configs/experiments/channel_head_local.yaml`
    - preprocessing.bandpass=(0.1,60.0); postprocessing default; channel_head.enabled=true; aggregator.enabled=false
  - `configs/experiments/channel_head_agg_fusion.yaml`
    - as above, plus aggregator.enabled=true and fusion=gate
- CI stays with cpu‑only path (no GPU extras); channel‑head/aggregator are pure PyTorch CPU‑compatible.

Losses, Shapes, and Contracts
- Inputs: `SeizureDetector` still consumes (B, 19, 15360) windows @256 Hz.
- Channel‑head input: (B, 22, 15360) if `use_bipolar_22`; otherwise (B, 19, 15360) with a fixed 22‑map projection disabled.
- Channel‑head output: (B, 22, 60) logits on 1 s grid.
- Aggregator output: (B, 60) logits on 1 s grid; upsample → (B, 15360) for fusion.
- Event reporting: add per‑event `topk_channels: list[str]` and `channel_heatmap: dict[str, float]`.

Acceptance Criteria
- Training/inference unaffected when new flags are disabled.
- With channel‑head enabled on TUSZ:
  - Channel‑level BCE converges and channel‑heatmaps look clinically plausible on validation.
  - Channel‑localization sensitivity (top‑k overlap with annotated channels) ≥ 0.55 avg across events.
- With aggregator enabled (gate fusion):
  - Meets or exceeds TAES at 10/5/1 FA/24h vs baseline, or no worse than −1% absolute.
  - Qualitative reductions in false positives on non‑focal channels; improved calibration (ECE not worse).

Ablations (Paper‑Aligned)
- Preprocessing:
  - bandpass 0.1–60 vs 0.5–120 (current default) with dataset‑specific notch.
- Montage:
  - 22 bipolar vs 19 referential to assess impact on channel localization.
- Block settings:
  - 1D‑CNN kernels [3,5,10] vs [3,5,7]; transformer on/off (depth=0 vs 2).
- Fusion:
  - Gate vs blend; smoothing kernel 3 s vs 5 s before fusion.

Testing Plan
- Unit tests (pytest):
  - `data/io`: bipolar derivation correctness (synthetic sine leads), synonym mapping, 1 s label creation.
  - `models/channel_head`: shape contracts, 1 s mapping, grad flow with BCE.
  - `models/aggregator`: shape contracts, determinism, top‑k extraction.
  - `eval/channel_eval`: metric correctness on synthetic annotations.
- Integration tests:
  - End‑to‑end batch with flags enabled produces events and channel reports without API breakage.

Risks & Mitigations
- A1/A2 availability: TUSZ often includes auricular references; if missing, skip A1‑/A2‑derived pairs with a warning and adjust input_channels.
- 250 vs 256 Hz mismatch: keep 256 Hz canonical; labels mapped on 1 s grid are robust to tiny resample differences.
- Compute overhead: channel‑head runs once per window; keep transformer shallow or optional.

Execution Milestones
- M1: Data + labels + bipolar cache
- M2: Channel‑head forward path + loss
- M3: Aggregator + fusion path
- M4: Eval + channel‑localization metrics + visuals
- M5: Ablations + thresholds to hit TAES at 10/5/1 FA/24h

Open Questions / Defaults
- Default fusion: gate with α=1.0 (pure gate) or learnable α? Proposed: start with gate.
- Channel‑head transformer: off by default to minimize latency; enable in ablations.
- λ_ch weighting: start 0.3; tune 0.2–0.6 on validation.

References
- literature/markdown/channel_annotation_DL/channel_annotation_DL.md (extracted figures and details)

Next Steps
- Confirm we should target 22‑channel bipolar first; if yes, I’ll scaffold config stubs + empty modules and a small test for bipolar derivation.
