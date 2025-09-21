Post-processing (Hysteresis → Morphology → Duration)

Scope
- Convert per‑timestep probabilities to discrete seizure events.

Pipeline
- Hysteresis: tau_on=0.86, tau_off=0.78 to reduce flicker and capture continuity; stability windows min_onset≈128, min_offset≈256 samples (at 256 Hz).
- Morphology: remove spurious spikes, fill short gaps.
- Duration filter: discard events shorter than minimal clinically relevant duration.
- Event generation: convert masks to interval events for evaluation.
 - Window stitching: overlap‑add to reconstruct full‑record timelines (60s windows, 10s stride).
 - Merging: merge gaps ≤ tau_merge (e.g., 2.0s). Confidence = mean/peak/percentile within event.

Code anchors
- src/brain_brr/post/postprocess.py (hysteresis, morphology, duration)
- src/brain_brr/events/* (event generation)

Cross-references
- Architecture spec: ../02-architecture/canonical-spec.md (post-processing section)
- Evaluation metrics: ./evaluation.md (TAES + FA/24h points)
- Data pipeline: ../01-data-pipeline/tusz-preflight.md (operational pitfalls)

Notes
- Tune thresholds on dev set; keep eval rules consistent with TAES.
