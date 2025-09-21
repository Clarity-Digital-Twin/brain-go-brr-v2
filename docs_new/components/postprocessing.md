Post-processing (Hysteresis → Morphology → Duration)

Scope
- Convert timestep probabilities to events.

Code anchors
- src/brain_brr/post/postprocess.py (tau_on=0.86, tau_off=0.78; morphology; duration)
- src/brain_brr/events/* (event generation)

Docs
- phases/PHASE4_POSTPROCESSING.md
- implementation/EVALUATION_CHECKLIST.md (TAES + FA/24h points)

Notes
- Keep thresholds tuned on dev set, not eval.
