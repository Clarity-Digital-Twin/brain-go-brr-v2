# Time–Frequency Hybrid Strategy (TCN/Mamba2 + Lightweight STFT)

Purpose

- Complement the time‑domain backbone (TCN + BiMamba2) with explicit frequency cues that are cheap and clinically interpretable.
- Aligns with 2025 consensus: hybrid time–frequency approaches generally outperform either view alone for EEG seizure detection.

Summary

- Keep current backbone unchanged.
- Add a lightweight, 3‑band STFT side‑branch and fuse late with electrode features before the GNN.
- Alternative: spectral‑aware loss (à la EEGM2) to preserve spectra without a new branch.

Recommended Side‑Branch (low overhead)

- Bands: theta/alpha (4–12 Hz), beta/gamma (14–40 Hz), HFO (80–250 Hz)
- Features: log‑magnitude per band; optionally per‑channel bandpower
- Small conv stack on branch outputs; late fusion with node features just before `proj_to_electrodes`

Implementation sketch

- Compute 3 band features from `(B,19,15360)` input (per‑channel STFT or bandpass+envelope).
- Concatenate with TCN features immediately before `proj_to_electrodes` in V3 path.
- Keep memory impact <10% by using coarse bands and 1×1 convs.

Evaluation plan

- A/B compare vs baseline on TAES targets.
- Expect +1–2% AUROC with <10–15% compute overhead.
- If no win, revert to backbone‑only; if marginal win, prefer spectral‑aware loss variant.

Notes

- Clinical interpretability improves (bands are familiar to clinicians).
- Hybrid is robust to failure modes where one view under‑performs.

Cross‑refs

- See `STFT_SIDEBRANCH_IMPLEMENTATION.md` for a code‑level patch sketch.
- Model context: `docs/04-model/v3-architecture.md`

