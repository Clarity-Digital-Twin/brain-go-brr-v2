# Evaluation Checklist (Phase 5)

- [ ] Data manifests present (train/dev/test) with durations
- [ ] Config frozen (hash recorded) for the run
- [ ] Seeds set (torch, numpy, python) and deterministic flags
- [ ] Model outputs (B×T probs) available for the split OR on‑the‑fly inference
- [ ] Post‑processing config consistent with Phase 4 spec
- [ ] Threshold search converges for FA targets {10, 5, 2.5, 1}
- [ ] Metrics JSON saved (TAES, AUROC, sensitivity@FA, FA curve)
- [ ] Threshold table saved (target → τ_on)
- [ ] CSV_BI events exported (Temple format verified by tests)
- [ ] Plots generated (FA curve, ROC; optional calibration)
- [ ] Metadata captured (commit, config hash, env summary)
- [ ] CI run executes CPU‑deterministically (fallback forced if gpu extra installed)

Optional (recommended):
- [ ] Per‑patient metric breakdown (variance analysis)
- [ ] Bootstrapped CIs for TAES and sensitivity@FA
- [ ] Calibration (ECE) and temperature scaling report
