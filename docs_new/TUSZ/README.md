TUSZ Documentation Suite

## Quick Command Reference

```bash
# Build or rebuild cache from EDF+CSV files
python -m src build-cache --data-dir <edf_root> --cache-dir <cache_dir>

# Scan cache and create manifest.json (categorizes windows)
python -m src scan-cache --cache-dir <cache_dir>

# Train (auto-uses BalancedSeizureDataset if manifest exists)
python -m src train <config.yaml>

# Quick smoke test
python -m src train configs/smoke_test.yaml
```

## Critical Information

**TUSZ v2.0.3 Seizure Types** (source: CSV_BI_PARSER.md):
- ✅ Valid: `seiz, gnsz, fnsz, cpsz, absz, spsz, tcsz, tnsz, mysz`
- ❌ Invalid: `spkz` (doesn't exist in v2.0.3)
- ⚠️ Cache rebuild required if seizure detection logic changes

## Documentation (Read in Order)

1) OVERVIEW.md — Dataset, goals, split usage
2) DATA_FLOW.md — Where each concern is handled in code
3) CSV_BI_PARSER.md — Correctly parsing TUSZ CSV_BI (channel,start,stop,label)
4) CHANNELS_AND_MONTAGE.md — 19-channel mapping and synonyms
5) CACHE_AND_SAMPLING.md — Manifest + BalancedSeizureDataset (SeizureTransformer ratios)
6) PREFLIGHT_AND_TROUBLESHOOTING.md — Stop conditions and fixes
7) EDF_HEADER_REPAIR.md — Handling rare broken EDF headers

Related postmortem:
- MYSZ_CRISIS_POSTMORTEM.md — rare seizure label coverage and cache rebuild decision
