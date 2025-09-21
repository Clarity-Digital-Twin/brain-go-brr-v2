Deployment Troubleshooting

Zero seizures detected (manifest)
- Symptom: `partial=0, full=0, none=N` after scan-cache; training collapses to all-negative
- Fixes:
  - Verify CSV_BI parsing (channel,start,stop,label,confidence)
  - Ensure seizure type set includes {gnsz,fnsz,cpsz,absz,spsz,tcsz,tnsz,mysz}
  - Rebuild cache; re-scan; only train if partial>0 or full>0

Hangs or deadlocks (WSL2)
- Cause: multiprocessing DataLoader
- Fix: `num_workers=0`; avoid pin_memory; keep cache/data on WSL ext4

CUDA/Mamba kernel mismatch
- Symptom: errors about d_conv sizes
- Fix: kernels coerce unsupported d_conv to 4; or set `SEIZURE_MAMBA_FORCE_FALLBACK=1`

EDF read failure
- Symptom: MNE ValueError/OSError on EDF header
- Fix: header repair path; see TUSZ/EDF_HEADER_REPAIR.md

Slow IO on Modal or WSL
- Fix: mount datasets on fast storage; avoid network-mounted paths for cache writes; keep local caches on ext4

Sampler used with balanced dataset
- Symptom: WeightedRandomSampler still applied
- Fix: training loop bypasses sampler for BalancedSeizureDataset; verify config’s balanced flag and logs

