# Bug Archive Resolution Status

Last updated: 2025-09-21

This index summarizes the status of archived bug reports and points to the concrete code that resolved or mitigated each issue. For any item marked Mitigated/Partial, see the note for remaining risks.

- CRITICAL_CACHE_FUCKUP_POSTMORTEM.md — RESOLVED
  - Fix: TUSZ CSV_BI parser corrected; seizure labels expanded
  - Files: src/brain_brr/data/io.py:220, src/brain_brr/data/io.py:294
  - Guards: scan-cache/build-cache exit non‑zero on 0 seizures (src/brain_brr/cli/cli.py:212)
  - Data balance: BalancedSeizureDataset + manifest (src/brain_brr/data/datasets.py:190, src/brain_brr/data/cache_utils.py:52); training integration (src/brain_brr/train/loop.py:1061)

- P0_TUSZ_SAMPLING_FIX.md — RESOLVED
  - Fix: Pre-categorization manifest + BalancedSeizureDataset adopted (exact D = Dps ∪ D*fs ∪ D*ns)
  - Files: src/brain_brr/data/cache_utils.py:52, src/brain_brr/data/datasets.py:190, src/brain_brr/train/loop.py:1061

- P0_TUSZ_CHANNEL_SELECTION.md / P0_TUSZ_CHANNELS.md — RESOLVED
  - Fix: Robust channel cleaning + synonym mapping + ordered picking
  - Files: src/brain_brr/data/io.py:126 (clean_tusz_name), src/brain_brr/constants.py:33 (CHANNEL_SYNONYMS), src/brain_brr/utils/pick_utils.py:30

- TUSZ_EDF_HEADER_FIX.md / TUSZ_HEADER_FIX_INTEGRATION.md — RESOLVED
  - Fix: Header repair for malformed startdate with retry
  - Files: src/brain_brr/data/io.py:34 (_repair_edf_header_inplace), used in load path at src/brain_brr/data/io.py:86

- CRITICAL_FOCAL_LOSS_IMBALANCE.md / BUG_POSTMORTEM_FOCAL_LOSS.md — RESOLVED
  - Fix: Focal loss alpha neutral (0.5); anti‑double‑count with pos_weight; defaulted in configs
  - Files: src/brain_brr/train/loop.py:323 (focal vs pos_weight logic), configs/smoke_test.yaml: training.loss=focal, focal_alpha=0.5

- P0_TRAINING_PIPELINE_INVESTIGATION.md — RESOLVED
  - Fix: Stable training loop with preflight, AMP, scheduler, checkpointing; balanced data
  - Files: src/brain_brr/train/loop.py (main pipeline)

- P0_PYTORCH_HANG_INVESTIGATION.md — MITIGATED
  - Fix: WSL‑safe defaults (num_workers=0, pin_memory=false), spawn start method, optional persistence
  - Files: src/brain_brr/config/schemas.py: DataConfig defaults; src/brain_brr/train/loop.py:26 (spawn)
  - Risk: If users override to high worker counts on WSL2, hangs can recur.

- P0_CACHE_MISMATCH_ISSUE.md — MITIGATED
  - Fix: build‑cache/scan‑cache report hard counts; train warns on data vs experiment cache dir mismatch; manifest guard refuses training when seizures==0
  - Files: src/brain_brr/cli/cli.py:212, src/brain_brr/train/loop.py:1039–1058
  - Risk: Users must align config.data.cache_dir and experiment.cache_dir; tooling warns but doesn’t auto‑rewrite.

- P1_CACHE_STRATEGY_OSS_BLOCKER.md — PARTIAL
  - Fix: Local manifest + balancing reduces wasted compute; clear CLI to build/scan
  - Remaining: OSS distribution/storage strategy outside project scope.

- BLOCKERS.md — SUPERSEDED
  - Superseded by the specific items above; see these entries for current status.

- LOCAL_CACHE_INVENTORY.md / TEST_* docs — INFORMATIONAL
  - Not bugs; retained for reference.

