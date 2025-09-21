Data I/O (TUSZ Ingestion)

Scope
- Read EDFs, parse CSV_BI annotations, enforce channel order, resample, normalize.

What we do (order of ops)
- Read EDF with MNE; repair rare headers when needed (see TUSZ/EDF_HEADER_REPAIR.md).
- Resample to 256 Hz.
- Apply bandpass 0.5–120 Hz and 60 Hz notch.
- Map channels to canonical 19‑ch order; handle synonyms (T7→T3, T8→T4, P7→T5, P8→T6).
- Window into 60s segments with 10s stride; per‑channel z‑score normalization.
- Filters: Butterworth bandpass (order=3) with lfilter; iirnotch at powerline; resample via scipy.signal.resample.
- Build per‑timestep binary label masks from CSV_BI events.

Shapes, units, dtypes
- Target shape per window: (19, 15360) → 60 s at 256 Hz.
- Batches: (B, 19, 15360).
- Units: microvolts (µV) after conversion from Volts.
- dtype: float32 throughout (windows and masks).

CSV_BI essentials
- Format: channel,start_time,stop_time,label,confidence
- Parser reads start/stop from columns 2/3 and label from column 4.
- Seizure label set (TUSZ v2.0.3): {seiz, gnsz, fnsz, spsz, cpsz, absz, tnsz, tcsz, mysz}.
- See TUSZ/CSV_BI_PARSER.md for details and pitfalls.

Code anchors
- src/brain_brr/data/io.py (parse_tusz_csv, events_to_binary_mask)
- src/brain_brr/data/loader.py
- src/brain_brr/data/dataset.py
- src/brain_brr/constants.py (channel order)

Docs
- TUSZ/DATA_FLOW.md (end‑to‑end pipeline)
- TUSZ/CSV_BI_PARSER.md (annotation parsing)
- TUSZ/CHANNELS_AND_MONTAGE.md (canonical ordering)
 - TUSZ/EDF_HEADER_REPAIR.md (rare colon→period date fix)

Related Phase
- phases/PHASE1_DATA_PIPELINE.md

Notes
- Keep repo/venv on WSL ext4 (not /mnt/c) to avoid import/I/O stalls.
- Before training, ensure scan‑cache reports partial>0 or full>0.
