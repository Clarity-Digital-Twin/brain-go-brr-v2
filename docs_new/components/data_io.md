Data I/O (TUSZ Ingestion)

Scope
- EDF reading, CSV_BI parsing, channel mapping, resampling, normalization.

Code anchors
- src/brain_brr/data/loader.py
- src/brain_brr/data/dataset.py
- src/brain_brr/data/io.py (parse_tusz_csv, events_to_binary_mask)
- src/brain_brr/constants.py (channel order)

Docs
- TUSZ/CSV_BI_PARSER.md (channel,start,stop,label,confidence)
- TUSZ/CHANNELS_AND_MONTAGE.md (19-ch + synonyms)
- TUSZ/DATA_FLOW.md (order of operations)

Related Phase
- phases/PHASE1_DATA_PIPELINE.md

Notes
- Keep repo and venv on WSL ext4 (not /mnt/c) for reliability.
- Verify CSV_BI manifests show partial>0 or full>0 before training.
