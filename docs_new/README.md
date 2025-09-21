Brain-Go-Brr v2 Docs (Canonical)

This folder is the single source of truth for project docs going forward. It replaces scattered, historical notes with concise, developer-focused references that map 1:1 to the current codebase and CI.

Structure:

- TUSZ/ — Everything to ingest, cache, sample, and train with TUSZ
- components/ — Canonical, code-aligned docs per component (replacing Phase docs)
- deployment/ — Local (WSL2/Linux) and Modal guides
- architecture/ — Canonical model and pipeline specs
- implementation/ — Preprocessing, evaluation checklist, benchmarks, setup notes
- WSL2/ — Deep dives for Windows + WSL2 specifics (linked from deployment)
- future_work/ — Plans and research directions
- archive/ — Historical docs and postmortems

Start here (TUSZ):

- TUSZ/OVERVIEW.md
- TUSZ/DATA_FLOW.md
- TUSZ/CSV_BI_PARSER.md
- TUSZ/CHANNELS_AND_MONTAGE.md
- TUSZ/CACHE_AND_SAMPLING.md
- TUSZ/PREFLIGHT_AND_TROUBLESHOOTING.md
- TUSZ/EDF_HEADER_REPAIR.md

Deployment:

- deployment/PREFLIGHT.md
- deployment/LOCAL_WSL2.md
- deployment/MODAL_SSOT.md
- deployment/TROUBLESHOOTING.md

Also see:
- DOCS_SSOT.md — canonical commands and entry points
- HISTORY.md — archive index and replacements
 - components/README.md — migration map from Phases to Components
