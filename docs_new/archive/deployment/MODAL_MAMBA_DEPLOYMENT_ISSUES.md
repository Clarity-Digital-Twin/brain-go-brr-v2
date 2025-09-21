Archived: Modal Mamba-SSM Deployment Notes

This document has been consolidated into the canonical guidance:
- docs_new/deployment/MODAL_SSOT.md (CUDA/Mamba section)

Summary
- Compile mamba-ssm from source against PyTorch 2.2.2+cu121 in the CUDA 12.1 devel image.
- Install build tools first (build-essential, ninja); install packaging before build.
- Use `--no-build-isolation` and `CC=gcc CXX=g++` during pip install.
- Modal timeout max is 86400s.
- If CUDA kernels fail at runtime, set `SEIZURE_MAMBA_FORCE_FALLBACK=1` for Conv1d fallback.

Code anchor
- deploy/modal/app.py (image definition and train function)

Reason for archive
- The actionable content is integrated into the SSOT; this file is kept as a lightweight historical index to avoid duplication.
