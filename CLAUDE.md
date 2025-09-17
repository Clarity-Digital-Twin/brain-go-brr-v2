# CLAUDE.md - Brain-Go-Brr v2 - AI Agent Instructions

## Project: First-Ever Bi-Mamba-2 + U-Net + ResCNN for Clinical EEG

**Revolutionary**: World's first bidirectional Mamba-2 SSM + U-Net CNN + ResNet for seizure detection. O(N) complexity vs transformers' O(N²).

## Critical Commands - RUN THESE

```bash
make q          # MUST run after EVERY code change - quality check
make t          # Test fast (no coverage)
make l          # Auto-fix linting issues
make train-local # Local training run
uv run train    # Direct training access
```

## Architecture Specifications

| Component | Specification |
|-----------|--------------|
| Input | 19-channel EEG @ 256 Hz |
| U-Net Encoder | 4 stages [64, 128, 256, 512], ×16 downsample |
| ResCNN | 3 residual blocks, multi-kernel [3, 5, 7] |
| Bi-Mamba-2 | 6 layers, d_model=512, d_state=16, bidirectional |
| Hysteresis | τ_on=0.86, τ_off=0.78 (FA reduction) |
| Output | Per-timestep probabilities @ 256 Hz |

## Clinical Targets (TAES @ NEDC)

- **10 FA/24h**: >95% sensitivity (clinical deployment)
- **5 FA/24h**: >90% sensitivity (ICU standard)
- **1 FA/24h**: >75% sensitivity (home monitoring)

## Workflow - YOU MUST FOLLOW

1. **ALWAYS** read existing code before modifying
2. **ALWAYS** run `make q` after EVERY code change
3. **NEVER** write comments unless explicitly requested
4. **ALWAYS** use type hints for all functions
5. **ALWAYS** write unit tests for new functions
6. **ALWAYS** follow patterns in neighboring files

## Project Structure

```
src/experiment/     # Core modules - DO NOT break APIs
├── schemas.py      # Pydantic configs
├── data.py        # EEG preprocessing pipeline
├── models.py      # Bi-Mamba-2 implementation
├── pipeline.py    # Training orchestration
└── infra.py       # Infrastructure utilities

configs/           # YAML experiment configs
tests/            # Pytest suite (unit/integration/slow markers)
data/            # Datasets (git-ignored)
results/         # Outputs (git-ignored)
```

## Data Pipeline

1. **MNE**: Read EDF files, channel selection, montage
2. **scipy**: Bandpass 0.5-120 Hz, 60 Hz notch, resample 256 Hz
3. **Window**: 60s windows, 10s stride (50s overlap)
4. **Normalize**: Per-channel z-score over full recording

## Tech Stack

- **Python 3.11+** with UV package manager (10-100x faster than pip)
- **PyTorch 2.0+** for deep learning
- **mamba-ssm 2.0+** for Bi-Mamba-2 implementation
- **MNE 1.5+** for EEG I/O
- **Ruff** for formatting/linting (replaces Black/isort/flake8)
- **mypy** strict mode for type checking

## Training Strategy

- **Train**: TUH EEG Seizure Corpus (from scratch - no pretrained weights)
- **Validate**: CHB-MIT dataset
- **Evaluate**: epilepsybenchmarks.com
- **Note**: Cannot use SeizureTransformer weights (architecture mismatch)

## Development Rules

1. **Code Style**: Line length 100, 4-space indent, snake_case
2. **Imports**: standard → third-party → first-party (sorted)
3. **Testing**: pytest with markers (@pytest.mark.unit/integration/slow)
4. **Commits**: Conventional (feat:, fix:, docs:, test:)
5. **Caching**: ExCa uses config hash - change config to refresh cache

## Critical Notes

- **Quality**: Run `make q` BEFORE considering any task complete
- **APIs**: Preserve public APIs in `src/experiment/`
- **WSL**: If on WSL, `export UV_LINK_MODE=copy`
- **GPU**: For Mamba-2 GPU support: `uv sync -E gpu`
- **Focus**: We optimize TAES, not accuracy

## Key Files Reference

- README.md: Full architecture details
- pyproject.toml: Dependencies and tool configs
- configs/local.yaml: Experiment configuration
- Makefile: All available commands

## Git Workflow

- Conventional commits ONLY
- Run `make q` before EVERY commit
- Include tests with code changes
- Update docs with API changes

---
**Mission: Shock the world with O(N) clinical seizure detection that actually works.**