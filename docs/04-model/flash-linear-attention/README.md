# Flash Linear Attention (FLA) Research Documentation

**Status**: 🟢 Active Research - Production Stack (v4.0.0)
**Location**: `docs/04-model/flash-linear-attention/`
**Parent Documentation**: `docs/04-model/mamba.md` (canonical architecture reference)

---

## Purpose

This folder contains **detailed research documentation** for the Flash Linear Attention (BiGatedDeltaNet) variant of the V3 dual-stream architecture. While the high-level architecture is described in `/docs/04-model/mamba.md`, this folder preserves:

- Validation methodology and phase-by-phase implementation logs
- Architectural hypotheses and experimental design
- Detailed configuration analysis (head constraints, alignment requirements)
- Research decisions and trade-off discussions

**For operational use** (training configs, troubleshooting), see the main documentation tree.

---

## Document Index

### Core Research Documents

1. **FLASH_LINEAR_ATTENTION_RESEARCH.md** (Doc 0)
   - Master research memo with hypotheses and methodology
   - Architectural rationale for BiGatedDeltaNet over BiMamba2
   - Performance predictions and O(N) complexity analysis

2. **FLA_QUICK_REFERENCE.md**
   - Operational status tracker (BiMamba2 vs FLA training progress)
   - Quick decision tree for when to use each stack

3. **FLA_ROADMAP.md**
   - Phase-by-phase validation plan (Phase 0 → 1a → 1b → 2)
   - Historical workflow documentation (preserved for reproducibility)

### Validation Logs (Detailed Evidence)

4. **FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md**
   - Phase 1a: Edge stream validation
   - BiMamba2 (10k params) → BiGatedDeltaNet (30k params) migration
   - Alignment constraint discovery (`edge_mamba_d_model: 32`)

5. **FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md**
   - Phase 1b: Node stream validation
   - 0.75× head constraint implementation
   - Parameter reduction analysis (398k → 284k)

6. **FLASH_LINEAR_ATTENTION_DOC3_FULL_MIGRATION.md**
   - Phase 2: Full dual-stream validation
   - Integration testing and smoke test results
   - WSL2 SIGBUS discovery timeline

7. **FLASH_LINEAR_ATTENTION_DOC4_HYBRID_SWA.md**
   - Future research: Hybrid GDN-H1 with sliding window attention
   - Deferred until baseline comparison complete

---

## Integration with Main Docs

**Architecture Overview**: `docs/04-model/mamba.md`
- Describes both BiMamba2 and FLA streams
- Configuration examples and operational notes
- Parameter counts and alignment constraints

**Configuration Details**: `docs/03-configuration/`
- `configs/local/{smoke,train}_fla.yaml` — Local RTX 4090
- `configs/modal/{smoke,train}_fla.yaml` — Modal A100

**Installation**: `docs/01-installation/gpu-stack.md`
- FLA requires `flash-linear-attention>=0.3.0,<0.4.0` (per `pyproject.toml`)
- Triton dependency notes (3.1.0 vs 3.2.0 compatibility)

**Training Notes**: `docs/05-training/local.md`, `docs/05-training/modal.md`
- FLA-specific configurations (batch sizes, mixed precision)
- WSL2 cache requirements (ext4 filesystem mandatory)

---

## Current Status (v4.3.0)

### BiMamba2 Stack (Baseline)
- **Training**: Modal A100, **PAUSED at Epoch 6** due to high costs
- **Status**: Baseline established (6 epochs @ $1,118 total cost)
- **Purpose**: High-end comparison baseline (when budget permits)

### FLA Stack (Research)
- **Training**: Local RTX 4090, **Exp4 COMPLETE** (78 epochs, best epoch 63)
- **Status**: PRIMARY validated stack (cost-effective at $0)
- **Held-out TUSZ eval**: 35.9% sensitivity @ 10 FA/24h (AUROC 0.8654)
- SSOT: `results/local_fla_exp4_cyclic/eval_results_v2.json`
- **Purpose**: Production candidate for clinical EEG seizure detection

### Next Milestone
- Add a 4 FA/24h operating point + run official NEDC scoring for publication-ready results

---

## Why Keep This Folder?

**Research Reproducibility**: Detailed validation logs provide evidence trail for:
- Design decisions (why BiGatedDeltaNet over alternatives)
- Constraint discovery (alignment requirements, head ratios)
- Smoke test methodology (3-file local, 50-file Modal)

**Future Work**: Hybrid architectures (Doc 4) and alternative SSM variants benefit from preserved experimental methodology.

**Audit Trail**: If FLA underperforms, this folder documents *why we tried* and *how we validated*, which is publishable regardless of outcome.

---

**For day-to-day training and operations, use the canonical `/docs/` structure. This folder is the research lab notebook.**
