# Reference Documentation

**Last Updated**: October 1, 2025
**Purpose**: Historical context, incident reports, and development plans

---

## What's Here

This directory contains **historical and reference material** that provides context for architectural decisions but is not required for day-to-day usage.

### Structure

```
reference/
├── incidents/          # Historical incident reports (3 files)
│   ├── modal-xid31-recurrence.md           # XID 31 GPU crash investigation
│   ├── modal-cuda-failure-analysis.md      # A100 CUDA failure analysis
│   └── pytorch-upgrade-incident.md         # PyTorch 2.5.0 upgrade issues
│
├── development/        # Development plans & histories (13 files)
│   ├── PR implementation plans (PR1, PR2, PR4, PR708)
│   ├── Stack upgrade plans (PyTorch, Mamba)
│   ├── Test suite fixes
│   └── Training comparisons (local vs cloud)
│
└── meta/               # Documentation process (2 files)
    ├── docs-status-october-2025.md         # Documentation evolution
    └── docs-update-summary-oct1.md         # Update summary
```

---

## User-Facing Docs (Use These Instead)

The **synthesized, user-facing content** has been integrated into the main docs:

| Original (RECENT-WORK-SYNTHESIZED) | New Location (00-09) |
|------------------------------------|----------------------|
| 01-nan-and-stability/GRADIENT_MONITORING_GUIDE.md | **08-operations/gradient-monitoring.md** |
| 01-nan-and-stability/ARCHITECTURE_V3_STABILITY.md | **04-model/v3-stability-evolution.md** |
| 02-warmup-schedules/WARMUP_SCHEDULES_GUIDE.md | **05-training/warmup-schedules.md** |
| 01-nan-and-stability/NAN_PREVENTION_COMPLETE.md | **08-operations/nan-prevention-complete.md** (already existed, updated) |

---

## When to Use This Directory

### Incidents (incidents/)

**Use when**:
- Troubleshooting similar Modal issues
- Understanding XID 31 root causes
- Learning from past PyTorch upgrade problems

**Examples**:
- "Why did Modal training crash with XID 31?" → Read `modal-xid31-recurrence.md`
- "What issues arose with PyTorch 2.5.0?" → Read `pytorch-upgrade-incident.md`

### Development (development/)

**Use when**:
- Understanding why a PR was implemented
- Learning about stack upgrade decisions
- Reviewing test suite changes
- Comparing local vs cloud training strategies

**Examples**:
- "Why was boundary normalization added?" → Read `PR1_BOUNDARY_NORMALIZATION_PLAN.md`
- "How was PR #708 applied?" → Read `PR708_APPLICATION.md`
- "What changed in PyTorch 2.5.0 upgrade?" → Read `STACK_UPGRADE_PLAN_V3.md`

### Meta (meta/)

**Use when**:
- Understanding documentation evolution
- Reviewing how docs were synthesized
- Learning the consolidation process

---

## DO NOT Use for Implementation

**These docs are HISTORICAL REFERENCE ONLY.**

For actual implementation guidance, use the main docs:
- **Architecture**: `docs_v2/04-model/v3-architecture.md`
- **Training**: `docs_v2/05-training/`
- **Operations**: `docs_v2/08-operations/`
- **Configuration**: `docs_v2/03-configuration/`

---

## Maintenance

This directory is **stable** - files here are rarely updated since they document past events and decisions.

If you find yourself repeatedly referencing a document here, consider whether its content should be extracted into the main user-facing docs.

---

## Related

- **Main docs**: `docs_v2/00-overview/` through `docs_v2/09-development/`
- **Quick start**: `docs_v2/00-overview/overview.md`
- **Installation**: `docs_v2/01-installation/`
