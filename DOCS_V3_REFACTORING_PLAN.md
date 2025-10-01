# docs_v3 Refactoring Plan

**Created**: October 1, 2025
**Target**: Transform 70 scattered docs into a clear, hierarchical system
**Framework**: Diátaxis (Divio Documentation System) + PyTorch patterns

---

## 🎯 Problem Statement

### Current State (docs_v3)
- **70 markdown files** across 11 directories
- **Inconsistent naming**: Some kebab-case, some ALL_CAPS, some mixed
- **Unclear hierarchy**: 00-09 numbered directories + reference/ subdirectory
- **Content overlap**: Multiple NaN troubleshooting docs, unclear which is canonical
- **Historical baggage**: 19 "reference" docs of unclear relevance

### Pain Points
1. ❌ **Discovery**: "Where do I find how to fix NaN losses?"
2. ❌ **Confidence**: "Is this doc up-to-date for v3.4.1?"
3. ❌ **Duplication**: Multiple docs about same topic (gradient monitoring, NaNs, training)
4. ❌ **Navigation**: 00-09 numbering doesn't match mental model

---

## 📚 Research: Industry Best Practices

### Diátaxis Framework (Standard for Technical Docs)
Separates documentation into **4 clear purposes**:

| Type | Purpose | Audience | Example |
|------|---------|----------|---------|
| **Tutorials** | Learning-oriented | New users | "Your First Training Run" |
| **How-To Guides** | Task-oriented | Practitioners | "How to Fix NaN Losses" |
| **Reference** | Information-oriented | Lookup | "Config Schema Reference" |
| **Explanation** | Understanding-oriented | Curious learners | "Why Eigendecomposition Matters" |

**Key Principle**: Each type has different **writing style**, **structure**, and **reader expectations**.

### PyTorch Documentation Pattern
- **Get Started** → Quick wins (tutorials)
- **Tutorials** → Comprehensive learning paths
- **Docs** → Reference material (API, configs)
- **Notes** → Explanations (architecture, design decisions)

### AWS/Google Standards
- **Single source of truth** per topic (no duplicates)
- **Explicit versioning** (last updated, codebase version)
- **Layered depth**: Overview → Details → Deep Dives
- **Clear status tags**: Stable, Experimental, Deprecated

---

## 🏗️ Proposed Structure (docs_v3)

```
docs_v3/
│
├── README.md                    # Navigation hub + quick links
│
├── 01-getting-started/          # TUTORIALS (learning-oriented)
│   ├── quickstart.md            # 5-minute smoke test
│   ├── first-training-run.md    # End-to-end walkthrough
│   └── understanding-outputs.md # Interpreting logs/metrics
│
├── 02-guides/                   # HOW-TO (task-oriented)
│   ├── installation/
│   │   ├── local-setup.md       # RTX 4090 setup
│   │   ├── modal-setup.md       # Cloud setup
│   │   └── gpu-troubleshooting.md
│   ├── training/
│   │   ├── local-training.md    # RTX 4090 workflows
│   │   ├── modal-training.md    # A100 workflows
│   │   ├── resume-checkpoint.md
│   │   └── smoke-testing.md
│   ├── troubleshooting/
│   │   ├── nan-losses.md        # SINGLE canonical NaN guide
│   │   ├── gradient-explosions.md
│   │   ├── modal-xid31.md
│   │   └── wsl2-issues.md
│   └── optimization/
│       ├── hyperparameter-tuning.md
│       └── performance-profiling.md
│
├── 03-reference/                # REFERENCE (information-oriented)
│   ├── architecture/
│   │   ├── v3-overview.md       # High-level architecture
│   │   ├── tcn.md               # Component deep-dive
│   │   ├── mamba.md
│   │   ├── gnn.md
│   │   ├── laplacian-pe.md
│   │   └── edge-features.md
│   ├── configuration/
│   │   ├── schema.md            # Complete config reference
│   │   ├── env-vars.md          # Environment variables
│   │   └── validation.md
│   ├── data/
│   │   ├── preprocessing.md
│   │   ├── cache-format.md
│   │   └── tusz-dataset.md
│   ├── cli/
│   │   ├── makefile.md          # make commands reference
│   │   └── python-api.md        # python -m src commands
│   └── metrics/
│       └── taes.md              # TAES evaluation details
│
├── 04-explanations/             # EXPLANATION (understanding-oriented)
│   ├── architecture/
│   │   ├── why-dual-stream.md   # Design rationale
│   │   ├── why-dynamic-pe.md
│   │   └── stability-evolution.md # v3.3.0 → v3.4.1 journey
│   ├── training/
│   │   ├── gradient-norms.md    # Why P95=20 is normal
│   │   ├── warmup-schedules.md  # When/why to use
│   │   └── balanced-sampling.md # Dataset strategy
│   └── operations/
│       ├── modal-architecture.md # Why persistent SSD
│       └── cache-design.md      # Manifest architecture
│
├── 05-development/              # For contributors
│   ├── coding-standards.md
│   ├── testing.md
│   ├── versioning.md
│   └── roadmap.md
│
└── archive/                     # Historical context (NOT user-facing)
    ├── incidents/               # Keep 3 incident reports
    ├── deprecated/              # Old approaches (pre-v3.3.1)
    └── README.md                # "Why this exists"
```

**Total**: ~45 files (down from 70 by merging duplicates)

---

## 🔄 Systematic Refactoring Process

### Phase 1: Audit & Categorize (1-2 hours)
**Goal**: Understand what we have and what's missing

**Tasks**:
1. ✅ List all 70 files with word counts
2. ✅ Categorize each into Diátaxis framework:
   - Tutorial (learning)
   - How-To (task)
   - Reference (lookup)
   - Explanation (understanding)
   - Archive (historical)
3. ✅ Identify duplicates (e.g., 5 NaN-related docs → 1 canonical)
4. ✅ Mark obsolete content (pre-v3.3.1 edge clamping, old configs)
5. ✅ Find gaps (missing: quickstart, hyperparameter tuning)

**Deliverable**: `DOCS_AUDIT.csv` with columns:
- Current path
- Category (Tutorial/HowTo/Reference/Explanation/Archive)
- Status (Keep/Merge/Deprecate/Obsolete)
- Target path
- Notes

### Phase 2: Create Structure & README (30 min)
**Goal**: Build skeleton + navigation hub

**Tasks**:
1. ✅ Create new directory structure (01-getting-started, 02-guides, etc.)
2. ✅ Write root README.md as **navigation hub**:
   ```markdown
   # Brain-Go-Brr Documentation (v3.4.1)

   ## 🚀 New Users Start Here
   - [5-Minute Quickstart](01-getting-started/quickstart.md)
   - [Your First Training Run](01-getting-started/first-training-run.md)

   ## 📖 Common Tasks
   - [Fix NaN Losses](02-guides/troubleshooting/nan-losses.md)
   - [Train on Modal (A100)](02-guides/training/modal-training.md)
   - [Optimize Performance](02-guides/optimization/performance-profiling.md)

   ## 📚 Reference
   - [V3 Architecture](03-reference/architecture/v3-overview.md)
   - [Config Schema](03-reference/configuration/schema.md)
   - [CLI Commands](03-reference/cli/makefile.md)

   ## 💡 Understanding the System
   - [Why Dual-Stream Architecture?](04-explanations/architecture/why-dual-stream.md)
   - [Gradient Norms Explained](04-explanations/training/gradient-norms.md)
   - [Stability Evolution (v3.3.0→v3.4.1)](04-explanations/architecture/stability-evolution.md)
   ```
3. ✅ Add "Last Updated" and "Codebase Version" to template

**Deliverable**: Empty structure + navigable README

### Phase 3: Migrate & Consolidate (3-4 hours)
**Goal**: Move files, merge duplicates, update cross-references

**Order**: Low-risk → High-risk
1. **Reference docs** (straightforward moves, minimal edits)
2. **How-To guides** (merge duplicates, update paths)
3. **Explanations** (extract from reference, rewrite for clarity)
4. **Tutorials** (create new from scattered content)

**For Each File**:
```bash
# Example: Migrating TCN reference
1. Read: docs_v3/04-model/tcn.md
2. Categorize: Reference (component deep-dive)
3. Edit:
   - Update header (last updated, version)
   - Fix cross-references (../02-guides/...)
   - Remove obsolete sections
   - Add "See also" links
4. Move: docs_v3/03-reference/architecture/tcn.md
5. Update: Root README + parent folder READMEs
```

**Consolidation Examples**:

**Before** (5 NaN docs):
```
08-operations/nan-prevention-complete.md      (23KB, comprehensive)
08-operations/nan-troubleshooting.md          (10KB, overlaps)
08-operations/nan-logits-dynamic-pe.md        (5KB, specific case)
08-operations/v3-nan-explosion-resolution.md  (12KB, historical)
08-operations/gradient-monitoring.md          (16KB, related)
```

**After** (2 docs):
```
02-guides/troubleshooting/nan-losses.md       (25KB, canonical how-to)
  └─ Merges: nan-prevention + nan-troubleshooting + current fixes
  └─ Links to: gradient-norms explanation, v3-nan-explosion archive

04-explanations/training/gradient-norms.md    (18KB, understanding)
  └─ Extracted from gradient-monitoring
  └─ Focus: Why P95=20 is normal, architecture-specific expectations

archive/incidents/v3-nan-explosion-sept24.md  (12KB, historical)
  └─ Preserved for context, not user-facing
```

**Deliverable**: All files migrated, duplicates merged, cross-references fixed

### Phase 4: Create Missing Content (2-3 hours)
**Goal**: Fill gaps identified in Phase 1

**Priority 1 (Critical)**:
1. `01-getting-started/quickstart.md` - 5-minute smoke test walkthrough
2. `02-guides/troubleshooting/nan-losses.md` - Canonical NaN guide (merged)
3. `03-reference/architecture/v3-overview.md` - High-level architecture (simplified)

**Priority 2 (Important)**:
4. `01-getting-started/first-training-run.md` - End-to-end tutorial
5. `04-explanations/architecture/stability-evolution.md` - Extract from current docs
6. `04-explanations/training/gradient-norms.md` - Explain architecture-specific norms

**Priority 3 (Nice-to-have)**:
7. `02-guides/optimization/hyperparameter-tuning.md` - Currently missing
8. `01-getting-started/understanding-outputs.md` - Log interpretation

**Deliverable**: 8 new/rewritten docs filling critical gaps

### Phase 5: Validation & Polish (1 hour)
**Goal**: Ensure quality and consistency

**Automated Checks**:
```bash
# Check for broken internal links
grep -r "\[.*\](.*\.md)" docs_v3/ | while read line; do
  # Validate each link exists
done

# Check for RECENT-WORK-SYNTHESIZED references
grep -r "RECENT-WORK-SYNTHESIZED" docs_v3/

# Check for ALL_CAPS file references (old naming)
grep -r "[A-Z_]{10,}\.md" docs_v3/

# Ensure all docs have headers
for file in $(find docs_v3 -name "*.md"); do
  head -1 "$file" | grep -q "^#" || echo "Missing header: $file"
done
```

**Manual Review Checklist**:
- [ ] Root README navigable (5 clicks to any doc)
- [ ] No duplicate content (same topic covered once)
- [ ] Clear "Last Updated" on all docs
- [ ] Consistent naming (kebab-case)
- [ ] Cross-references work (no 404s)
- [ ] Archive clearly marked as historical

**Deliverable**: Production-ready docs_v3

---

## 📊 Success Metrics

### Quantitative
- **Files**: 70 → ~45 (35% reduction by merging)
- **Duplicates**: 0 (single source of truth)
- **Broken links**: 0
- **Avg depth to find info**: ≤3 clicks from root

### Qualitative
- ✅ New user can start training in 5 minutes (quickstart)
- ✅ Clear which doc to read for any task
- ✅ Historical context preserved but separated
- ✅ Confident docs are up-to-date (explicit versioning)

---

## 🚨 Risks & Mitigations

### Risk 1: Breaking existing references
**Mitigation**: Keep docs_v2 intact, only delete after v3 validation

### Risk 2: Losing historical context
**Mitigation**: archive/ directory preserves everything, with README explaining purpose

### Risk 3: Too aggressive merging
**Mitigation**: Phase 3 creates MERGED_DOCS.md tracking what was combined (for rollback)

### Risk 4: Inconsistent voice/style
**Mitigation**: Use templates for each doc type (Tutorial/HowTo/Reference/Explanation)

---

## 🎬 Execution Plan

### Recommended Approach: **Phased with Checkpoints**

```bash
# Phase 1: Audit (can run now)
python scripts/audit_docs.py > DOCS_AUDIT.csv
# Review manually, adjust categories

# Phase 2: Structure (30 min)
# Create dirs + root README

# Phase 3: Migrate (2-3 hours)
# Process one category at a time:
#   1. Reference (easiest)
#   2. How-To (consolidation)
#   3. Explanations (extraction)
#   4. Tutorials (creation)

# Phase 4: Create Missing (2 hours)
# Priority 1 → Priority 2 → Priority 3

# Phase 5: Validate (1 hour)
# Run checks, manual review

# Total: ~7-10 hours spread over 2-3 days
```

### Checkpoint After Each Phase
- Commit changes
- Run validation script
- User review before proceeding

---

## 📝 Templates for New Docs

### Tutorial Template
```markdown
# [Task Name]

**Last Updated**: [Date]
**Codebase Version**: v3.4.1
**Estimated Time**: [X minutes]
**Prerequisites**: [List]

## What You'll Learn
[Bullet points of outcomes]

## Step 1: [First Step]
[Instructions with code blocks]

## Step 2: [Next Step]
...

## Verify It Worked
[Expected output]

## What's Next?
- [Link to next tutorial]
- [Link to related how-to]
```

### How-To Template
```markdown
# How to [Task]

**Last Updated**: [Date]
**Codebase Version**: v3.4.1
**Status**: Stable | Experimental | Deprecated

## Problem
[What user wants to accomplish]

## Solution
[Step-by-step instructions]

## Troubleshooting
[Common issues + fixes]

## See Also
- [Related docs]
```

### Reference Template
```markdown
# [Component/Feature Name]

**Last Updated**: [Date]
**Codebase Version**: v3.4.1

## Overview
[1-2 sentence description]

## API Reference
[Complete technical details]

## Examples
[Code snippets]

## See Also
- [Related references]
```

### Explanation Template
```markdown
# Why [Design Decision]

**Last Updated**: [Date]
**Context**: [When this was decided]

## The Question
[What we needed to solve]

## The Answer
[Our approach + rationale]

## Trade-offs
[What we gained vs. what we gave up]

## Historical Context
[How this evolved]

## See Also
- [Related explanations]
```

---

## 🎯 Next Steps

### Option A: Full Automated Audit (Recommended)
Run comprehensive analysis to generate DOCS_AUDIT.csv:
```bash
# Would you like me to create this script now?
python scripts/audit_docs_v3.py
```

### Option B: Manual Phase-by-Phase
Start with Phase 1 audit manually:
1. I categorize all 70 files
2. You review categorization
3. We proceed to Phase 2

### Option C: Proof of Concept
Refactor just one section (e.g., 02-guides/troubleshooting/) to validate approach before full migration.

---

**Recommendation**: Start with **Option A** (automated audit) to see exactly what we have, then review together before executing phases.

Would you like me to create the audit script now?
