# FLA Documentation Methodology: Cross-Referenced Accuracy Validation

**Date**: October 8, 2025
**Status**: 🔄 IN PROGRESS - Docs being updated for Phase 1b/2 consistency
**Purpose**: Document the rigorous methodology used for FLA (Flash Linear Attention) documentation revision to ensure 1000% accuracy, zero magic numbers, and seamless codebase integration.

**Note**: Previous "ironclad" status was premature. Issues found and being addressed (Oct 8, 2025):
- Doc 2: Parameter analysis needs update (edge_d_model=32 impact)
- Doc 3: Infrastructure status outdated (Phase 1a complete, not "doesn't exist")
- Doc 4: edge_mamba_d_model hardcoded to 16 (should be 32)
- Implementation status: Phase 1b/2 configs marked TODO (they exist)

---

## 🎯 Methodology Name

**"Cross-Referenced Accuracy Validation (CRAV)"**

A systematic approach to documentation revision that ensures every claim, code snippet, and configuration is verified against the actual codebase, with cross-referencing between documents to eliminate duplication and maintain a single source of truth (SSOT).

---

## 📋 Core Principles

1. **Single Source of Truth (SSOT)**: Doc 0 (Research) is the SSOT. All other docs reference it, never duplicate it.
2. **Codebase Verification**: Every code snippet, constant, and config field is verified against actual implementation.
3. **Zero Magic Numbers**: All hardcoded values extracted to `constants.py` with clear naming.
4. **Coexistence Over Replacement**: BiMamba2 remains stable default; GDN is experimental option via config flags.
5. **Instant Rollback**: Config-based architecture switching (no git operations needed).
6. **Iterative Approval**: Complete one doc at a time, get approval/feedback, iterate until perfect, then move to next.

---

## 🔍 Validation Steps Per Document

### Phase 1: Structural Analysis
1. **Read entire document** top-to-bottom
2. **Identify duplication** with Doc 0 or other docs
3. **Extract SSOT content** to Doc 0 if needed
4. **Map dependencies** between sections

### Phase 2: Codebase Cross-Reference
1. **Verify every import** exists in codebase
   - Example: `from src.brain_brr.constants import GDN_EDGE_NUM_HEADS_DEFAULT`
   - Check: Does `constants.py` have this constant?

2. **Verify every config field** exists in schemas
   - Example: `temporal_type_edge: "gated_deltanet"`
   - Check: Does `schemas.py` have `temporal_type_edge` field?

3. **Verify every builder call** uses correct factory pattern
   - Example: `build_edge_stream(config)` returns BiMamba2 or BiGatedDeltaNet
   - Check: Does `edge_stream.py` have conditional logic?

4. **Verify checkpoint structure** in code snippets
   - Example: `torch.load()` returns dict with `model_state_dict`
   - Check: Does `checkpoint.py` save dict with `model_state_dict` key?

5. **Verify W&B query methods** are robust
   - Example: Query by `config.experiment.name` not `run.name`
   - Check: Does this handle W&B suffix like "baseline_edge-1"?

### Phase 3: Bug Hunting
1. **Check for incorrect assumptions**
   - Example: Assuming `torch.load()` returns model, not dict

2. **Check for version mismatches**
   - Example: `transformers>=4.45.0` vs actual requirement `>=4.53.0`

3. **Check for missing prerequisites**
   - Example: Importing constants before defining them

4. **Check for mypy override collisions**
   - Example: Creating new `[[tool.mypy.overrides]]` instead of appending to existing

### Phase 4: Language Precision
1. **Replace "migration" with "validation"** (not replacing, just testing)
2. **Replace "replacement" with "coexistence"** (both architectures coexist)
3. **Add prerequisite warnings** (Phase 0 MUST be complete before Phase 1a)
4. **Add rollback procedures** (instant via config, git as last resort)

### Phase 5: Approval Loop
1. **Present summary** of changes to user
2. **Get feedback** on accuracy, completeness, clarity
3. **Iterate** based on feedback
4. **Mark as COMPLETE** only when user confirms "1000% banger"
5. **Move to next doc** only after approval

---

## 📊 Progress Tracking

### Completed Documents
- ✅ **Doc 0 (Research) v4.1**: COMPLETE AND VERIFIED
  - Fixed transformers version (4.53.0)
  - Fixed mypy override instruction (append, don't replace)
  - Added explicit import instruction for constants
  - Added Phase 0 infrastructure (Section 14)
  - Added coexistence strategy (Section 15)

- ✅ **Doc 1 (Edge Validation) v2.0**: COMPLETE AND VERIFIED
  - Removed ~600 lines of Phase 0 duplication
  - Changed "migration" → "validation" throughout
  - Added Phase 0 prerequisite verification checklist
  - Added instant config rollback
  - Fixed checkpoint loading bug (torch.load returns dict, not model)
  - Reduced from 1681 lines → 726 lines

- ✅ **Doc 2 (Node Validation) v2.1**: COMPLETE AND VERIFIED
  - **CRITICAL FIX**: Changed `temporal_type_edge: null` to explicit value (preserves Phase 1a choice)
  - **CRITICAL FIX**: Fixed attribute path `model.edge_stream.edge_mamba` → `model.edge_mamba`
  - Removed ~200 lines of Phase 0 duplication (builder implementation)
  - Changed "migration" → "validation" throughout
  - Added Phase 0 AND Phase 1a prerequisite verification
  - Added coexistence strategy (BiMamba2 default, GDN experimental)
  - Added instant config rollback (prioritized over git)
  - Fixed checkpoint loading (consistent with Doc 1)
  - Reduced from 913 lines → 827 lines

- ✅ **Doc 3 (Full Validation) v2.1**: COMPLETE AND VERIFIED
  - **CRITICAL CLARITY FIX**: Added MASSIVE warning banner (🛑 STOP section) for future state
  - **CRITICAL CLARITY FIX**: Changed status to "ROADMAP DOCUMENT" (was "Ready for Implementation")
  - Listed current reality (❌ wrapper/schema/scripts don't exist YET)
  - Added execution order diagram (Phase 0 → 1a → 1b → 2)
  - Added warnings to verification commands (will fail until Phase 0 complete)
  - Added warning to analysis script (needs to be created)
  - Changed "What Should Already Exist" → "What MUST Exist Before Phase 2"
  - Used 🔨 (to be built) and 🔬 (to be validated) emojis for clarity
  - Changed "migration" → "validation" throughout (18 instances)
  - Added Phase 0 + Phase 1a + Phase 1b prerequisite verification
  - Added coexistence strategy (BiMamba2 default, GDN experimental)
  - Added instant config rollback (prioritized over git)
  - Added validation gates (require both Phase 1a+1b success)
  - Added partial deployment option (if Phase 2 < individual phases)
  - Consistent attribute paths (model.edge_mamba, not model.edge_stream.edge_mamba)
  - Increased from 904 lines → 953 lines (added clarity warnings)

- ✅ **Doc 4 (Hybrid SWA) v2.0**: COMPLETE AND VERIFIED
  - **CRITICAL CLARITY FIX**: Added MASSIVE conditional warning banner (🛑 STOP section)
  - **CRITICAL CLARITY FIX**: Changed status to "CONDITIONAL ROADMAP" (was "Ready for Implementation")
  - **CRITICAL CONDITIONAL GATE**: Execution order diagram with DECISION GATE after Phase 2
  - Listed current reality (❌ SWA/HybridNodeStream/configs don't exist YET)
  - Emphasized Phase 3 is HIGHLY CONDITIONAL (only if <5s recall deficiency > 5%)
  - Added concrete decision examples (85% overall but 75% <5s → PROCEED, 85% vs 82% → SKIP)
  - Added 5-point prerequisite checklist (Phase 0 + 1a + 1b + 2 + manual short-event analysis)
  - Added warnings to ALL implementation sections (2, 3, 4, 5)
  - Changed all "will create" → "TO BE CREATED" with 🔨 emojis
  - Made it IMPOSSIBLE to proceed without Phase 2 results
  - Increased from 1485 lines → ~1520 lines (added conditional clarity)

### Pending Documents

**None** - All 4 validation docs (0-3) now CRAV-verified and ironclad

---

## 🔧 Tools and Commands

### Verification Commands

**Check constants exist**:
```bash
python -c "from src.brain_brr.constants import GDN_EDGE_NUM_HEADS_DEFAULT; print('✅')"
```

**Check config fields exist**:
```bash
python -c "from src.brain_brr.config.schemas import MambaConfig; cfg = MambaConfig(); assert hasattr(cfg, 'temporal_type_edge'); print('✅')"
```

**Check builder factory pattern**:
```bash
python -c "
from src.brain_brr.models.builders.edge_stream import build_edge_stream
from src.brain_brr.config.schemas import ModelConfig, MambaConfig

cfg = ModelConfig(mamba=MambaConfig(temporal_type='bimamba2'))
edge = build_edge_stream(cfg)
print(f'BiMamba2: {type(edge.edge_mamba).__name__}')

cfg.mamba.temporal_type_edge = 'gated_deltanet'
edge_gdn = build_edge_stream(cfg)
print(f'GDN: {type(edge_gdn.edge_mamba).__name__}')
"
```

**Check checkpoint structure**:
```bash
rg "torch.save" -n src/brain_brr/train -g'*.py'
# Then read checkpoint.py to verify dict structure
```

### Documentation Commands

**Search for duplicates**:
```bash
rg "BiGatedDeltaNet wrapper" FLASH_LINEAR_ATTENTION_DOC*.md
```

**Search for magic numbers**:
```bash
rg "\b(16|64|0\.75)\b" FLASH_LINEAR_ATTENTION_DOC*.md
```

**Verify references**:
```bash
rg "Doc 0 Section" FLASH_LINEAR_ATTENTION_DOC*.md
```

---

## 🚨 Common Pitfalls (Learned)

1. **Checkpoint loading**: `torch.load()` returns dict, NOT model (requires instantiation + load_state_dict)
2. **W&B name suffixes**: Query by `config.experiment.name`, not `run.name` (handles "baseline-1" suffixes)
3. **Mypy overrides**: APPEND to existing `[[tool.mypy.overrides]]`, don't create new block
4. **Import order**: Constants must be defined BEFORE being imported in schemas
5. **Version precision**: Check exact versions (e.g., transformers>=4.53.0, not 4.45.0)
6. **Duplication**: Phase 0 infrastructure only in Doc 0, not in Doc 1-4
7. **Config preservation**: Never use `null` for stream-specific overrides - use explicit values to preserve prior phase results
8. **Attribute paths**: Verify against actual detector code (e.g., `model.edge_mamba` not `model.edge_stream.edge_mamba`)
9. **Future vs Current Documentation**: MUST distinguish "roadmap" (what will exist after Phase 0) from "instructions" (what to do now). Add MASSIVE warnings if doc describes future state that doesn't exist yet. Use status badges like "🚧 ROADMAP DOCUMENT" and explicit "❌ Does NOT exist yet" lists.

---

## 📍 Current Status (October 8, 2025)

**Completed**:
- ✅ Doc 0 v4.1 (SSOT with Phase 0 infrastructure + coexistence strategy)
- ✅ Doc 1 v2.0 (Edge validation with prerequisite checks + rollback)
- ✅ Doc 2 v2.1 (Node validation with critical fixes for config preservation + attribute path)
- ✅ Doc 3 v2.1 (Full validation with validation gates + MASSIVE future-state clarity warnings)
- ✅ Doc 4 v2.0 (Hybrid SWA with CONDITIONAL ROADMAP + decision gate after Phase 2)

**Next Steps**:
1. **Await user decision**: Start Phase 0 implementation (Doc 0 Section 14) OR take different direction
2. **All documentation complete**: Docs 0-4 are now CRAV-verified, ironclad, and ready for phased execution
3. **Execution order**: Phase 0 (4-6d) → Phase 1a (2-3d) → Phase 1b (2-3d) → Phase 2 (2-3d) → [conditional] → Phase 3 (2-3d)

---

## 🎓 Methodology Benefits

**Before CRAV**:
- ❌ 600+ lines of duplication across docs
- ❌ Magic numbers hardcoded in examples
- ❌ Assumptions about checkpoint structure (wrong)
- ❌ W&B queries that break with name suffixes
- ❌ Language implying "replacement" not "coexistence"

**After CRAV**:
- ✅ Zero duplication (SSOT in Doc 0)
- ✅ All constants extracted and imported
- ✅ Checkpoint loading verified against codebase
- ✅ Robust W&B queries by config.experiment.name
- ✅ Clear coexistence strategy throughout
- ✅ 1000% accuracy with codebase

---

## 📚 References

- **Doc 0 (SSOT)**: [FLASH_LINEAR_ATTENTION_RESEARCH.md](FLASH_LINEAR_ATTENTION_RESEARCH.md)
- **Doc 1 (Edge)**: [FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC1_EDGE_MIGRATION.md)
- **Doc 2 (Node)**: [FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC2_NODE_MIGRATION.md)
- **Doc 3 (Full)**: [FLASH_LINEAR_ATTENTION_DOC3_FULL_MIGRATION.md](FLASH_LINEAR_ATTENTION_DOC3_FULL_MIGRATION.md)
- **Doc 4 (Hybrid)**: [FLASH_LINEAR_ATTENTION_DOC4_HYBRID_SWA.md](FLASH_LINEAR_ATTENTION_DOC4_HYBRID_SWA.md)

---

**Document Status**: ✅ Methodology Defined
**Next Action**: Resume with Doc 2 after pivot task complete
**Estimated Time**: ~3-4 hours per doc (2 docs/day pace)
