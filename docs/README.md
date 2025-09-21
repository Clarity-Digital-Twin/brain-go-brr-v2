# Documentation Structure

Last updated: 2025-09-19

## 📚 Documentation Organization

### Core Documentation (Root)
- `AGENTS.md` - AI assistant operating rules (single source of truth)
- `CLAUDE.md` - Claude-specific configuration
- `README.md` - Project overview and quickstart
- `DOCS_SSOT.md` - Canonical commands/configs and current deployment/CLI pointers
- `LICENSE` - Apache 2.0 License

### `/docs/phases/` - Implementation Phases
- `PHASE1_DATA_PIPELINE.md` - Data loading and preprocessing
- `PHASE2_MODEL_ARCHITECTURE.md` - Overall model design
- `PHASE2.1_UNET_ENCODER.md` - U-Net encoder details
- `PHASE2.2_RESCNN_STACK.md` - ResCNN implementation
- `PHASE2.3_BIMAMBA.md` - Bidirectional Mamba-2
- `PHASE2.4_DECODER.md` - Decoder architecture
- `PHASE2.5_FULL_MODEL.md` - Complete model integration
- `PHASE3_TRAINING_PIPELINE.md` - Training implementation
- `PHASE4_POSTPROCESSING.md` - Post-processing pipeline
- `PHASE5_EVALUATION.md` - Evaluation metrics and scoring

### `/docs/architecture/` - Architecture Documentation
- `CANONICAL_ARCHITECTURE_SPEC.md` - Complete technical specification
- `ASCII_PIPELINE_PLAN.md` - Visual pipeline representation
- `ARCHITECTURE_COMPARISON.md` - Literature/context vs our stack
- `FINAL_STACK_ANALYSIS.md` - Full stack analysis
- `MAMBA_KERNEL_DECISION.md` - CUDA kernel decision notes

### `/docs/implementation/` - Implementation Details
- `IMPLEMENTATION_PHASES.md` - Development roadmap
- `PREPROCESSING_STRATEGY.md` - Data preprocessing strategy
- `EVALUATION_CHECKLIST.md` - Testing and validation checklist
- `SETUP_NOTES.md` - Environment setup instructions
- `benchmarks.md` - Literature benchmarks

### `/docs/deployment/` - Deployment
- `MODAL_DEPLOYMENT_SSOT.md` - Modal deployment Single Source of Truth
- `MODAL_PIPELINE_SETUP.md` - How the Modal app is configured
- `PREFLIGHT_STRATEGY.md` - Verify parser/balancing before long runs
- `MODAL_MAMBA_DEPLOYMENT_ISSUES.md` - CUDA/Mamba deployment notes
- `MODAL_LOGGING_TODO.md` - Logging/observability TODOs

### `/docs/testing/` - Testing Strategy
- `TEST_PLAN.md` - Canonical testing strategy and commands

### `/docs/references/` - External References
- `REFERENCE_REPOS.md` - Related projects and papers
- `TUSZ_CSV_BI_PARSER.md` - Correct CSV_BI parsing + labels
- `TUSZ_SAMPLING_STRATEGY.md` - SeizureTransformer-style balancing (manifest + dataset)
- `TUSZ_CHANNELS.md` - Canonical 19‑ch order + synonyms
- `TUSZ_EDF_HEADER_FIX.md` - TUH dataset header fixes
- `TUSZ_HEADER_FIX_INTEGRATION.md` - Header fix integration details

### `/docs/archive/` - Historical Documentation
Contains older documentation versions for reference.

## 🚀 Quick Navigation

- **Getting Started**: See root `README.md`
- **Development Rules**: See `AGENTS.md` and `CLAUDE.md`
- **Architecture Details**: `/docs/architecture/`
- **Implementation Status**: `/docs/phases/`
- **Known Issues**: `/docs/KNOWN_ISSUES.md`
- **History/Archive Index**: `/docs/HISTORY.md`
