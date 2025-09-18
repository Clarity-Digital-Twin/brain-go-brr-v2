# Documentation Structure

## 📚 Documentation Organization

### Core Documentation (Root)
- `AGENTS.md` - AI assistant operating rules (single source of truth)
- `CLAUDE.md` - Claude-specific configuration
- `README.md` - Project overview and quickstart
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
- `NOVEL_ARCHITECTURE.md` - Bi-Mamba-2 innovation details
- `ARCHITECTURE_CLARIFICATION.md` - Technical architecture specs
- `ASCII_PIPELINE_PLAN.md` - Visual pipeline representation
- `FINAL_STACK_ANALYSIS.md` - Complete stack analysis

### `/docs/implementation/` - Implementation Details
- `IMPLEMENTATION_PHASES.md` - Development roadmap
- `PREPROCESSING_STRATEGY.md` - Data preprocessing strategy
- `EVALUATION_CHECKLIST.md` - Testing and validation checklist
- `REPO_RESTRUCTURE_PLAN.md` - Repository organization
- `SETUP_NOTES.md` - Environment setup instructions

### `/docs/references/` - External References
- `REFERENCE_REPOS.md` - Related projects and papers
- `TUSZ_EDF_HEADER_FIX.md` - TUH dataset header fixes
- `TUSZ_HEADER_FIX_INTEGRATION.md` - Header fix implementation

### `/docs/archive/` - Historical Documentation
Contains older documentation versions for reference.

## 🚀 Quick Navigation

- **Getting Started**: See root `README.md`
- **Development Rules**: See `AGENTS.md` and `CLAUDE.md`
- **Architecture Details**: `/docs/architecture/`
- **Implementation Status**: `/docs/phases/`
- **Known Issues**: `/docs/KNOWN_ISSUES.md`