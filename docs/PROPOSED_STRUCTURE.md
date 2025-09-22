# Proposed Documentation Structure

## Current → New Organization

```
docs/
├── 01-data/                    # All data-related docs
│   ├── pipeline/               # Core pipeline docs
│   │   ├── architecture.md    # Main consolidated pipeline doc
│   │   ├── flow-diagram.md    # Mermaid visualization
│   │   └── cache-rebuild.md   # Quick rebuild guide
│   ├── tusz/                  # TUSZ-specific docs
│   │   ├── overview.md
│   │   ├── channels.md
│   │   ├── csv-parser.md
│   │   ├── edf-repair.md
│   │   ├── data-flow.md
│   │   ├── cache-sampling.md
│   │   └── preflight.md
│   ├── issues/                 # Historical issues
│   │   ├── critical-resolved.md
│   │   └── data-io.md
│   └── archive/               # Old redundant docs
│
├── 02-model/                   # Model architecture docs
│   ├── architecture/          # Core architecture
│   │   ├── canonical-spec.md # Official spec
│   │   ├── full-model.md
│   │   └── pipeline-diagram.md
│   ├── components/            # Individual components
│   │   ├── unet.md
│   │   ├── rescnn.md
│   │   ├── mamba.md
│   │   └── decoder.md
│   ├── deployment/            # Deployment-specific
│   │   ├── architecture.md   # CUDA/Modal specifics
│   │   └── mamba-kernels.md  # d_conv decisions
│   └── analysis/              # Analysis & audits
│       ├── comparison.md
│       ├── stack-analysis.md
│       ├── audit-summary.md
│       └── spec-audit.md
│
├── 03-deployment/              # All deployment docs
│   ├── modal/                 # Modal.com specific
│   │   ├── deploy.md
│   │   └── preflight.md
│   ├── local/                 # Local deployment
│   │   ├── wsl2.md
│   │   └── setup.md
│   ├── operations/            # Training & eval
│   │   ├── training.md
│   │   ├── evaluation.md
│   │   └── postprocessing.md
│   └── troubleshooting.md     # All troubleshooting
│
├── 04-research/                # Future work & experiments
│   ├── future/                # Future directions
│   │   ├── direction.md
│   │   ├── roadmap.md
│   │   └── gnn-tcn-stack.md
│   ├── benchmarks/            # Benchmarking
│   │   ├── plans.md
│   │   └── results.md
│   └── experiments/           # Experimental features
│       ├── channel-annotations.md
│       └── streaming.md
│
├── README.md                   # Main docs index
└── HISTORY.md                  # Project history
```

## Benefits of New Structure

1. **Clearer separation**: Data vs Model vs Deployment vs Research
2. **Logical grouping**: Related docs in subdirectories
3. **Easier navigation**: Clear hierarchy
4. **Scalable**: Room for growth in each category
5. **No lost docs**: Everything preserved, just reorganized

## Migration Plan

1. Create new directory structure
2. Move files to appropriate locations
3. Update any internal doc references
4. Create index files for each major section