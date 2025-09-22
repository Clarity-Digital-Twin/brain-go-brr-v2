# Data Pipeline Flow Diagram

```mermaid
flowchart TB
    %% Stage 1: Raw Input
    EDF[EDF Files<br/>19-channel EEG<br/>50-500MB each]
    CSV[CSV Files<br/>TUSZ Annotations<br/>start/stop/label]

    %% Stage 2: Preprocessing
    EDF --> LOAD[Load EDF<br/>Parse Channels]
    CSV --> PARSE[Parse CSV_BI<br/>Extract Events]

    LOAD --> FILT[Preprocessing<br/>• Bandpass 0.5-120Hz<br/>• Notch 60Hz<br/>• Resample 256Hz]
    PARSE --> MASK[Binary Mask<br/>15360 samples/window]

    FILT --> WIN[Window Extraction<br/>60s windows<br/>10s stride]
    MASK --> WIN

    WIN --> NPZ[NPZ Cache Files<br/>windows: N×19×15360<br/>labels: N×15360<br/>26-152MB each]

    %% Stage 3: Manifest Generation
    NPZ --> SCAN{First Run?}
    SCAN -->|Yes| BUILD[Scan ALL NPZ Files<br/>Categorize Windows]
    SCAN -->|No| LOADM[Load manifest.json]

    BUILD --> CAT[Categorize Each Window<br/>• >50% seizure → full<br/>• 1-50% seizure → partial<br/>• 0% seizure → background]

    CAT --> MAN[manifest.json<br/>Lists exact window locations<br/>by seizure type]

    %% Stage 4: Dataset Creation
    MAN --> BAL[BalancedSeizureDataset<br/>• ALL partial windows<br/>• 0.3× full windows<br/>• 2.5× background]
    LOADM --> BAL

    BAL --> RATIO[Calculate Exact Ratio<br/>seizure_ratio = 0.342<br/>Known instantly from manifest]

    %% Stage 5: Training
    RATIO --> TRAIN{Training Mode}

    TRAIN -->|OLD: Sampling| SAMP[Sample 1000 Windows<br/>Load NPZ files randomly<br/>2+ HOURS on Modal!]
    TRAIN -->|NEW: Direct| DIRECT[Use Known Ratio<br/>No I/O needed<br/>INSTANT!]

    SAMP --> EST[Estimate ratio ≈ 0.342<br/>±sampling error]
    DIRECT --> EXACT[Exact ratio = 0.342<br/>No error]

    EST --> POS[Calculate pos_weight<br/>sqrt((1-ratio)/ratio)<br/>= 1.387]
    EXACT --> POS

    POS --> LOSS[BCE Loss with pos_weight<br/>Train Model]

    %% Styling
    style EDF fill:#e1f5fe
    style CSV fill:#e1f5fe
    style NPZ fill:#fff3e0
    style MAN fill:#f3e5f5
    style BAL fill:#e8f5e9
    style DIRECT fill:#c8e6c9,stroke:#4caf50,stroke-width:3px
    style SAMP fill:#ffcdd2,stroke:#f44336,stroke-width:2px
    style RATIO fill:#c8e6c9
```

## Key Optimizations Visualized

### 🔴 OLD PATH (Slow):
```
BalancedSeizureDataset → Sample 1000 windows → Load NPZ files → Estimate ratio
                                    ↓
                            2+ HOURS on Modal!
                          (Network I/O bottleneck)
```

### 🟢 NEW PATH (Fast):
```
BalancedSeizureDataset → Use seizure_ratio property → Direct value
                                    ↓
                                 INSTANT!
                            (No I/O required)
```

## Data Flow Summary

1. **Input Layer**: Raw EDF + CSV annotations
2. **Processing Layer**: Filter, resample, window → NPZ cache
3. **Index Layer**: Manifest tracks exact window locations
4. **Balance Layer**: Strategic sampling (partial/full/background)
5. **Training Layer**: Direct ratio access eliminates bottleneck

## File Sizes & Performance

| Stage | File Type | Size | Access Time (Local) | Access Time (Modal) |
|-------|-----------|------|---------------------|---------------------|
| Input | EDF | 50-500MB | 100ms | 1-5s |
| Cache | NPZ | 26-152MB | <1ms (cached) | 100-700ms |
| Index | manifest.json | <1MB | <1ms | <10ms |
| Sample 1000 | - | - | 1-2s | 12min - 2hrs |
| Direct Ratio | - | - | 0ms | 0ms |

## Why This Works

**Mathematical Identity**:
- Sampling 1000 windows: `E[ratio] = true_ratio ± error`
- Using manifest: `ratio = true_ratio` (exact)
- Both produce same `pos_weight = 1.387`

**Engineering Win**:
- Eliminate redundant I/O
- Use pre-computed knowledge
- Maintain exact same training behavior
- 2+ hours → instant on Modal!