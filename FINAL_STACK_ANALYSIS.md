# CRITICAL STACK ANALYSIS: Which Architecture for Seizure Detection?

## THE CONTENDERS

### 1. MAYBE_STACK (U-Net + Bi-Mamba + Hysteresis)
**Strengths:**
- **LINEAR TIME COMPLEXITY** - O(N) vs O(N²) of transformers
- **Hysteresis decoding** - mimics clinical decision making (commit later, release later)
- **Proven U-Net for morphology** - captures spikes, rhythmic patterns
- **Bi-Mamba for long context** - handles minute-scale patterns without memory explosion
- **20-30M params** - deployable on edge devices
- **Direct TAES alignment** - per-sample predictions at 256Hz

**Weaknesses:**
- Mamba is relatively new (less proven in production)
- Limited interpretability compared to attention mechanisms
- May miss very subtle cross-channel interactions

### 2. SEIZURE_ML_STACK_DESIGN (My Hybrid Proposal)
**Strengths:**
- Comprehensive multi-scale processing
- Ensemble robustness
- Patient-specific adaptation capability
- Extensive feature engineering pipeline

**Weaknesses:**
- **OVERCOMPLICATED** - too many components
- Would take months to implement properly
- High computational cost
- Risk of overfitting with so many modules

### 3. PYSEIZURE (Classical Ensemble)
**Strengths:**
- **PROVEN CROSS-DATASET GENERALIZATION** (CHB-MIT ↔ TUSZ)
- Multiple classifiers voting = robustness
- SHAP explainability
- Relatively simple to implement
- Good baseline performance (AUC 0.904)

**Weaknesses:**
- **Window-based** not time-step level (bad for TAES)
- Requires extensive feature engineering
- Lower ceiling on performance
- Not end-to-end differentiable

### 4. SeizureTransformer (Competition Winner)
**Strengths:**
- **WON 2025 CHALLENGE** - proven in blind evaluation
- Time-step level predictions (TAES aligned!)
- Fast inference (3.98s per hour)
- U-Net + Transformer proven combo
- Published code available

**Weaknesses:**
- **QUADRATIC COMPLEXITY** - O(N²) for attention
- Memory hungry for long sequences
- May struggle with very long recordings
- Requires significant GPU resources

### 5. BI-MAMBA/FEMBA (Foundation Model)
**Strengths:**
- **MASSIVE PRE-TRAINING** (21,000 hours of EEG!)
- Linear complexity like MAYBE_STACK
- Tiny variant (7.8M) works on edge
- Strong results (0.918 AUROC on TUAR)
- Bidirectional processing

**Weaknesses:**
- Requires massive pre-training dataset
- Complex self-supervised setup
- Not specifically optimized for seizure detection
- May be overkill for focused task

---

## THE HARD TRUTH

Looking at the evidence:

1. **SeizureTransformer ACTUALLY WON** a real competition with blind data
2. **PYSEIZURE showed REAL cross-dataset transfer** (critical for clinical use)
3. **BI-MAMBA/FEMBA proved linear scaling WORKS** at foundation scale
4. **MAYBE_STACK combines the BEST IDEAS** in a focused way

## MY VERDICT: GO WITH MAYBE_STACK (Modified)

### Why?

1. **It's RIGHT-SIZED** - Not overengineered like my proposal, not underengineered like classical ML

2. **MAMBA IS THE FUTURE** - Linear complexity is non-negotiable for continuous monitoring. Transformers will hit a wall with 24-hour recordings.

3. **HYSTERESIS IS GENIUS** - This single trick probably accounts for much of real-world performance

4. **U-Net ALWAYS WORKS** - Every winning architecture has U-Net bones (SeizureTransformer, EventNet, etc.)

5. **FOCUSED ON WHAT MATTERS** - TAES scoring and FA/24h, not chasing AUROC

### BUT with these modifications from the literature:

```python
class OptimalStack:
    """The real winner based on all evidence"""

    def __init__(self):
        # Core from MAYBE_STACK
        self.encoder = UNet1D(skip_connections=True)
        self.bottleneck = BiMamba2(
            layers=6,  # As in MAYBE_STACK
            d_model=512,
            bidirectional=True
        )
        self.decoder = UNet1DDecoder()

        # Key addition from SeizureTransformer
        self.res_cnn = ResidualCNNStack(
            kernels=[3, 5, 7],  # Multi-scale local patterns
            before_bottleneck=True
        )

        # Key addition from PYSEIZURE
        self.auxiliary_head = XGBoostHead(
            features=['power_spectral', 'zero_crossings'],
            weight=0.2  # Light ensemble for robustness
        )

        # Keep MAYBE_STACK's hysteresis exactly
        self.eventizer = HysteresisEventizer(
            tau_on=0.86,
            tau_off=0.78,
            min_duration=3.0
        )
```

### Implementation Priority:

1. **Week 1**: Get MAYBE_STACK core working exactly as specified
2. **Week 2**: Add SeizureTransformer's ResCNN for local patterns
3. **Week 3**: Add light XGBoost head for robustness
4. **Week 4**: Validate on CHB-MIT → TUSZ transfer

---

## THE BIOLOGICAL ALIGNMENT

MAYBE_STACK actually maps to seizure biology better than transformers:

1. **U-Net encoder** = Captures fast ripples and spikes (local)
2. **Bi-Mamba** = Tracks seizure state evolution (global)
3. **Hysteresis** = Models seizure threshold dynamics
4. **Per-timestep** = Matches continuous neural activity

Transformers treat everything as "attention" which isn't how seizures work. Seizures are **state transitions** with **memory** - exactly what SSMs model!

---

## FINAL RECOMMENDATION

**USE MAYBE_STACK AS THE FOUNDATION**

It's:
- Scientifically sound (SSM matches neural dynamics)
- Computationally efficient (linear scaling)
- Clinically aligned (TAES + hysteresis)
- Right-sized (20-30M params)
- Implementable in 4 weeks

The only risk is Mamba being newer, but FEMBA's results prove it works at scale.

Don't overthink it. The community is converging on:
**U-Net + Efficient Sequence Model + Smart Decoding**

MAYBE_STACK nails all three. Ship it.

---

## Appendix: Why Not the Others?

- **My proposal**: Academic masturbation. Would never finish implementing.
- **Pure PYSEIZURE**: Window-based is dead for TAES. Full stop.
- **Pure SeizureTransformer**: O(N²) is dead for 24-hour EEG. Full stop.
- **Pure FEMBA**: Foundation model for focused task = using a sledgehammer on a nail.

The winner is the one that **respects the constraints**:
1. Must handle long sequences (Mamba ✓)
2. Must preserve local morphology (U-Net ✓)
3. Must output per-timestep (✓)
4. Must be implementable (✓)
5. Must reduce false alarms (Hysteresis ✓)

MAYBE_STACK hits all five. Case closed.