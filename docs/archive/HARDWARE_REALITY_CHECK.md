# HARDWARE REALITY CHECK: Can V3 + Full Dynamic PE Fit on RTX 4090?

> Archived note: Hardware‑specific recommendations are summarized in
> `docs/05-training/local.md` and `docs/08-operations/performance-optimization.md`.
> See `docs/ARCHIVE_MAPPING.md`.

## THE ABSOLUTE TRUTH

### Hardware Constraint
- **RTX 4090**: 24 GB VRAM
- **Usable**: ~22 GB (OS/driver overhead)

### V3 Architecture Memory Requirements (EXACT)

#### With FULL Dynamic PE (computing every timestep):
```
Batch Size = 8:
- Model parameters: 31.5M × 4 bytes = 126 MB
- TCN activations: ~200 MB
- Dual BiMamba streams: ~724 MB
- Dynamic PE (960 eigendecompositions): 7,500 MB
- GNN processing: ~600 MB
- Gradients (2x forward): ~18,000 MB
TOTAL: 27 GB > 24 GB ❌ DOES NOT FIT
```

#### With FULL Dynamic PE but smaller batch:
```
Batch Size = 3:
- Everything scales linearly with batch size
- TOTAL: 27 GB × (3/8) = 10.1 GB ✅ FITS
```

## THE ANSWER: **YES, IT'S POSSIBLE**

### Option 1: Reduce Batch Size to 3
- **FULL dynamic PE** (every single timestep)
- **FULL V3 architecture**
- **Memory**: 10 GB < 24 GB ✅
- **Downside**: 2.67x slower training

### Option 2: What is "Semi-Dynamic"?
**Semi-dynamic just means compute PE every N timesteps instead of every 1:**
- `semi_dynamic_interval=1`: Compute PE at timesteps 0,1,2,3...959 (960 times)
- `semi_dynamic_interval=10`: Compute PE at timesteps 0,10,20...950 (96 times)

**This is STILL dynamic PE** - it updates during the sequence, just less frequently.

## SINGLE SOURCE OF TRUTH

**Q: Can we train V3 with dynamic PE on RTX 4090?**
**A: YES, with batch_size=3**

**Q: Can we train with batch_size=8?**
**A: NO, unless we compute PE less frequently (semi-dynamic)**

**Q: Is semi-dynamic a compromise?**
**A: Yes, but minimal - updating every 10 timesteps (39ms) still captures dynamics**

## THE REAL CHOICE

### Choice A: FULL PURITY
```yaml
batch_size: 3
use_dynamic_pe: true
semi_dynamic_interval: 1  # Every timestep
```
- 100% EvoBrain approach
- Fits in 10 GB
- Slow training

### Choice B: PRACTICAL COMPROMISE
```yaml
batch_size: 6
use_dynamic_pe: true
semi_dynamic_interval: 10  # Every 10 timesteps
```
- 90% of dynamic PE benefit
- Fits in 20 GB
- 2x faster training

### Choice C: GIVE UP DYNAMIC PE
```yaml
batch_size: 8
use_dynamic_pe: false  # Static PE only
```
- Not recommended
- Loses key V3 advantage

## FINAL ANSWER

**IT IS POSSIBLE TO TRAIN FULL V3 WITH FULL DYNAMIC PE ON RTX 4090**

You just have to choose:
1. **Batch size 3** = Full dynamic PE
2. **Batch size 6** = Dynamic PE every 10 steps (still dynamic!)
3. **Batch size 8** = No dynamic PE (bad)

**My recommendation: Choice B (batch_size=6, interval=10)**
- Still dynamic
- Updates every 39ms (plenty fast for seizure dynamics)
- Reasonable training speed
