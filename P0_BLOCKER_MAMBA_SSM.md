# P0 BLOCKER: Mamba-SSM Not Loading - Training with FALLBACK

## CRITICAL ISSUE
**Local training is using Conv1d fallback instead of real Mamba-SSM!**

## Evidence
```
[MAMBA] Mamba-SSM not available, using fallback
[MAMBA] Mamba-SSM not available, using fallback
[MAMBA] Mamba-SSM not available, using fallback
```

## Impact
- **Not using real Bi-Mamba-2** - the whole point of this architecture!
- **Using Conv1d fallback** - just for shape validation
- **Training a crippled model** - Conv1d can't do selective scan
- **Wasting compute** - training wrong architecture

## Root Cause
The mamba-ssm package isn't installed or isn't detected properly.

From `src/brain_brr/models/mamba.py`:
```python
try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    warnings.warn("mamba-ssm not available; using Conv1d fallback")
```

## Why This Happened
1. `uv sync` doesn't install GPU extras by default
2. Need to run `uv sync -E gpu` to get mamba-ssm
3. mamba-ssm requires CUDA and proper compilation

## Fix Required
```bash
# Stop current training immediately
tmux kill-session -t train_full

# Install GPU extras including mamba-ssm
uv sync -E gpu

# Verify mamba-ssm is available
python -c "from mamba_ssm import Mamba2; print('Mamba-SSM OK')"

# Restart training with REAL Mamba
tmux new -d -s train_full "python -m src train configs/local/train.yaml"
```

## Verification
After fix, logs should NOT show:
- ❌ `[MAMBA] Mamba-SSM not available, using fallback`

Instead should show proper Mamba2 initialization.

## Modal Impact
Modal might ALSO have this issue if the Docker image doesn't include mamba-ssm!
Need to verify Modal's `app.py` includes:
```python
.pip_install("mamba-ssm", pre=True)  # or similar
```

## Severity: CRITICAL
- **Local training**: BROKEN (using fallback)
- **Modal training**: UNKNOWN (need to verify)
- **All training**: Potentially training wrong architecture!

This is worse than the manifest bug - we're not even training the right model!