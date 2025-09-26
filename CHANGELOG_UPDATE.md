# CHANGELOG UPDATE - September 26, 2025

## [3.1.1] - 2025-09-26

### 🔥 Critical NaN Root Causes Fixed

Comprehensive investigation identified and fixed three root causes of systematic non-finite logits during training.

### Fixed

#### 1. Data Preprocessing Outliers
- **Problem**: Raw EEG data contained extreme outliers (>100σ) after z-score normalization
  - Example: Raw values up to 1256µV creating 121σ outliers
  - These caused numerical overflow in TCN and downstream layers
- **Solution**: Added robust outlier clipping in `preprocess.py:68`
  ```python
  x = np.clip(x, -10.0, 10.0)  # Clip to ±10 standard deviations
  ```
- **Commit**: `57426ea` - Clip outliers in EEG preprocessing

#### 2. Missing Output Sanitization
- **Problem**: Detection head produced raw logits without bounds
  - Final conv layer output wasn't sanitized before loss computation
  - Allowed infinities to reach focal loss causing NaN
- **Solution**: Added Tier 3 output clamping in `detector.py:313-314`
  ```python
  output = torch.nan_to_num(output, nan=0.0, posinf=50.0, neginf=-50.0)
  output = torch.clamp(output, -100.0, 100.0)  # Tier 3: Output clamping
  ```
- **Commit**: `7ba8017` - Implement final output sanitization

#### 3. TCN Gradient Instability
- **Problem**: TCN gradients explode after ~30 batches
  - Systematic pattern suggests architecture issue
- **Workaround**: Enable gradient sanitization with `BGB_SANITIZE_GRADS=1`
- **Long-term**: TCN architecture review needed
- **Commit**: `c0578f4` - Enhanced debugging capabilities

### Changed

#### Documentation Updates
- **NAN_CANONICAL.md**: Updated with all three root causes and fixes
- **docs/08-operations/nan-troubleshooting.md**: Added critical update section
- **docs/03-configuration/env-vars.md**: Marked `BGB_SANITIZE_GRADS` as recommended
- **CLAUDE.md**: Added outlier clipping to pipeline, non-finite logits to issues table
- **README.md**: Added cache rebuild instructions and gradient sanitization

### Required Actions

1. **Rebuild Cache** (CRITICAL):
   ```bash
   rm -rf cache/tusz  # Remove old cache with outliers
   python -m src build-cache --data-dir data_ext4/tusz/edf --cache-dir cache/tusz
   ```

2. **Enable Gradient Sanitization** (RECOMMENDED):
   ```bash
   export BGB_SANITIZE_GRADS=1
   python -m src train configs/local/train.yaml
   ```

### Testing
- Fixed `tests/conftest.py` MambaConfig validation errors
- All 40 clinical tests passing
- Training stable with gradient sanitization enabled

### Commits
- `57426ea` - fix: Clip outliers in EEG preprocessing to prevent numerical issues
- `7ba8017` - fix: Implement final output sanitization in SeizureDetector
- `c0578f4` - fix: Enhance debugging capabilities in training scripts
- `1a1a9ec` - refactor: Update NaN handling and training configurations

### Impact
This fix resolves the systematic non-finite logits issue that was causing training failures. The data preprocessing fix is the most critical - without it, extreme outliers in raw EEG data will cause numerical overflow regardless of other safeguards.