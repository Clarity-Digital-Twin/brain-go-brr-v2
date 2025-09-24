# Frequency Analysis Experiment Plan

## Background
EvoBrain uses STFT for frequency features, showing improved performance in ablations. Our V3 uses TCN with multi-scale kernels as an implicit frequency decomposition.

## Hypothesis
Adding explicit frequency features alongside TCN could improve seizure detection by capturing both learned (TCN) and fixed (STFT) frequency patterns.

## Proposed Implementation

### Option 1: Parallel Frequency Branch (Recommended)
```python
class FrequencyBranch(nn.Module):
    """Parallel STFT branch to complement TCN features."""

    def __init__(self, n_fft=512, hop_length=128, n_freqs=64):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Process frequency features
        self.freq_conv = nn.Sequential(
            nn.Conv2d(19, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )

        # Project to match TCN output dims
        self.proj = nn.Linear(64 * n_freqs, 512)

    def forward(self, x):
        # x: (B, 19, 15360) raw EEG
        B, C, T = x.shape

        # Compute STFT per channel
        stft_out = []
        for i in range(C):
            stft = torch.stft(
                x[:, i, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True
            )
            # Log amplitude of positive frequencies
            mag = torch.abs(stft[:, :self.n_fft//2, :])
            log_mag = torch.log(mag + 1e-8)
            stft_out.append(log_mag)

        # Stack channels: (B, C, F, T')
        stft_tensor = torch.stack(stft_out, dim=1)

        # Process with 2D convolutions
        freq_features = self.freq_conv(stft_tensor)

        # Pool over frequency dimension
        freq_pooled = freq_features.mean(dim=2)  # (B, 64, T')

        # Interpolate to match TCN temporal resolution
        freq_interp = F.interpolate(
            freq_pooled,
            size=960,  # Match TCN output
            mode='linear'
        )

        return freq_interp


class HybridEncoder(nn.Module):
    """Combine TCN and frequency features."""

    def __init__(self, tcn_config, use_frequency=True):
        super().__init__()
        self.tcn = TCNEncoder(**tcn_config)
        self.use_frequency = use_frequency

        if use_frequency:
            self.freq_branch = FrequencyBranch()
            # Combine features
            self.fusion = nn.Sequential(
                nn.Conv1d(512 + 64, 512, 1),
                nn.BatchNorm1d(512),
                nn.ELU()
            )

    def forward(self, x):
        # TCN features
        tcn_out = self.tcn(x)  # (B, 512, 960)

        if self.use_frequency:
            # Frequency features
            freq_out = self.freq_branch(x)  # (B, 64, 960)

            # Concatenate and fuse
            combined = torch.cat([tcn_out, freq_out], dim=1)
            output = self.fusion(combined)
        else:
            output = tcn_out

        return output
```

### Option 2: Replace TCN Input (Not Recommended)
```python
# Convert raw EEG to frequency domain before TCN
# Less flexible, loses phase information
```

### Option 3: Hybrid Features at Each TCN Layer
```python
# Add frequency skip connections at multiple scales
# More complex, harder to train
```

## Configuration

```yaml
model:
  encoder:
    use_frequency_branch: false  # Start disabled for baseline
    frequency:
      n_fft: 512
      hop_length: 128
      n_freq_bins: 64
```

## Evaluation Plan

1. **Baseline**: Current V3 with TCN only
2. **Frequency-Enhanced**: V3 + FrequencyBranch
3. **Metrics**:
   - AUROC improvement
   - Sensitivity at fixed FA rates
   - Computational overhead

## Expected Outcomes

- **Best Case**: +3-5% AUROC improvement (based on EvoBrain ablation)
- **Likely Case**: +1-2% AUROC improvement with 10-15% compute overhead
- **Worst Case**: No improvement, validates TCN as sufficient

## Priority: MEDIUM 🟡

While EvoBrain shows frequency helps, our TCN approach is defensible:
- TCN multi-scale kernels act as learned frequency filters
- More end-to-end than fixed STFT
- Avoids hyperparameter tuning for STFT

This is a good experiment but not critical like dynamic PE was.