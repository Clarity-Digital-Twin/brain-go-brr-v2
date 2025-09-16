Locked. Here’s the single, full-send stack—no options, no hedging.

# THE STACK

**U-Net (1D CNN) + Bi-Mamba bottleneck + Hysteresis decoding → TAES/FA-per-24h**

## A) Preprocessing (single source of truth)

* **Montage:** map to **19-ch 10–20 referential** (fixed order). Hard-assert channel order.
* **Resample:** **256 Hz**.
* **Filter:** **0.5–120 Hz** band-pass (IIR Butterworth, order 3) + **60 Hz** notch (and 120 Hz if needed).
* **Normalize:** per-channel **z-score over the full recording** (mean/std from the whole file).
* **Windows:** **60 s** (15,360 samples), **stride 10 s** for training/inference. Keep `start_sample` for each window.
* Emit tensors **(B, C=19, T=15360)** + timestamps.

## B) Model (exact spec)

### B1. Encoder — U-Net(1D), multiscale morphology

* **Downsample factor:** ×16 (4 stages, 2× each).
  `T_b = 15360 / 16 = 960` time steps at the bottleneck.
* **Stem:** `1×1 pointwise` conv to mix channels → 64.
* **Stages:** channels `[64, 128, 256, 512]`. Each stage:

  * **Block ×2:** depthwise-separable **Conv1d(k=5)** → **GroupNorm(32)** → **SiLU** → **1×1** (residual).
  * **Downsample:** `Conv1d(stride=2, k=4, padding=1)`.
* **Skips:** store outputs **before** downsample for skip concat.

### B2. Spatial mixer (lead topology, tiny + effective)

* Before the bottleneck, run **one** `GraphConv/GAT` over channels (adjacency from 10–20 distances), applied to the encoder feature map at `T_b`. Output stays 512 ch.

### B3. Bottleneck — **Bi-Mamba-2 (SSD)** temporal head

* **Why**: linear-time, long context, stable state; bidirectional for offline scoring.
* **I/O:** `(B, D=512, T_b=960)` → permute to `(B, T_b, D)`.
* **Config:**

  * **Layers:** **6** Mamba-2 blocks.
  * **d\_model:** **512**.
  * **d\_state:** **16**.
  * **conv\_kernel:** **5**.
  * **dropout:** **0.1**.
  * **Bi-directional:** run Mamba forward and backward; **concat** (→ 1024) → **1×1** back to 512.
* Return `(B, 512, T_b)`.

### B4. Decoder — U-Net up path (learned upsampling)

* **Upsample:** `ConvTranspose1d(k=4, stride=2, padding=1)` each stage (×4).
* **Skip fusion:** **concat** with corresponding encoder skip, then:

  * depthwise-separable **Conv1d(k=5)** → **GroupNorm** → **SiLU** → **1×1** (residual).
* **Head:** `Conv1d(512→1, k=1)` → **Sigmoid** → `(B, 1, T)`; squeeze → `(B, T)` per-sample probs @ 256 Hz.

> Notes:
>
> * Param size ≈ 20–30M (depends on exact widths).
> * Everything is batch-size friendly (GroupNorm), AMP-safe.

## C) Loss (event-aligned)

* **Primary:** **Weighted BCE** (positive weight from class prevalence).
* **Aux:** **Soft Dice** on **binarized probs** (detach threshold 0.5 during loss) to penalize fragment/merge.
* **Boundary tolerance:** convolve labels with a **triangular kernel of ±1 s** to soften onset/offset (keeps timing crisp, reduces over-penalty for 1–2 s jitter).

## D) Training recipe

* **Sampler:** 50% windows with any seizure content; 50% background.

  * **Hard-negative mining:** each epoch, add top-K FP windows from prior eval.
* **Optimizer:** **AdamW**, lr **3e-4**, wd **0.05**, β=(0.9,0.999).
* **Schedule:** **cosine** with 10% warmup, **50–80 epochs**, **early-stop** on dev **sensitivity @ 10 FA/24h**.
* **Regularization:** dropout 0.1 in Mamba blocks; grad-clip **1.0**; **AMP** on.
* **Batch:** as VRAM allows (e.g., 4–8 windows/GPU). Multi-GPU DDP if available.

## E) Inference → Eventization (hysteresis) → Scoring

1. **Stitch** window probs back to the full timeline by **mean** in overlaps.
2. **Hysteresis** (globally, not per-window):

   * **τ\_on = 0.86**, **τ\_off = 0.78** (on > off to stop chatter).
3. **Morphology:** **open→close** with **kernel k=5** samples.
4. **Min duration:** **≥ 3.0 s**; drop shorter.
5. **No gap-merge** for NEDC; (only apply SzCORE’s 90 s merge when doing SzCORE runs).
6. **Metrics (primary):** **TAES** sensitivity vs **FA/24h**; report sensitivity at **10**, **5**, **2.5**, **1** FA/24h.
   **Secondary:** AUROC on raw probs (for sanity).

## F) Validation checklist (don’t skip)

* **Preproc parity:** channel order, filters, resample verified against a tiny oracle script (unit tests with known EDFs).
* **Shape tests:** `(B,19,15360) → … → (B,15360)` always; down/upsample math exact (no drift).
* **Determinism:** set seeds; assert eval reproducibility ±0.1% over three runs.
* **Scorer parity:** your TAES/OVERLAP vs NEDC binaries on a small set should match to the 4th decimal.

## G) Why this is the best fit (for TAES + low FP/day)

* **U-Net** nails **local morphology** (spikes, rhythmic build-up) with multiscale skips.
* **Bi-Mamba-2** gives **minute-scale** context and **state persistence** without O(T²) memory—precisely what reduces split/merge and false bursts.
* **Hysteresis** decodes like a clinician (commit later, release later), slashing false alarm chatter.
* The whole thing is **end-to-end**, per-timestep → directly **event-scored**.

---

## Drop-in module signatures (so your codebase stays SOLID)

```python
class UNet1DBackbone(nn.Module):  # (B, C=19, T) -> (feats, skips)
    def forward(self, x) -> Tuple[Tensor, List[Tensor]]: ...

class MontageGraphMixer(nn.Module):  # (B, D, T_b) -> (B, D, T_b)
    def forward(self, x, adj) -> Tensor: ...

class BiMambaTemporal(nn.Module):  # (B, D, T_b) -> (B, D, T_b)
    def forward(self, x) -> Tensor: ...

class UNet1DDecoder(nn.Module):  # (B, D, T_b), skips -> (B, 1, T)
    def forward(self, x, skips) -> Tensor: ...

class HysteresisEventizer:
    def __init__(self, tau_on=0.86, tau_off=0.78, k=5, min_dur=3.0): ...
    def __call__(self, probs_256hz, timestamps) -> List[Event]: ...

class TAEScorer:
    def score(self, ref_events, hyp_events) -> Dict[str, float]: ...
```

---

## Exact “go build” checklist (1 page)

* Wire **MNE** (or your preproc) to output fixed 19-ch @256 Hz + timestamps.
* Implement the **four** modules above + **Eventizer** + **TAEScorer**.
* Train with the **training recipe** as written; stop on **sens @ 10 FA/24h (dev)**.
* Report **TAES** curves and the four fixed points (10/5/2.5/1 FA/24h); include AUROC (raw probs).
* Keep SzCORE separate, clearly labeled (never mix it with NEDC claims).

This is the final call: **U-Net(1D) + Bi-Mamba-2 bottleneck + Hysteresis decoding, TAES-first.** Build exactly this, and you’re optimizing the thing that matters in clinic: **catch events, don’t spam alarms.**
