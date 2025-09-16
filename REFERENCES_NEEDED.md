Yes. Keep the spec and back it with these **best-fit references** (one or two per concept). Short, high-signal, and enough to cite in methods.

## Architecture (U-Net + ResCNN placement)

* **U-Net (biomed segmentation):** Ronneberger et al., *MICCAI* 2015 — canonical encoder-decoder with skips.
* **SeizureTransformer (U-Net + ResCNN in EEG):** Wu et al., 2025 — shows the ResCNN stack at the bottleneck before the temporal block (we swap their attention for Mamba).

## Temporal module (linear-time, long context)

* **Mamba (Selective State-Space Models):** Gu & Dao, 2023 — linear-time sequence modeling; template for our Mamba blocks.
* **S4 (SSM antecedent):** Gu et al., 2021 — long-sequence SSM foundations used widely before Mamba.

## Event scoring / evaluation (what we optimize)

* **TAES (Time-Aligned Event Scoring):** Shah et al., 2021 — defines TAES and why overlap-only can mislead.
* **NEDC EEG Evaluation (official scorer):** Picone group, NEDC EEG Eval v6.0.x — dataset-matched reference implementation for TUSZ.

## Post-processing (what we do after logits)

* **Hysteresis thresholding (two-threshold start/stop):** Canny, 1986 — classic double-threshold with connectivity; identical logic adapted to 1D sequences.
  (Practical API reference: van der Walt et al., *scikit-image*, 2014 “hysteresis\_threshold”.)
* **Morphological open/close:** Serra, 1982 (*Mathematical Morphology*); Soille, 2003 (*Morphological Image Analysis*). Standard for spike removal and gap filling in 1D/2D signals.

## Stitching (overlap-average)

* **Overlap-Add (OLA) principle:** Crochiere & Rabiner, 1983 (*Multirate DSP*) or Oppenheim & Schafer, *Discrete-Time Signal Processing*. Justifies uniform averaging in overlaps.

## Losses / imbalance

* **Generalized Dice (class imbalance):** Sudre et al., 2017 — Dice-family loss for imbalanced segmentation (we pair with weighted BCE).

## Sampler / hard negatives

* **Online Hard Example Mining:** Shrivastava et al., 2016 — the standard recipe for mining FPs to improve precision.

## Optimization & layers (for completeness of Methods)

* **AdamW:** Loshchilov & Hutter, 2019 — decoupled weight decay.
* **Cosine annealing (warm restarts):** Loshchilov & Hutter, 2017.
* **GroupNorm:** Wu & He, 2018 — stable with small batch sizes.
* **Depthwise-separable conv:** Howard et al., 2017 (*MobileNet*).
* **SiLU / Swish activation:** Ramachandran et al., 2017.

## Streaming normalization (long recordings)

* **Welford’s one-pass mean/variance:** Welford, 1962 — numerically stable streaming z-score.

If you want the bare-minimum cites in the paper:

* Ronneberger 2015; Wu 2025 (ST)
* Gu & Dao 2023 (Mamba); Gu 2021 (S4)
* Shah 2021 (TAES); NEDC eval docs
* Canny 1986 (hysteresis); Serra 1982 (morph)
  Everything else can live in the supplementary “implementation details” section.
