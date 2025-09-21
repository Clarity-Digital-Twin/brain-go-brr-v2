Short answer: **yes—use artifact/“non-seizure” corpora.** They help slash false positives when you either (a) pretrain on them, or (b) co-train an “artifact vs brain” head alongside seizure detection.

# Useful datasets

* **TUH EEG Artifact Corpus (TUAR)** — curated labels for eye movement, chewing, shiver, electrode pop/static/lead, muscle, incl. overlaps; built from the broader TUEG. Great for hard-negative mining and an auxiliary artifact head. ([Signal and Information Institute][1])
* **TUH EEG Events (TUEV)** — annotations of ictal and non-ictal rhythmic/periodic patterns (LPD, GPD, LRDA, GRDA, etc.). These are “non-epileptiform discharges” that confuse detectors; negative (or separate-class) exposure improves specificity. ([PMC][2])
* **TUH Abnormal (TUAB)** — global normal/abnormal labels; not artifact-typed, but helpful to broaden non-seizure brain phenotypes for contrastive/aux tasks. ([PMC][3])

# Does it help?

* Studies that **combine seizure + artifact detection** report **large false-alarm reductions** compared to seizure-only models (e.g., tree-based and embedded implementations; principle generalizes). ([Nature][4])

# How to use them in your stack (fast & effective)

1. **Multi-task head**
   Keep your main seizure head, add a small head for **artifact vs brain** (and optionally artifact subtype). Train with TUAR labels + your usual seizure labels. At inference, down-weight seizure probability when artifact head is confident.

2. **Hard-negative mining**
   Sample **artifact windows** that previously triggered false alarms; up-weight them as negatives. This is the highest ROI trick with TUAR. ([Signal and Information Institute][1])

3. **Curriculum / pretrain**
   Briefly pretrain the U-Net/ResCNN front-end to classify **artifact vs clean**; then fine-tune the full model on seizures. Stabilizes early layers to ignore EMG/EOG texture.

4. **Augmentations that mimic artifacts**
   Add EMG-like high-beta bursts, eye-blink ramps, electrode pops (steps/offsets) as **non-seizure** augmentations so the model sees “lookalikes” during training.

5. **Evaluation gates**
   Report metrics with and without **artifact suppression** enabled to prove real-world value (false alarms per 24h drop).

# Quick mapping to your modules

* **ResCNN**: learns local artifact textures (muscle, chewing). Give it TUAR exposure so it stops calling them seizures.
* **Bi-Mamba-2**: learns long-context “artifact persistence” (e.g., muscle noise across long spans) vs evolving ictal context.
* **U-Net decoder + skips**: preserves sharp on/off so **electrode pops** don’t look like ictal edges.
* **Post-processing**: keep hysteresis + min-duration; combine with artifact head to veto short artifact bursts.

# Practical starter recipe

* Mix **TUSZ (seizures)** + **TUAR (artifacts)** + **TUEV (periodic/rhythmic non-ictal)** in one dataloader with labels `{seizure, nonseizure_brain, artifact_subtype}`.
* Loss = `BCE(seizure)` + `λ * CE(artifact_subtype)` with small λ (e.g., 0.2).
* Inference: `p_final = p_seizure * (1 - α * p_artifact)` (α≈0.6–0.8 to start).
* Track **FA/24h** on a held-out split before/after adding artifact conditioning.

If you want, I can sketch the exact dataloader schema + minimal multi-task head patch for your `SeizureDetector` to plug TUAR/TUEV in cleanly.

[1]: https://isip.piconepress.com/projects/nedc/html/tuh_eeg/?utm_source=chatgpt.com "Temple University EEG Corpus - Downloads"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4865520/?utm_source=chatgpt.com "The Temple University Hospital EEG Data Corpus - PMC"
[3]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4868184/?utm_source=chatgpt.com "Automated Identification of Abnormal Adult EEGs - PMC"
[4]: https://www.nature.com/articles/s41598-024-52551-0?utm_source=chatgpt.com "Minimizing artifact-induced false-alarms for seizure ..."
