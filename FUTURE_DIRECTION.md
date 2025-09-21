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

Totally possible—if you wire it right. The trick is to **share one backbone** and attach **small task-specific heads**, then use **masked multi-task losses** so each batch can come from a different dataset with different labels. You train offline on windows, but you **design the heads + post-processing to run causally** so the exact same model can run live.

# Big picture

* **One backbone** (U-Net → ResCNN → Bi-Mamba) learns general EEG features.
* **Multiple heads** read those features:

  * `head_seizure` (TUSZ labels)
  * `head_artifact` (TUAR: blink, EMG, pop, etc.)
  * `head_rhythmic_periodic` (TUEV: LRDA, GRDA, LPD, GPD… non-ictal)
  * *(optional)* `head_abnormal` (TUAB: normal vs abnormal)
* **Masked loss** so each batch only contributes to the heads that have labels.
* **Inference fusion**: combine heads into one live **seizure score** with a small state machine (hysteresis + veto/min-duration).

# Why this works in real time

* Training uses 30–60s windows, but inference runs on a **sliding stream** (e.g., hop = 1s).
* Each hop: run backbone on the newest chunk (with overlap), get per-timestep logits from heads, then update the **stateful post-processor** (hysteresis) to emit/extend/end events.
* Artifact/rhythmic heads act as **contextual gates**: they **down-weight** seizure when artifact is high or **explain away** confusing non-ictal rhythms.

# Label strategy (what each dataset gives you)

* **TUSZ** → `y_seizure[t] ∈ {0,1}` (ictal timeline).
* **TUAR** → `y_artifact[t] ∈ {clean, blink, emg, pop, …}` (hard negatives).
* **TUEV** → `y_rp[t] ∈ {LRDA, GRDA, LPD, …}` (non-ictal brain patterns often mistaken as seizures).
* **TUAB** → weak/global `y_abnormal ∈ {0,1}` (help backbone generalize beyond seizure vs artifact).

# Training loop (concrete, minimal)

```python
# shared backbone: unet -> rescnn -> bimamba
features = backbone(x)             # [B, C, T]

logit_seiz = head_seizure(features)      # [B, 1, T]
logit_artf = head_artifact(features)     # [B, K_art, T]
logit_rp   = head_rp(features)           # [B, K_rp, T]
logit_abn  = head_abnormal(features.mean(-1))  # [B, 1] (global)

loss = 0.0
if batch.has_tusz:  loss += bce(sigmoid(logit_seiz), y_seiz_masked)
if batch.has_tuar:  loss += λ1 * ce(logit_artf,   y_artf_masked)
if batch.has_tuev:  loss += λ2 * ce(logit_rp,     y_rp_masked)
if batch.has_tuab:  loss += λ3 * bce(sigmoid(logit_abn), y_abn)

loss.backward(); opt.step()
```

* **Masked labels**: if a dataset doesn’t have a head’s label, you **don’t include** that term.
* Start with λ’s like `λ1=0.4, λ2=0.3, λ3=0.1`, then tune by FA/24h impact.

# Inference fusion (single seizure score)

```python
p_seiz = sigmoid(logit_seiz)               # [T]
p_art  = softmax(logit_artf).max(dim=1)    # max artifact prob per t
p_rp   = softmax(logit_rp).max(dim=1)      # max non-ictal RP prob per t

# Gate: explain-away & veto knobs
α, β = 0.6, 0.5
p_final = p_seiz * (1 - α * p_art) * (1 - β * p_rp)

# Hysteresis + min-duration (stateful across hops)
events = hysteresis(p_final, on=0.86, off=0.78, min_len=2.0)  # seconds
```

* This yields **stable**, low-FA decisions online.
* You can make the gate **learned** later (tiny MLP over \[p\_seiz, p\_art, p\_rp]).

# Real-time details you asked about

* **“Some heads are feature vs time…”**
  All heads read the **same temporal feature map**. Heads differ only in **label space**. That’s why multi-tasking works.
* **Bidirectional vs causal**

  * For **offline** benchmarks: Bi-Mamba **bidirectional** is great.
  * For **live** use: switch Bi-Mamba to **causal** (or allow a tiny **look-ahead** like 1–2 s). Everything else stays the same.
  * You can keep one codepath and toggle `bidirectional=False, lookahead=K`.
* **Windowing vs stream**
  Use overlapping chunks (e.g., 60s window, hop 1s). Maintain hidden state for causal blocks (Mamba) and **reuse** encoder features with a small cache to keep latency sub-100 ms per hop.

# Sampling that makes this work

* **Balanced sampler across datasets**: each step picks a dataset, then a window. Prevent TUSZ from dominating; you want the model to **see artifacts and non-ictal patterns often**.
* **Hard-negative mining**: periodically add windows that your current model flagged as seizure but were **artifact/TUEV**—this is where FA drops happen.

# What “success” looks like in numbers

* Offline (dev/test): **FA/24h** goes down after adding TUAR+TUEV heads (even at fixed sensitivity).
* Online (stream sim): latency <100 ms per hop; events align within ±1–2 s; far fewer “blip” alarms.

# If you want this plugged into your repo now

* Add heads in `models/detector.py` and loss masking in `train/loop.py`.
* Extend dataloader to emit `has_tusz/has_tuar/has_tuev/has_tuab` + masked labels.
* Add the fusion gate in `post/postprocess.py` (keep your hysteresis).
* Provide two configs: `multitask_offline.yaml` (bidirectional) and `multitask_streaming.yaml` (causal + small look-ahead).

If you want, I’ll sketch the exact `SeizureDetector` head stubs + the masked loss helper so you can paste it in.
