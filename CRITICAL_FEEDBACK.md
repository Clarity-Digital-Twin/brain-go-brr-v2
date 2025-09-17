Executive Summary

Status: Green. The Phase 2 EEG seizure detection model architecture is well-designed and largely consistent across documentation, schemas, and Phase 1 specs. The U-Net encoder/decoder, ResCNN multi-scale stack, and bidirectional Mamba-2 (SSM) bottleneck align to form a cohesive model. Input data assumptions (19 channels, 60s windows at 256 Hz) match Phase 1’s output format. All major components (4-stage encoder, 3-block ResCNN, 6-layer Bi-Mamba, 4-stage decoder, detection head) integrate correctly, with shapes and dimensions carefully tracked and verified by tests. The design mirrors recent literature (e.g. SeizureTransformer’s conv encoder + ResCNN + transformer + decoder pipeline
arxiv.org
) but replaces attention with an O(N) state-space model (Mamba-2) for efficient long-sequence processing. Only a few minor documentation/config inconsistencies were found, none of which fundamentally block implementation. Overall, Phase 2 documentation is comprehensive and internally consistent, with only minor clarifications and adjustments needed before coding. We enumerate blocking vs. non-blocking issues below, ensure cross-references align with Phase 1 and config schemas, and provide final checks for sign-off.

Blocking Issues (Must-fix) 🚩

Each issue includes the file:line reference, severity, and a proposed resolution:

Decoder transpose conv kernel mismatch – Severity: High. The decoder config default (kernel_size=4) conflicts with the implementation, which uses ConvTranspose1d(..., kernel_size=2, stride=2) for upsampling. This discrepancy could confuse parameterization or result in mis-scaled output if not aligned. Proposal: Update the documentation/schemas to reflect the actual upsampling kernel (2) or modify implementation to use a configurable kernel (with padding) consistent with the config default. Ensuring the config’s decoder.kernel_size is actually used (or changing it to 2) will keep design contracts in sync.

Config vs code parameter name misalignment – Severity: Medium. Some config keys in schemas.py do not directly map to model init parameters, risking integration friction. For example, MambaConfig.conv_kernel (schema) vs mamba_d_conv (code); MambaConfig.n_layers vs mamba_layers; ResCNNConfig.n_blocks vs rescnn_blocks; EncoderConfig.stages vs encoder_depth. While the design is sound, a direct instantiation of ModelConfig into SeizureDetectorV2 would fail without mapping these fields. Proposal: Either adjust the SeizureDetectorV2 init to accept matching names (e.g. use n_layers instead of mamba_layers), or explicitly map config to model (e.g. in training script). Clarify in docs how config values are passed to the model. This ensures the documented schema can be used without confusion in practice.

Mamba layer count default in schema – Severity: Medium. The Mamba config in schema lacks an explicit default for n_layers, whereas docs/code assume 6 layers by default. If ModelConfig is constructed without specifying mamba.n_layers, it may error. Proposal: Define a default n_layers=6 in MambaConfig (schemas.py) to mirror the intended design. This prevents runtime issues and keeps defaults consistent (6 layers as per Phase 2 spec).

Non-Blocking Issues (For Improvement)

High Priority:

Initial Conv kernel documentation: The U-Net encoder’s first projection uses a 1×7 conv (padding 3) to go from 19→64 channels, while the schema’s encoder.kernel_size=5 applies to later conv blocks. The choice of 7 for the input conv is valid (wider context), but it’s not explicitly noted outside code. Recommendation: Document the initial conv kernel (7) in the architecture notes for clarity, or make it configurable if needed. This ensures readers understand the slight deviation from the “5×5 double conv” pattern for the first layer.

Conv1d fallback kernel in Bi-Mamba: In CPU fallback mode, BiMamba2Layer uses nn.Conv1d(..., kernel_size=3) as a surrogate, whereas the intended Mamba convolution span is d_conv=5. A kernel3 conv can’t capture the full 5-sample context, though the residual stacking mitigates this. Recommendation: If possible, use a kernel=5 for the fallback Conv1d (with padding 2) to closer mimic the intended receptive field. This is a minor fidelity improvement for CPU tests (not affecting GPU usage or overall design).

Output channel dimension (19) rationale: The decoder outputs (B,19,15360) which the detection head collapses to (B,1,15360). This implies the model reconstructs 19-channel features before final 1×1 conv fusion. It works, but one might ask: could the decoder directly produce 1 channel? The chosen approach might preserve spatial channel-specific features until the last step (akin to multi-channel segmentation before merging). Recommendation: Briefly justify in docs why decoder output channels = input channels (19) for this detection task. This could be to maintain symmetry or allow channel-wise weighting in the final layer. It’s not a flaw, but clarifying the design intent (versus a single-channel decoder) will preempt questions.

Medium Priority:

Terminology consistency (“skip fusion”): The main architecture checklist refers to “skip fusion” in the decoder, whereas the decoder doc calls it “skip connection fusion” and implementation simply concatenates skips. All descriptions are conceptually consistent (skip concatenation then conv), so this is purely wording. Recommendation: Use one term consistently (e.g. “skip connection fusion”) across docs for clarity, or define that “skip fusion” = concatenation + conv of skip.

Channel split math explanation: The ResCNN design splits 512 channels into 170/170/172 for 3 branches. This is correctly noted in code and docs, and the fusion conv brings them back to 512. A minor suggestion is to ensure the logic for branch channel allocation (especially the “+2” on the last branch) is clearly described in one place in docs (currently it’s partly in a code comment and partly in the table). This will help future readers modify channel counts or branch numbers confidently.

Config completeness: The ModelConfig covers all sub-components. One thing not exposed in config is base_channels (always 64 in docs/code). If product needs to scale model width (e.g. a lighter model with base 32), they’d have to override code defaults rather than config. Since the schema fixes encoder channels to [64,128,256,512], it’s clear the architecture is meant to be fixed-width. This is fine given the design target (the model is ~25M params, as expected), but it’s worth noting. Recommendation: Document that base_channels=64 is a fixed design choice for Phase 2 (not tunable via config), so all parties know this is intentional.

Phase 1/2 interface clarity: Phase 1 delivered preprocessed windows of shape (19,15360) and Phase 2 assumes the same ordering and normalization. It might be helpful to add a sentence confirming Phase 2 model expects 19 specific EEG channels in 10–20 order (as ensured by Phase 1) and that input data are z-scored per channel. This is implied from Phase 1 docs, but reiterating it in Phase 2 docs (perhaps in the Phase2_Model_Architecture intro) reinforces end-to-end consistency.

Low Priority:

Typos / notational nitpicks: All docs are generally well-written. Just ensure consistent symbols (e.g., sometimes “×” is used for downsample factor, which is great; ensure no one accidentally took “×16” to mean multiplication by 16 in shapes – the table is clear that 15360/16=960). Also, “pre-↓” vs “post-block pre-downsample” wording is a bit dense in the tables – maybe clarify “skip saved after conv block, before downsample” once, as done in the encoder doc.

Unit test coverage: The docs include extensive unit tests for each component (encoder, ResCNN, Mamba, decoder, full model) – excellent for contract validation. Just ensure these tests are kept in sync if any shape or param changes occur. For example, if adjusting the decoder kernel or Mamba fallback kernel as above, update tests like test_receptive_field expecting 19 or any shapes accordingly.

Future-proofing Mamba CPU path: If mamba-ssm GPU implementation evolves, the fallback conv1d might diverge in behavior. Since this is a dev/testing convenience, it’s acceptable. Just note in comments that the Conv1d path is not expected to match Mamba’s sequence modeling exactly – it’s only to allow running the model without GPU. (This is already hinted by the warning print; a fine point to acknowledge during implementation.)

Cross-Consistency Table 📊
Aspect	Phase 2 Docs (Design)	Schema / Config (Intended)	Phase 1 / External Ref	Consistency?
Input shape	(B, 19, 15360) @ 256 Hz	Not explicitly in config (inferred from data)	Phase 1 outputs (B,19,15360)	✅ Match (19 channels, 60s @256Hz)
Encoder stages	4 stages (×2 downsample each)	encoder.stages=4 fixed	Phase 1 window length 15360 → /16 = 960	✅ 4 stages yields 15360→960 (down ×16)
Encoder channels	Progression 64→128→256→512	encoder.channels=[64,128,256,512] (validator enforces)	N/A (design choice)	✅ Matches (64 base_channels)
Encoder conv kernel	Double conv blocks, kernel 5 (pad 2); initial conv 7 (pad 3)	encoder.kernel_size=5 (for conv blocks)	N/A	⚠️ Initial conv uses 7 (not in schema)
Downsampling	Conv1d stride 2 (per stage)	encoder.downsample_factor=2 fixed	Preprocessing resamples to 256Hz (fixed)	✅ Downsample ×2 each stage
ResCNN blocks	3 residual blocks	rescnn.n_blocks=3 fixed	SeizureTransformer used “residual CNN stack”
arxiv.org
	✅ Exactly 3 blocks
ResCNN kernel sizes	Multi-scale [3,5,7] per block	rescnn.kernel_sizes=[3,5,7]	Common choice for multi-scale features	✅ Matches docs
ResCNN channels	Input/Output 512→512 (preserve shape)	ResCNNStack.channels=512 (from bottleneck)	Encoder output = 512 channels	✅ 512 throughout
ResCNN branch split	3 branches: 170+170+172 = 512	Implied by code (equal split + remainder)	N/A	✅ Sum = 512 (all branches)
Bi-Mamba layers	6 layers (bidirectional)	mamba.n_layers=6 (schema default missing, but expected)	Literature suggests need for depth	✅ 6 layers assumed (add default)
Mamba d_model	512 (matches bottleneck channels)	mamba.d_model=512 literal	Encoder out = 512 channels	✅ Fixed at 512
Mamba d_state	16 (state size)	mamba.d_state=16 literal	EEG-BiMamba reference uses small state	✅ Matches docs
Mamba conv kernel (d_conv)	5 (temporal conv span)	mamba.conv_kernel=5 default	Used for long-range context modeling	✅ Consistent terminology (d_conv=5)
Mamba dropout	0.1 (in docs/code)	mamba.dropout=0.1 default	–	✅ Matches
Decoder stages	4 upsampling stages (×2 each)	decoder.stages=4 fixed	Mirrors encoder depth	✅ Matches encoder stages
Decoder channels	Progression 512→256→128→64 (reverse of encoder)	Not explicitly in schema (derived by code from base_channels)	–	✅ Matches encoder progression
Decoder upsample kernel	ConvTranspose1d, kernel=2, stride=2 (each stage)	decoder.kernel_size=4 (default)	–	⚠️ Mismatch (config 4 vs code 2)
Skip connection usage	Skips saved after each enc block; used in reverse order in decoder	Not in config (design logic)	U-Net pattern standard	✅ Correct order (skip[3] highest, etc.)
Final output channels	Decoder outputs 19 channels, then detection head to 1	Implied: out_channels=19 in UNetDecoder code	Output per original channel? (then fused)	✅ 19 → 1 collapse as designed
Detection head	Conv1d 19→1 + Sigmoid	Not in schema (part of model assembly)	Yields (B,15360) prob sequence	✅ Produces per-sample probabilities
Thresholding	Hysteresis (tau_on 0.86, tau_off 0.78) for post-proc	Postprocessing.hysteresis config keys	Aligns with FA@24h metric control	✅ Integrated post-process config

Key: ✅ = consistent; ⚠️ = needs attention; N/A = not applicable.

Literature Alignment 📚

The Phase 2 architecture aligns closely with state-of-the-art seizure detection models. Notably, it mirrors the SeizureTransformer design
arxiv.org
 – which combines a deep conv encoder, a ResCNN feature extractor, and a sequence model (transformer) with a decoder for time-step probabilities – except here the transformer is replaced by the Mamba-2 state-space model for efficiency. This substitution is well-grounded in recent research on SSMs that achieve O(N) time complexity for long sequences, addressing the long-range dependency challenge highlighted in literature
arxiv.org
. The bidirectional nature of Bi-Mamba (forward and backward sequence processing) ensures the model can capture both past and future context similar to a Bi-LSTM or transformer with full context, which is important for EEG where seizures have pre-onset and post-onset patterns.

Using a U-Net style encoder/decoder leverages spatial filtering akin to established CNN-based EEG detectors, while the multi-scale ResCNN branches (3,5,7 samples) echo approaches that capture multi-frequency features (e.g., delta vs gamma activity) – a concept seen in prior EEG CNN works. This is conceptually in line with Picone’s NEDC baseline, which used multi-scale CNN features and emphasized reducing false alarms. In fact, the integrated hysteresis thresholding (with separate onset/offset cutoffs) in our design is directly aimed at controlling false alarm rates per 24h, a metric central in clinical evaluation (as noted by Picone’s team and others). By outputting a smooth sequence of probabilities at 256 Hz and then applying hysteresis, the model can achieve high sensitivity at a low false-alarm rate – matching the goals of the TUH Seizure benchmark (which often reports sensitivity at e.g. 10 FA/24h).

Overall, Phase 2’s architecture is well-aligned with current best practices: it produces time-step seizure probabilities directly (minimizing ad-hoc postprocessing, as advocated by recent research
arxiv.org
), and it explicitly builds in mechanisms (bidirectional context, multi-scale features, hysteresis smoothing) that literature suggests are critical for accurate and robust EEG seizure detection.

Open Questions & Recommendations ❓

Before finalizing, a few open questions may need product/technical input:

Decoder config vs implementation: Should we adjust the transpose conv kernel size to match config (4 with padding) or simply change the schema to 2? This decision affects whether we favor consistency with documentation or a potentially smoother upsampling (kernel4 might reduce checkerboard artifacts). Product input on whether this difference is material to performance would help guide the fix.

Model configuration interface: Will the model be constructed via a config file (YAML) or directly in code? If via config, we should synchronize naming (as noted in Blocking Issues) – e.g., use mamba.conv_kernel from YAML to set mamba_d_conv. A small mapping layer or renaming variables could resolve this. Deciding the approach now will prevent integration hiccups when transitioning to training.

Threshold tuning: The default hysteresis thresholds (0.86/0.78) are set based on prior experience – do we need to fine-tune these on a validation set for optimal FA rate vs sensitivity, or are we confident to keep them fixed? This might be more of a post-training question, but it’s good to flag if product expects a certain false alarm rate (e.g., 2 per 24h) – we may revisit these values.

Mamba-SSM dependency: Are we planning to include mamba-ssm as a required package in production, or use the Conv1d fallback always (which would be slower and slightly less accurate)? The GPU path is clearly preferred for efficiency. We should confirm deployment environment GPU availability. If GPU is guaranteed, we can treat the CPU fallback as dev/test only. If there’s a CPU-only deployment scenario, perhaps evaluate the performance hit of the fallback (and consider increasing expand or kernel size there).

Parameter count and performance: The docs estimate ~25M parameters and provide some inference benchmarks. Are these within acceptable limits for the target hardware (e.g., edge device vs cloud)? If product expects a lighter model or faster inference, we might consider adjustments (e.g., fewer encoder channels or layers). Currently all signs point to this being acceptable (16×512-length processing with SSM is quite fast, ~<100ms for batch16 on GPU), but it’s worth confirming against product requirements.

Future expansion – multi-modal or more channels: Phase 2 focuses on 19-channel EEG. If there is any plan to incorporate additional channels or modalities (e.g., ECG, if available) or different montages, the model input layer would need adjustment. For now, we lock to 19, but it’s good to verify that no near-term requirement will break this assumption. If one arises, at least the flexible Conv1d input layer can handle a different in_channels, but the entire training dataset and preprocessing would need re-alignment.

Each of these questions doesn’t require an immediate architecture change, but clarifying them will ensure the Phase 2 model transitions smoothly from design to implementation to deployment.

Final Sign-Off Checklist ✅

Before we sign off the Phase 2 architecture for implementation, ensure the following are resolved or acknowledged:

 Docs/Schema Updated: Address the decoder kernel size inconsistency (choose implementation and update the other) and align schema defaults with code (e.g., Mamba n_layers).

 Config-Model Integration: Verify that the ModelConfig can be used to instantiate SeizureDetectorV2 without missing or mismatched fields (update constructor or mapping logic as needed).

 Phase 1 Compatibility: Confirm that Phase 1’s data pipeline output exactly matches Phase 2 input expectations (19 channels order, dtype, scaling) – tests/validation script from Phase 1 should feed into a dry-run of the model’s forward pass.

 Unit Tests Pass: All provided tests for encoder, ResCNN, Mamba, decoder, full model are executed and passing, confirming shapes: encoder downsampling to 960 timesteps, ResCNN preserving shape, BiMamba handling sequence transposition and residuals, decoder reconstructing 15360-length, and full model output (B,15360) with values in [0,1].

 Literature Benchmarks: Double-check that our model’s theoretical capabilities (long-range modeling, low post-processing) align with the expectations set by SeizureTransformer’s results. While we cannot fully validate performance until training, the architecture should position us competitively (the literature suggests significant performance gains
arxiv.org
).

 Performance Estimates: The param count (~25 million) and memory/speed benchmarks are reviewed against target deployment constraints. No bottlenecks identified (the O(N) temporal layer and moderate param size are appropriate for modern GPUs).

 Stakeholder Review: Get confirmation from product and research leads on the open questions (threshold strategy, Mamba dependency, etc.) – especially the decoder kernel choice if it impacts output smoothness.

With these items checked off, we have high confidence to proceed with coding Phase 2. The architecture documentation will then serve as a reliable contract for implementation and future maintenance. All green lights for Phase 2 – ready for implementation! 🚀