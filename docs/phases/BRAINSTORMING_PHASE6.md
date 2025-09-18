# BRAINSTORMING_PHASE6 — Streaming Infrastructure Research & Implementation Strategy

**Purpose**: Consolidate all streaming research, identify the best approaches from literature, and determine which reference repos/papers to leverage for Phase 6 implementation.

## 🎯 Executive Summary

After extensive research into streaming inference for EEG/medical signals, **THREE KEY INSIGHTS** emerge:

1. **Mamba already has streaming support** - The recurrent mode in `mamba_ssm` is designed for stateful inference
2. **Don't reinvent buffering** - MNE-LSL or FieldTrip patterns are battle-tested for EEG streaming
3. **Our existing code is 70% there** - We have HysteresisState, StreamingPostProcessor; just need to add duration filtering & flush

## 🔬 Research Findings

### 1. **Mamba's Built-in Streaming Capabilities** ⭐⭐⭐⭐⭐

**Why this matters**: We're already using Bi-Mamba-2, and it has THREE modes:
- **Convolutional** (for parallel training) ← We use this
- **Recurrent** (for streaming inference) ← We SHOULD use this for Phase 6
- **Continuous** (for theory/analysis)

**Key implementation detail from `mamba_ssm/ops/selective_scan_interface.py`**:
```python
# Mamba maintains hidden states between chunks!
def selective_scan_fn(
    u, delta, A, B, C, D=None,
    z=None, delta_bias=None, delta_softplus=False,
    return_last_state=False,  # <-- KEY FLAG
    prev_state=None,           # <-- PASS PREVIOUS STATE
):
    """
    u: (batch, length, d_in)
    prev_state: (batch, d_in, d_state) - carry this between chunks!
    """
```

**Action**: Use Mamba's recurrent mode with state caching between chunks

### 2. **EEG-BiMamba Paper (Our Literature)** ⭐⭐⭐⭐

From `/literature/markdown/EEG-BIMAMBA/EEG-BIMAMBA.md`:
- Implements bidirectional Mamba for EEG (exactly what we're doing!)
- Handles variable-length sequences via ST-Adaptive module
- **Key insight**: They use class tokens for temporal adaptability
- Achieves "fast inference speed and efficient memory-usage in processing long EEG signals"

**Action**: Study their bidirectional implementation for streaming hints

### 3. **MNE-LSL vs FieldTrip Buffer** ⭐⭐⭐⭐

**MNE-LSL** (Python-native):
- Pure Python implementation with Lab Streaming Layer
- Ring buffer with head/tail pointers
- Ships with liblsl packaged in wheels
- Direct integration with MNE (which we already use)

**FieldTrip** (C/MATLAB):
- Mature 10+ year solution
- TCP server architecture
- Multiple concurrent readers
- More complex setup but proven in production

**Verdict**: Use MNE-LSL for simplicity; keep FieldTrip as fallback

### 4. **Real-Time Performance Benchmarks (2024)** ⭐⭐⭐

Recent papers show these are achievable targets:
- **56.7ms inference** per chunk (Real-time Sub-milliwatt paper, Oct 2024)
- **30MB memory footprint** for entire pipeline
- **<100ms latency** per 10s chunk is standard
- **5× faster than Transformers** with Mamba

### 5. **Continuous Batching Research (2024-2025)** ⭐⭐⭐

Key papers:
- **"Memory-aware Dynamic Batching"** (March 2025, arXiv:2503.05248)
  - 8-28% throughput gains with dynamic batch sizing
  - Real-time memory monitoring

- **"BucketServe"** (July 2025, arXiv:2507.17120)
  - Groups by sequence length to minimize padding
  - 3.58× higher throughput than baseline

**Relevance**: These patterns work for variable-size chunks in streaming

### 6. **Pyannote-style Stitching** ⭐⭐⭐

From speaker diarization community:
- Overlap-add for smooth boundaries
- Proven in production VAD systems
- Handles chunk boundaries elegantly
- See: https://huggingface.co/pyannote/segmentation

## 📚 Literature & Reference Repos to Leverage

### Must-Have Reference Repos to Clone:

```bash
# 1. Mamba official (for selective_scan_interface.py)
git clone https://github.com/state-spaces/mamba
# Study: mamba_ssm/ops/selective_scan_interface.py
#        mamba_ssm/models/mixer_seq_simple.py (stateful example)

# 2. MNE-LSL (for streaming infrastructure)
git clone https://github.com/mne-tools/mne-lsl
# Study: mne_lsl/stream/stream_lsl.py (ring buffer)
#        examples/30_real_time_evoked.py

# 3. Pyannote-audio (for stitching patterns)
git clone https://github.com/pyannote/pyannote-audio
# Study: pyannote/audio/core/inference.py (sliding window)
#        pyannote/audio/pipelines/utils/diarization.py (stitching)
```

### Existing Reference Repos We Have:

1. **`/reference_repos/mamba`** ✅ - Already have it!
2. **`/reference_repos/braindecode`** - Has some streaming utils
3. **`/reference_repos/nedc-bench`** - For NEDC scoring (critical for our benchmarks)

### PDFs to Study:

From our `/literature/pdfs/`:
- **EEG-BIMAMBA.pdf** - Bidirectional Mamba implementation details
- **picone-2021-NEDC-SCORING.docx** - NEDC evaluation protocol (our target)

New papers to download:
- **Stateful Conformer** (arXiv:2312.17279) - Cache-based inference patterns
- **Real-time Sub-milliwatt** (arXiv:2410.16613) - Edge deployment proof

## 🏗️ Proposed Architecture (Best of All Worlds)

### Core Components:

```python
# 1. STREAMING INGESTION (MNE-LSL pattern)
class StreamIngester:
    def __init__(self, chunk_size=2560, overlap=256):
        self.ring_buffer = RingBuffer(capacity=chunk_size * 2)
        self.overlap = overlap  # For morphology context

    def chunk_generator(self, edf_path):
        # Yield overlapping chunks
        pass

# 2. STATEFUL MAMBA (from mamba_ssm)
class StatefulBiMamba:
    def __init__(self, model):
        self.model = model
        self.forward_states = [None] * 6  # 6 layers
        self.backward_states = [None] * 6

    def process_chunk(self, x):
        # Use selective_scan_fn with prev_state
        # Return new states for next chunk
        pass

# 3. BOUNDARY-SAFE STITCHING (pyannote pattern)
class StreamStitcher:
    def __init__(self, overlap=256):
        self.overlap = overlap
        self.prev_chunk_end = None

    def stitch(self, curr_probs, prev_probs=None):
        # Overlap-add at boundaries
        # Weight by distance from edge
        pass

# 4. ENHANCED STREAMING PROCESSOR (our existing + new)
class StreamingPostProcessor:
    # Our existing code PLUS:
    - Duration filtering with event buffer
    - Flush() method for stream end
    - SeizureEvent creation with confidence
```

### Data Flow:

```
EDF File → MNE-LSL Ingester → Overlapping Chunks (10s + 1s overlap)
    ↓
Stateful Bi-Mamba (with cached states)
    ↓
Probability Stream → Stitcher (overlap-add)
    ↓
Hysteresis (existing) → Morphology (existing) → Duration (NEW)
    ↓
SeizureEvent objects → NEDC Scoring
```

## 🚀 Implementation Strategy

### Phase 6.1: Core Streaming (2 days)
1. **Hour 1-4**: Study Mamba's selective_scan_interface.py
2. **Hour 5-8**: Implement StatefulBiMamba wrapper
3. **Day 2**: Add duration filtering & flush to StreamingPostProcessor

### Phase 6.2: Ingestion & Stitching (1 day)
1. **Morning**: Implement MNE-LSL style ring buffer
2. **Afternoon**: Add pyannote-style overlap-add stitching

### Phase 6.3: Testing & Benchmarking (2 days)
1. **Day 1**: Streaming vs offline equivalence tests
2. **Day 2**: Latency benchmarks, memory profiling

## ⚡ Performance Targets (From Research)

| Component | Target | Source |
|-----------|--------|--------|
| **Chunk Latency** | <100ms per 10s | Industry standard |
| **Memory** | <50MB overhead | Real-time SNN paper |
| **Throughput** | >100× realtime | Mamba paper claims |
| **State Size** | ~10MB per model | Mamba d_state=16 |
| **Equivalence** | ±1 sample vs offline | Clinical requirement |

## 🎓 Key Papers & Links

### Must-Read Papers:
1. **Mamba Paper** (Dec 2023): Linear-time sequence modeling
   - arXiv:2312.00752
   - Section 3.4: Recurrent mode for inference

2. **Mamba-2** (May 2024): State Space Duality
   - Blog: https://goombalab.github.io/blog/2024/mamba2-part1-model/
   - Key: SSD algorithm for efficient matmuls

3. **Stateful Conformer** (Dec 2023): Cache-based streaming ASR
   - arXiv:2312.17279
   - Pattern: Activation caching between chunks

### GitHub Resources:
- **Mamba SSM**: https://github.com/state-spaces/mamba
- **MNE-LSL**: https://mne.tools/mne-lsl/stable/
- **Pyannote**: https://github.com/pyannote/pyannote-audio
- **NEDC Tools**: Our existing `/reference_repos/nedc-bench`

## ⚠️ Avoid These Pitfalls

### Don't Reinvent:
- ❌ Custom SSM scan implementations (use Mamba's)
- ❌ Ring buffer from scratch (use MNE-LSL pattern)
- ❌ TCP/WebSocket server (use LSL if needed)
- ❌ Event scoring (use NEDC tools we have)

### Don't Chase:
- ❌ Neuromorphic/SNN approaches (different hardware)
- ❌ FieldTrip C server (unless MNE-LSL fails)
- ❌ Complex batching (we process one stream)

## 📋 Action Items

### Immediate (Before Starting Phase 6.1):
```bash
# 1. Study Mamba recurrent mode
cd reference_repos/mamba
grep -r "return_last_state" .
grep -r "prev_state" .

# 2. Check our BiMamba implementation
cd ../..
grep -r "class.*Mamba" src/

# 3. Review existing streaming.py
cat src/experiment/streaming.py | head -100
```

### To Download/Clone:
1. ✅ Mamba repo (already have)
2. ⏳ MNE-LSL (needed)
3. ⏳ Pyannote-audio (for stitching reference)

### PDFs to Read:
1. ✅ EEG-BIMAMBA.pdf (our literature)
2. ✅ NEDC scoring docs (our target metric)
3. ⏳ Stateful Conformer paper
4. ⏳ Mamba-2 paper (for SSD insights)

## 💡 Final Recommendation

**GO WITH**: Mamba recurrent mode + MNE-LSL patterns + Our existing code

**WHY**:
- We're 70% there with existing StreamingPostProcessor
- Mamba already supports stateful inference
- MNE-LSL is Python-native and integrates with our stack
- This is 2-3 days of work, not 5

**SKIP**:
- External auditor's SzCORE suggestion (we use NEDC)
- FieldTrip buffer (too complex for our needs)
- Neuromorphic approaches (different paradigm)

## 🔬 CODEBASE ANALYSIS - Current State vs Phase 6 Needs

### **What We Have (src/experiment/):**

1. **`streaming.py`** - Already has:
   - ✅ `HysteresisState` dataclass (lines 19-27)
   - ✅ `StreamingPostProcessor` class (lines 30-100+)
   - ✅ Stateful hysteresis across chunks
   - ✅ Buffer management for morphology
   - ❌ Missing: Duration filtering
   - ❌ Missing: Event finalization/flush
   - ❌ Missing: SeizureEvent creation

2. **`models.py`** - BiMamba2 implementation:
   - ✅ `BiMamba2Layer` class (line 306)
   - ✅ Uses `mamba_ssm.Mamba2` when available (lines 340-353)
   - ✅ Bidirectional processing (forward + backward)
   - ❌ Currently using convolutional mode for training
   - ❌ No stateful/recurrent mode for streaming

3. **`postprocess.py`** - Has all algorithms:
   - ✅ `apply_hysteresis()`
   - ✅ `apply_morphology()` (CPU/GPU paths)
   - ✅ `filter_duration()`
   - ✅ `stitch_segments()`

### **Critical Integration Points:**

```python
# 1. STATEFUL MAMBA - Need to modify BiMamba2Layer
class StatefulBiMamba2Layer(BiMamba2Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward_state = None
        self.backward_state = None

    def forward_streaming(self, x, use_cache=True):
        # Use Mamba's selective_scan_fn with:
        # - return_last_state=True
        # - prev_state=self.forward_state
        # This is the KEY change!
```

```python
# 2. RING BUFFER - Adapt MNE-LSL pattern
class RingBuffer:
    def __init__(self, capacity=5120):  # 20s @ 256Hz
        self.buffer = np.zeros(capacity)
        self.head = 0
        self.tail = 0
        self.size = 0

    def push(self, data):
        # Circular write with overlap management
        pass
```

```python
# 3. OVERLAP-ADD - From Pyannote
def overlap_add_chunks(chunks, overlap=256):
    # Hamming window
    window = np.hamming(overlap * 2)
    # Weighted aggregation at boundaries
    # Lines 562-644 from pyannote/inference.py
```

### **Phase 6 Implementation Roadmap:**

#### **Phase 6.1: Stateful Mamba (HIGH PRIORITY)**
1. Create `StatefulBiMamba2Layer` that wraps our existing model
2. Add `return_last_state=True` to Mamba2 forward calls
3. Cache states between chunks (forward & backward)

#### **Phase 6.2: Enhanced Streaming (MEDIUM)**
1. Add duration filtering to `StreamingPostProcessor`
2. Add `flush()` method for stream termination
3. Create `SeizureEvent` objects with confidence

#### **Phase 6.3: Boundary Management (LOW)**
1. Implement ring buffer for overlapping chunks
2. Add overlap-add stitching for smooth boundaries
3. Handle warm-up/cool-down at chunk edges

### **Memory & Performance Estimates:**

| Component | Memory | Latency |
|-----------|--------|---------|
| BiMamba2 states (6 layers) | ~6MB | <10ms |
| Ring buffer (20s) | ~20KB | <1ms |
| Overlap buffers | ~10KB | <1ms |
| **Total overhead** | <7MB | <15ms |

**Target: <100ms for 10s chunk ✅ ACHIEVABLE**

---

**Bottom Line**: We enhance our existing streaming.py with Mamba's recurrent mode and MNE-LSL buffer patterns. No wheel reinvention, just smart integration of proven components.

Ready to pull the trigger on Phase 6.1? Just say the word and I'll start implementing StatefulBiMamba!

---

## 📦 REFERENCE REPOS TO CLONE (IRON-CLAD LIST)

### Critical - MUST HAVE:
```bash
# 1. Mamba SSM (OFFICIAL) - For recurrent/streaming mode
git clone https://github.com/state-spaces/mamba
# KEY FILES: mamba_ssm/ops/selective_scan_interface.py
#            mamba_ssm/modules/mamba_simple.py

# 2. MNE-LSL (OFFICIAL) - Real-time EEG streaming
git clone https://github.com/mne-tools/mne-lsl
# KEY FILES: mne_lsl/stream/stream_lsl.py
#            examples/30_real_time_evoked.py

# 3. NEDC Evaluation Tools (OFFICIAL) - Our benchmark target
git clone https://github.com/NeuroTechX/NEDC-Epilepsy-Benchmark
# KEY FILES: scoring/nedc_eval_eeg.py
```

### Important - SHOULD HAVE:
```bash
# 4. Pyannote-audio (OFFICIAL) - Sliding window & stitching
git clone https://github.com/pyannote/pyannote-audio
# KEY FILES: pyannote/audio/core/inference.py

# 5. TorchEEG (OFFICIAL) - EEG processing utilities
git clone https://github.com/torcheeg/torcheeg
# KEY FILES: torcheeg/io/eeg_io.py (streaming loaders)

# 6. Braindecode (OFFICIAL) - We already have this
git clone https://github.com/braindecode/braindecode
# KEY FILES: braindecode/datautil/windowers.py
```

### Reference Implementations:
```bash
# 7. EEG-Conformer - Stateful inference example
git clone https://github.com/eeyhsong/EEG-Conformer

# 8. Streaming Conformer - Cache-based inference
git clone https://github.com/k2-fsa/icefall
# KEY FILES: egs/librispeech/ASR/conformer_ctc3/streaming_decode.py

# 9. Vision Mamba - Efficient Mamba implementation
git clone https://github.com/hustvl/Vim
```

---

## 📄 PAPERS TO DOWNLOAD (ARXIV LINKS)

### Must-Read Core Papers:
```
1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
   https://arxiv.org/abs/2312.00752

2. Mamba-2: Transformers are SSMs
   https://arxiv.org/abs/2405.21060

3. Stateful Conformer with Cache-based Inference
   https://arxiv.org/abs/2312.17279
```

### Streaming & Real-Time Papers:
```
4. Real-time Sub-milliwatt Epilepsy Detection (2024)
   https://arxiv.org/abs/2410.16613

5. Continuous Batching for LLM Inference
   https://arxiv.org/abs/2401.08671

6. Memory-aware Dynamic Batching (2025)
   https://arxiv.org/abs/2503.05248
```

### EEG-Specific Papers:
```
7. EEGMamba: Bidirectional State Space Models for EEG
   (Check our literature/pdfs/EEG-BIMAMBA.pdf)

8. 1D-Vision Transformer for EEG (2025)
   https://www.sciencedirect.com/science/article/pii/S001048252400XXXXX

9. Unsupervised Transformers for Seizure Identification
   https://arxiv.org/abs/2301.03470
```

### Benchmarking Papers:
```
10. NEDC TUH EEG Seizure Corpus Benchmark
    https://isip.piconepress.com/publications/reports/2021/tuh_eeg/seizure_detection/

11. SzCORE: Standardized Seizure Detection Evaluation
    https://arxiv.org/abs/2310.09237
```

---

## 🔧 QUICK CLONE SCRIPT

```bash
#!/bin/bash
# Save as: clone_phase6_refs.sh

echo "Cloning Phase 6 reference repositories..."

# Create refs directory
mkdir -p phase6_refs
cd phase6_refs

# Clone critical repos
git clone --depth 1 https://github.com/state-spaces/mamba
git clone --depth 1 https://github.com/mne-tools/mne-lsl
git clone --depth 1 https://github.com/pyannote/pyannote-audio

# Clone if not already in reference_repos
if [ ! -d "../reference_repos/nedc-bench" ]; then
    git clone --depth 1 https://github.com/NeuroTechX/NEDC-Epilepsy-Benchmark
fi

echo "Done! Check phase6_refs/ directory"
```

---

## 📊 PRIORITY MATRIX

| Repository | Priority | Why We Need It |
|------------|----------|----------------|
| mamba | CRITICAL | Selective scan interface for stateful inference |
| mne-lsl | CRITICAL | Ring buffer & streaming patterns |
| NEDC tools | CRITICAL | Our evaluation benchmark |
| pyannote | HIGH | Sliding window stitching |
| torcheeg | MEDIUM | EEG-specific utilities |
| icefall | MEDIUM | Streaming conformer reference |

| Paper | Priority | Key Insight |
|-------|----------|-------------|
| Mamba-1 | CRITICAL | Recurrent mode implementation |
| Mamba-2 | HIGH | SSD algorithm optimizations |
| Stateful Conformer | HIGH | Cache management patterns |
| NEDC Benchmark | CRITICAL | Scoring protocol |
| Real-time SNN | MEDIUM | Performance targets |

---

**ACTION**: Clone the CRITICAL repos first, then papers 1, 2, 3, and 10. That's your iron-clad foundation.

---

## 🔎 CLONED REPOS EXAMINATION RESULTS

### ✅ **What We Now Have:**

1. **`/reference_repos/mamba/`** - OFFICIAL Mamba SSM ✅
   - Key file: `mamba_ssm/ops/selective_scan_interface.py`
   - Has `return_last_state=False` and `prev_state` parameters for stateful inference
   - Perfect for our streaming needs

2. **`/reference_repos/mne-lsl/`** - MNE Lab Streaming Layer ✅ [NEW]
   - Key file: `src/mne_lsl/stream/base.py` - Circular buffer implementation (line 81)
   - Key file: `src/mne_lsl/stream/stream_lsl.py` - LSL streaming
   - Examples: `examples/30_real_time_evoked_responses.py`
   - **PyPI installable**: `pip install mne-lsl`

3. **`/reference_repos/pyannote-audio/`** - Overlap-add stitching ✅ [NEW]
   - Key file: `pyannote/audio/core/inference.py`
   - Lines 562-644: Hamming window overlap-add aggregation
   - Warm-up windows for chunk boundaries
   - **PyPI installable**: `pip install pyannote.audio`

4. **`/reference_repos/nedc-bench/`** - NEDC evaluation tools ✅
   - `nedc_eeg_eval/v6.0.0/lib/` - All scoring scripts

### 📦 **PyPI Installation Strategy:**

```bash
# For Phase 6, we can install via pip:
pip install mne-lsl  # Real-time streaming
# pip install pyannote.audio  # Optional - we can just copy the pattern

# We already have in pyproject.toml:
# - mamba-ssm (GPU extra)
# - mne (base)
# - torch, numpy, etc.
```

### 🔑 **Key Patterns to Extract:**

**From MNE-LSL (circular buffer):**
```python
# From base.py line 81
# "Pull new samples in the internal circular buffer"
# They use a numpy array as ring buffer with head/tail tracking
```

**From Pyannote (overlap-add):**
```python
# From inference.py lines 562-644
# Hamming window: np.hamming(num_frames_per_chunk)
# Warm-up windows to exclude edges
# Overlap-add aggregation:
aggregated_output[start_frame : start_frame + num_frames] += (
    score * mask * hamming_window * warm_up_window
)
```

**From Mamba (stateful inference):**
```python
# From selective_scan_interface.py
def selective_scan_fn(..., return_last_state=False, prev_state=None):
    # Use prev_state for continuity between chunks
    # Return last_state for next chunk
```

### 📄 **Do We Need Papers?**

**VERDICT: Not critically needed** - The code tells us everything:

1. **Mamba paper** - Nice for theory but `selective_scan_interface.py` shows the implementation
2. **MNE-LSL docs** - The examples are more useful than papers
3. **Pyannote papers** - The overlap-add code is self-explanatory

**If you want ONE paper**: Read Mamba-1 Section 3.4 on recurrent mode (5 min read)

### 🚀 **Next Steps:**

1. **Add to pyproject.toml**:
   ```toml
   [project.optional-dependencies]
   streaming = ["mne-lsl>=1.3.0"]
   ```

2. **Study these specific files**:
   - `mamba_ssm/ops/selective_scan_interface.py` (lines 27-108)
   - `mne-lsl/src/mne_lsl/stream/base.py` (buffer logic)
   - `pyannote-audio/pyannote/audio/core/inference.py` (lines 562-644)

3. **Implementation approach**:
   - Use MNE-LSL's circular buffer pattern (don't import, just copy pattern)
   - Use Pyannote's overlap-add math (don't import, just copy formula)
   - Use Mamba's existing stateful interface (we already have mamba-ssm)

---

## ⚡ IRON-CLAD IMPLEMENTATION CHECKLIST

### **Day 1: Stateful Mamba Integration**
- [ ] Study `/reference_repos/mamba/mamba_ssm/ops/selective_scan_interface.py`
- [ ] Create `StatefulBiMamba2` wrapper in `models.py`
- [ ] Add state caching between chunks
- [ ] Test state persistence with synthetic data

### **Day 2: Streaming Enhancement**
- [ ] Add duration filtering to `StreamingPostProcessor`
- [ ] Implement `flush()` method
- [ ] Create `SeizureEvent` with confidence scoring
- [ ] Add event buffer management

### **Day 3: Boundary Management**
- [ ] Implement ring buffer (MNE-LSL pattern)
- [ ] Add overlap-add stitching (Pyannote formula)
- [ ] Test chunk boundary continuity
- [ ] Benchmark memory usage

### **Day 4: Integration Testing**
- [ ] Streaming vs offline equivalence test
- [ ] Latency benchmarks (<100ms target)
- [ ] NEDC scoring integration
- [ ] End-to-end EDF streaming test

### **Day 5: CLI & Documentation**
- [ ] Add `stream` command to CLI
- [ ] Create streaming examples
- [ ] Update Phase 6 documentation
- [ ] Final performance validation

### **Success Metrics:**
- ✅ Streaming output within ±1 sample of offline
- ✅ <100ms latency per 10s chunk
- ✅ <10MB additional memory overhead
- ✅ NEDC scores match offline processing
- ✅ All tests pass with `make q`

---

**FINAL VERDICT**: This plan is iron-clad. We have all reference code, clear implementation path, and specific success metrics. Ready to execute!