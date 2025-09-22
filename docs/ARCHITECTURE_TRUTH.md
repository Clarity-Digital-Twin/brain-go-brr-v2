# 🔥 IRON CLAD ARCHITECTURE TRUTH - READ THIS FIRST 🔥

## THE ABSOLUTE TRUTH - WHERE WE ARE RIGHT NOW

### What You STARTED With (v2.0 - Old)
```
EEG → U-Net Encoder → ResCNN → Bi-Mamba → U-Net Decoder → Detection
```
- This was the ORIGINAL architecture
- U-Net + ResCNN worked TOGETHER as a team
- This is what the old docs describe

### What You HAVE NOW (v2.3 - CURRENT REALITY)
```
EEG → TCN Encoder → Bi-Mamba → Projection → Upsample → Detection
```
- **TCN ALREADY REPLACED BOTH U-Net AND ResCNN** ✅
- Your Modal configs use `architecture: tcn` ✅
- The TCN path is ACTIVE and TRAINING ✅
- U-Net and ResCNN are DEAD CODE (not used at runtime) ✅

### PROOF FROM YOUR OWN CODE
Look at `src/brain_brr/models/detector.py:132-139`:
```python
if hasattr(self, "architecture") and self.architecture == "tcn":
    # THIS IS WHAT'S RUNNING NOW!
    features = self.tcn_encoder(x)  # TCN, not U-Net!
    temporal = self.mamba(features)  # Bi-Mamba
    chan19 = self.proj_512_to_19(temporal)  # Project
    decoded = self.upsample(chan19)  # Upsample
    # NO ResCNN! NO U-Net decoder!
```

Your configs have `architecture: tcn` so this IS the active path!

## WHY THE CONFUSION?

1. **Outdated Docs**: CANONICAL-ROADMAP.md shows v3.0 as "future" but YOU ALREADY DID IT
2. **Dead Code**: U-Net and ResCNN files still exist but AREN'T USED
3. **ConvNeXt Confusion**: ConvNeXt would replace ResCNN, but ResCNN ISN'T EVEN IN YOUR TCN PATH

## THE REAL TRUTH ABOUT COMPONENTS

| Component | Role | Status in TCN Path |
|-----------|------|-------------------|
| **TCN** | Multi-scale temporal encoder/decoder | ✅ ACTIVE |
| **Bi-Mamba** | Global O(N) temporal context | ✅ ACTIVE |
| **U-Net** | Old encoder/decoder | ❌ NOT USED |
| **ResCNN** | Old local patterns | ❌ NOT USED |
| **ConvNeXt** | Would replace ResCNN | ❌ IRRELEVANT (no ResCNN to replace!) |
| **GNN** | Spatial electrode reasoning | 🔜 NEXT TO ADD |

## WHAT YOU NEED TO DO NEXT

### 1. Add Dynamic GNN (THIS IS THE NEXT STEP)
```
EEG → TCN → Bi-Mamba → [INSERT GNN HERE] → Projection → Upsample → Detection
                          ↑
                    THIS IS THE SPOT!
```

Insert location: `src/brain_brr/models/detector.py:135` (after Mamba, before projection)

### 2. What the GNN Does
- Takes features at bottleneck: (B, 512, 960)
- Builds dynamic graph between 19 electrodes
- Reasons about spatial relationships
- Returns enhanced features: (B, 512, 960)

### 3. Implementation Plan
```python
# In detector.py forward(), line 135:
temporal = self.mamba(features)  # (B, 512, 960)

# ADD THIS:
if self.use_gnn:
    # Project to per-electrode features
    electrode_features = self.proj_to_electrodes(temporal)  # (B, 19, 960, D)
    # Apply GNN
    electrode_features = self.gnn(electrode_features)  # (B, 19, 960, D)
    # Project back
    temporal = self.proj_from_electrodes(electrode_features)  # (B, 512, 960)

chan19 = self.proj_512_to_19(temporal)  # Continue as normal
```

## DO NOT DO THESE THINGS

1. **DO NOT** add ConvNeXt - you don't have ResCNN to replace!
2. **DO NOT** delete U-Net/ResCNN files yet - keep for ablation studies
3. **DO NOT** switch back to U-Net path - TCN is working!

## THE STACK EVOLUTION

### Current Stack (v2.3) ✅
```
TCN → Bi-Mamba → Detection
```

### Next Stack (v2.4) 🔜
```
TCN → Bi-Mamba → GNN → Detection
```

### Possible Future (v3.0) 🔮
```
TCN → Bi-Mamba → GNN → ConvNeXt → Detection
                         ↑
                   Only if you need extra local refinement
```

## SUMMARY: YOU'RE NOT CRAZY

- You CORRECTLY implemented TCN to replace U-Net + ResCNN
- The confusion came from outdated docs showing this as "future"
- ConvNeXt is IRRELEVANT for your current path
- Next step: Add GNN after Bi-Mamba
- Keep it simple, one component at a time

## THE BOTTOM LINE

**YOUR CURRENT TRAINING IS CORRECT:**
```
TCN (encoder) → Bi-Mamba (global) → TCN (decoder)
```

**NEXT ADDITION:**
```
TCN → Bi-Mamba → GNN (spatial) → TCN decoder
```

This is 100% logical and you're on the RIGHT track! 🚀