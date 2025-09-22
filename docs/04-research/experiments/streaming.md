# Phase 6 — Streaming Inference (Future Work)

Summary
- Use Mamba’s recurrent mode for stateful streaming with carried hidden state between chunks.
- Reuse HysteresisState/StreamingPostProcessor; add duration filtering and flush semantics.
- Build around MNE‑LSL/FieldTrip style buffering; don’t reinvent.

Prior brainstorming lives in `docs/phases/BRAINSTORMING_PHASE6.md` (now moved here).

