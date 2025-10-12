# Streaming Inference (Planned)

Summary
- Use Mamba’s recurrent mode for stateful streaming with carried hidden state between chunks.
- Reuse HysteresisState/StreamingPostProcessor; add duration filtering and flush semantics.
- Build around MNE‑LSL/FieldTrip style buffering.

Status
- Design sketched; not implemented in training codepaths yet.
- Will add a CLI once the streaming path is validated.

Pointers
- Post‑processing: `docs/04-model/postprocess.md`
- V3 overview (recurrent applicability): `docs/04-model/v3-architecture.md`
