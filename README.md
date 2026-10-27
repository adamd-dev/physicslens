# PhysicsLens

Measure real physics experiments from a phone video.

Film a ball drop, a pendulum, or a ball rolling off a table. PhysicsLens tracks
the object with SAM 3.1, converts pixels to metres, and an agent built on
Muse Spark picks a physics model, fits it, checks its own result, and predicts
what happens next.

Built for the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026).
All code in this repository was written during the hackathon.

## Status

- [x] Physics models and fitting (free fall, projectile, damped oscillation, FFT)
- [x] Pixel-to-metre calibration
- [x] Video ingest with real frame timestamps
- [x] Color-threshold fallback tracker
- [ ] SAM 3.1 tracker
- [ ] Agent loop (Muse Spark / Muse Spark Open)
- [ ] Streamlit UI and voice questions (Muse Voice Transcribe)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```
