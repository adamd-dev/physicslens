# PhysicsLens project memory

Durable decisions and hard-won facts that the code and git history do not explain.

## Constraints
- Meta Global AI Developer Hackathon, solo, Indie track. Feature freeze 2026-11-01; submission 2026-11-04 9:00 PM ET.
- Stage 1 is pass/fail on real runtime use of Meta models. The app must never call non-Meta models at runtime; every model call goes through `physicslens/llm/base.py`.
- API key lives only in `.env` as `MODEL_API_KEY` (gitignored). Never paste it in chat, code, or tests.

## External API facts (dev.meta.ai docs, checked 2026-10-27)
- Meta Model API: base `https://api.meta.ai/v1`, Bearer `MODEL_API_KEY`, OpenAI-compatible, tool calling supported. Text model `muse-spark-1.3`.
- Model id `meta-models/muse-spark-open-1026` from early notes was not found anywhere; docs show `meta-models/muse-spark-open`. Keep the model id configurable.
- SAM 3.1: model `sam-3.1` via Responses streaming; text prompt (e.g. "ball") or box/point tokens, not both. One text line per frame with boxes in source pixels and base85 masks. Frames can be missing or out of order; object ids are not dense. $0.20 per 1000 frames, 50-frame minimum per request.

## Real-video lessons
- Samsung slow motion (A37: 720p 240 fps) plays back at 30 fps: pass `--capture-fps 240` or g comes out 64x too small.
- OSP Tracker sample clips are interlaced (BallDrop): split fields for 2x samples; wrong field order is obvious from roughness. BallTossOut is progressive: splitting it duplicates points.
- A white metre stick defeats the color tracker; the motion tracker removes it via median background, and its tick marks flicker under compression, so pixels moving in >30% of frames are masked.
- Reference results: BallDrop g = 9.728 ± 0.021 (-0.8%, all checks pass); BallTossOut g = 10.01 (+2.1%, U-shaped x residuals, likely perspective); Pendulum L = 1.005 m vs 1 m (one cycle only, flagged).
- Tracker sample videos have no stated license: keep them out of git and out of the demo.

## Environment
- macOS 12 Intel + Python 3.13: opencv-python-headless >= 4.11 has no wheel and builds from source for 1h+. Pin `<4.11`.
- `~/.npm` is root-owned; use `npm --cache <tmpdir>` instead of sudo.
