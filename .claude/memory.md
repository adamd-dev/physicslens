# PhysicsLens project memory

Durable decisions and hard-won facts that the code and git history do not explain.

## Constraints
- Meta Global AI Developer Hackathon, solo, Indie track. Feature freeze 2026-11-01; submission 2026-11-04 9:00 PM ET.
- Stage 1 is pass/fail on real runtime use of Meta models. The app must never call non-Meta models at runtime; every model call goes through `physicslens/llm/base.py`.
- API key lives only in `.env` as `MODEL_API_KEY` (gitignored). Never paste it in chat, code, or tests.

## External API facts (dev.meta.ai docs, checked 2026-10-27)
- Meta Model API: base `https://api.meta.ai/v1`, Bearer `MODEL_API_KEY`, OpenAI-compatible, tool calling supported. Text model `muse-spark-1.3`.
- Model id `meta-models/muse-spark-open-1026` from early notes was not found anywhere; docs show `meta-models/muse-spark-open`. Keep the model id configurable.
- Confirmed by real runs 2026-10-28: Muse sends tool `arguments` as a JSON string, often several tool calls per turn, and empty text content on the final turn. A run costs about $0.01 to $0.03 (4 to 12 steps).
- python.org Python 3.13 on macOS ships without root certificates (`CERTIFICATE_VERIFY_FAILED`). Fixed by running `/Applications/Python 3.13/Install Certificates.command`.
- SAM 3.1: model `sam-3.1` via Responses streaming; text prompt (e.g. "ball") or box/point tokens, not both. One text line per frame with boxes in source pixels and base85 masks. Frames can be missing or out of order; object ids are not dense. $0.20 per 1000 frames, 50-frame minimum per request. Tested 2026-10-28: a text prompt ("ball", "pendulum bob") found nothing on the sample clips; a box prompt on frame 0 tracks well (BallDrop g 9.65, Commons pendulum L 0.590). The box is the main mode. `usage.video_frames_processed` is the billed count (minimum 50), not the decoded count.

## Real-video lessons
- Samsung slow motion (A37: 720p 240 fps) plays back at 30 fps: pass `--capture-fps 240` or g comes out 64x too small.
- OSP Tracker sample clips are interlaced (BallDrop): split fields for 2x samples; wrong field order is obvious from roughness. BallTossOut is progressive: splitting it duplicates points.
- A white metre stick defeats the color tracker; the motion tracker removes it via median background, and its tick marks flicker under compression, so pixels moving in >30% of frames are masked.
- Reference results: BallDrop g = 9.728 ± 0.021 (-0.8%, all checks pass); BallTossOut g = 10.01 (+2.1%, U-shaped x residuals, likely perspective); Pendulum L = 1.005 m vs 1 m (one cycle only, flagged).
- Tracker sample videos have no stated license: keep them out of git and out of the demo.
- Third-party clip that works: Wikimedia Commons "Fadenpendelgeschwindigkeit (2 grad).webm" by Mathe-AC Lernwerkstatt Aachen, CC BY 4.0 (credit required; commercial use allowed). It shows a 60 cm pendulum in front of a cm ruler (about 197 px/cm at 1080p).
  - Only 68.3 to 71.8 s is usable real-time footage. Elsewhere there are freezes, montage cuts and slow motion.
  - Its 24 fps footage sits in a 25 fps container with one frame duplicated per second, so residuals are structured.
  - Cut with: rows 600 to 980, half scale, start at 67.6 s, length 4.8 s. Run with `--tracker color --hsv-upper 180,255,75 --min-area 3000 --px-per-m 9850 --start 0.6`. Result: L = 0.590 m (−1.7%), T = 1.541 s (the author measured 1.54).
- Direct Measurement Videos (SERC) now redirect to Pivot Interactives (paid) and are no longer usable. Its "Impaktecholot" drop is filmed along the fall axis, so it cannot be tracked.

## Agent behaviour lessons
- Window shopping: on BallTossOut, Muse fitted 12 time windows until one passed the runs test (12/21 points, g = 9.845, "ok"). A shorter window weakens the test, so the curvature was hidden, not fixed.
- Fix in `check_plausibility`: flag windows with < 80% of the points, and flag any window when the full track fails the same model. Also flag free_fall when x moves > 0.1 m. `accept_flagged` needs a reason of at least 20 characters.
- The runs test pools x and y residuals, so curvature in x alone gets diluted. A test fixture needs curvature on both axes to reproduce the real failure.
- After the fix, Muse reports the full-track TossOut fit as flagged with an honest reason (perspective). Expect exit 1 on TossOut and Pendulum: that is the correct outcome.

## Environment
- macOS 12 Intel + Python 3.13: opencv-python-headless >= 4.11 has no wheel and builds from source for 1h+. Pin `<4.11`.
- `~/.npm` is root-owned; use `npm --cache <tmpdir>` instead of sudo.
