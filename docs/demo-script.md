# PhysicsLens: 3-minute demo video script

**Goal:** in under 3:00, show judges (1) a real measurement from a video, (2) that Meta SAM 3.1 and Meta Muse Spark do
real work at runtime, (3) that the app refuses to give a number it cannot check, (4) that it is online and open.

**Format:** screen recording (1920×1080), voice-over, English captions. No music needed; if any, royalty-free and quiet.

---

## Timeline

| Time | On screen | Voice-over (read at a calm pace) |
|---|---|---|
| **0:00–0:12** | Landing page hero (https://physicslens-lab.vercel.app): the ball falls, dots appear, the readout converges to g = 9.81 and turns green | "Can a phone video measure gravity? PhysicsLens tracks the ball, fits the physics, and lets an AI agent check the result before you see a number." |
| **0:12–0:25** | Scroll to *How it works*: the four steps Film, Track, Fit, Check | "You film an experiment with something of known length in view. The ball is tracked frame by frame, SciPy fits the motion, and a Meta Muse Spark agent checks the fit." |
| **0:25–0:55** | Click **Analyze a video** → the app. Upload `samples/commons/pendulum_clip2.mp4`. Set the start to **0.6 s**. **Scale:** type 9750 px/m (or click two ruler marks 1 cm apart, ~97.5 px). Tracker **SAM 3.1**, click **Ball**, box **180, 40 → 450, 189** around the bob. First run with the agent **off** | "Here is a 60-centimetre pendulum from Wikimedia Commons, filmed in front of a ruler. I mark the scale and draw one box around the bob. From that single box, Meta SAM 3.1 follows the bob through every frame." |
| **0:55–1:15** | The result card: **L = 0.590 m** (tape: 0.60 m, −1.6%), T = 1.541 s, R² 0.994, and the amber flag "residuals are structured" | "SciPy fits the swing: 0.590 metres, against 0.60 on the tape. But the checks flag it: the leftovers of the fit are not random." |
| **1:15–1:35** | Run again with the **Muse agent on** (SAM is cached, so it is instant and free). The agent timeline: it describes the track, fits, checks, tries trimmed windows, and ends with **no measurement** and its reason | "So I ask Meta Muse Spark to check. It describes the track, fits it, tests trimmed windows, and refuses to certify the number, and says why. A tool that knows when not to trust itself." |
| **1:35–1:50** | Video overlay playing: cyan tracked dots, violet fitted curve. The charts and residuals below | "Cyan is what was measured. Violet is the model. Every number on screen comes from a deterministic fit." |
| **1:50–2:15** | Upload *Ballistische Kurve* (CC BY-SA, Rhetos), whose white strobe dots are drawn over the throw. Analyze with the motion tracker. The result: **no measurement**, and the reason | "What if the video can't be trusted? This clip has dots drawn over the flight. A tracker follows the wrong things, and most apps would still print a number. PhysicsLens says: no measurement, and why. No agent was even paid to find that out." |
| **2:15–2:35** | **History**: the saved runs; open one, it replays exactly. Then the Home page *Pendulum lab*: drag the angle to 30° and show the large-angle note | "Every run is saved and replays exactly as it happened. And the home page explains the physics: at large angles the simple formula is off by percent, which is why PhysicsLens fits the swing instead." |
| **2:35–2:52** | *Results* table, then *Built with*: Meta SAM 3.1 · Meta Muse Spark · SciPy | "On real clips it lands within one to two percent. Meta SAM 3.1 tracks, Meta Muse Spark checks, SciPy computes. Nothing else runs at runtime." |
| **2:52–3:00** | Landing URL and the GitHub link, large | "PhysicsLens is open source and online. Try it at physicslens-lab dot vercel dot app." |

---

## Before recording

1. **Dry run the whole flow once**, the day before, with the same clips and settings, so nothing surprises you on camera. Ask Claude to do the dry run with you and write down the exact scale clicks and box for the pendulum.
2. Start the app online: `CLOUDFLARED=~/.local/bin/cloudflared deploy/tunnel.sh --publish`, then open https://physicslens-lab.vercel.app/app and check `/api/config` answers.
3. Check the budget: `PHYSICSLENS_MAX_USD` in `.env` (10). The pendulum run with SAM and the agent costs a few cents; repeated takes are cached and cost $0.
4. Browser: hide bookmarks (Cmd+Shift+B), close other tabs, zoom 100%, dark system theme, notifications off (Focus mode).
5. Recorder: QuickTime (File → New Screen Recording) or OBS, full screen, 1080p, microphone on. Record each row of the timeline as its own take; cut them together afterwards.
6. Use only the clips listed below. **Never show the OSP Tracker samples (BallDrop): they have no license.**

## Clips and credits (put these in the video's end card and description)

- Pendulum: *Fadenpendelgeschwindigkeit (2 grad)*, Mathe-AC Lernwerkstatt Aachen (Rhetos), Wikimedia Commons, CC BY-SA 4.0. File: `samples/commons/pendulum_clip2.mp4` (a crop of it).
- Overlaid throw: *Ballistische Kurve*, Mathe-AC Lernwerkstatt Aachen (Rhetos), Wikimedia Commons, CC BY-SA 4.0. File: `samples/open/Ballistische_Kurve.webm`.
- PhysicsLens is a hackathon project, not endorsed by Meta.

## Editing

- Cut waiting time (uploads, the agent thinking) with jump cuts; keep the agent's steps appearing, they are the story.
- Captions on, burned in or uploaded as a file.
- Export 1080p MP4, under 3:00. Upload to YouTube as **Unlisted** (or Public) and paste the link into Devpost and the README.

## If you have only 30 minutes

Record three takes: the hero (0:00–0:25), the pendulum run with the agent (0:25–1:50), the "no measurement" clip (1:50–2:15). Put the voice-over over them and end on the URL.
