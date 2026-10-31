# PhysicsLens: 3-minute demo video script

**Judged on** (indie track): creativity, technical ambition, and whether it works. Every scene below shows one of the
three. The official page gives no video length; keep it **under 3:00** anyway, as most Devpost hackathons require.

**Story in one line:** a phone video becomes a physics measurement, and the app refuses to give a number it cannot check.

**Format:** screen recording at 1920×1080, your voice, burned-in captions. Show the product in the first 10 seconds;
no title card, no slides.

---

## How winning demos are cut (watch these first)

The winners of Meta's first LlamaCon Hackathon (May 2025), with their demo videos:

| Project | Prize | Video |
|---|---|---|
| OrgLens | 1st | https://www.youtube.com/watch?v=8OmMnODOyLw |
| Compliance Wizards | 2nd | https://youtu.be/JgZQ6XJXBhE |
| Llama CCTV Operator (video understanding, closest to PhysicsLens) | 3rd | https://youtu.be/4YaZe4Gbt1g |
| Geo-ML | Best Llama API use | https://youtu.be/U1fDn1C3DJA |

Source: https://ai.meta.com/blog/llamacon-hackathon/

While watching, note:

1. How many seconds pass before the product is on screen.
2. Where they say which Meta model does what.
3. How they show a result the viewer can check.

Copy these three things. Common advice from hackathon guides agrees:

- show the product at once;
- spend most of the time on the live demo;
- end with one technical highlight and a short close.

## Timeline

| Time | On screen | Voice-over | Edit |
|---|---|---|---|
| **0:00–0:08** | The app, mid-run: the pendulum video with cyan tracked dots and the violet fitted curve on top | "This is a pendulum filmed in front of a ruler. PhysicsLens just measured its length from the video alone." | Start on motion. No logo, no intro. |
| **0:08–0:20** | Landing hero (https://physicslens-lab.vercel.app): the ball falls, the readout converges to g = 9.81 | "Physics labs time a falling ball with a stopwatch, and get it wrong by tenths of a second. A phone films 30 to 240 frames a second. PhysicsLens turns that video into g, a pendulum's length or where a throw lands." | Title text over the hero: **PhysicsLens**, then the one-line story. |
| **0:20–0:50** | **Analyze a video** → the app. Upload `samples/commons/pendulum_clip2.mp4`. Start **0.6 s**. **Scale:** click the left and right edges of the bob, type 0.028 (m). Tracker **SAM 3.1**, click **Ball**, box **180, 40 → 450, 189** around the bob. Agent **off**. Analyze | "I mark a known length, and draw one box around the bob. From that single box, Meta SAM 3.1 segments and follows the bob through every frame." | Speed up the upload 4×. Zoom in on the box being drawn. |
| **0:50–1:10** | The result card: **L = 0.590 m** (tape: 0.60 m, −1.6%), T = 1.541 s, R² 0.994, and the amber flag "residuals are structured" | "SciPy fits a damped swing: 0.590 metres. The tape says 0.60. But the checks flag it: what is left after the fit is not random noise." | Zoom on the number, then on the amber flag. Caption: **−1.6% vs tape**. |
| **1:10–1:40** | Run again with the **Muse agent on** (SAM is cached: instant, $0). The agent timeline: it describes the track, fits, checks, tries trimmed windows, ends with **no measurement** and its reason | "So I ask a Meta Muse Spark agent to check it. It calls the fit and check tools itself, tries shorter windows, and then refuses to certify the number, and says why. The numbers always come from SciPy, never from the model's text." | This is the key scene: let each agent step appear. Cut only the waiting. |
| **1:40–2:05** | Upload *Ballistische Kurve* (CC BY-SA, Rhetos), with white strobe dots drawn over the throw. Start **15**, end **20.9** s, motion tracker, **Scale** on the metre rule at the left (≈ y 188 → 905, 1 m). Analyze. Result: **no measurement**, and the reason | "What if the video itself is misleading? These dots were drawn over the flight. Most tools would still print a number. PhysicsLens says: no measurement, and why. That cost nothing: no model was called." | Split screen if you can: the clip left, the refusal right. |
| **2:05–2:25** | **History**: open a saved run, it replays exactly. Then the Home page *Pendulum lab*: drag the angle to 30° and show the large-angle note | "Every run is saved and replays exactly as it happened. And the home page explains the physics: at large angles the textbook formula is off by percent, which is why PhysicsLens fits the whole swing." | Quick cuts, about 5 seconds each. |
| **2:25–2:45** | *Built with* strip: Meta SAM 3.1 · Meta Muse Spark · SciPy. Then the *Results* table | "Meta SAM 3.1 tracks. Meta Muse Spark checks. SciPy computes. On real clips it lands within one to two percent, and when it cannot be sure, it says so." | Text on screen: the three roles, one per line. |
| **2:45–3:00** | Landing URL and the GitHub link, large | "PhysicsLens is open source and online. Film something falling, and try it at physicslens-lab dot vercel dot app." | End card: URL, GitHub, clip credits. |

## Editing it the same way (free tools)

- **Recorder with auto-zoom on clicks:** Cap, free and open source: https://cap.so.
  - Or QuickTime (File → New Screen Recording) and add zooms in the editor.
- **Editor:** CapCut desktop, free, with auto-captions, speed ramps and zoom keyframes: https://www.capcut.com.
  - For more control: DaVinci Resolve, free: https://www.blackmagicdesign.com/products/davinciresolve.
- **Steps:**
  1. Record each row of the timeline as its own take.
  2. Put them on the timeline in order.
  3. Cut every wait: uploads, the agent thinking.
  4. Speed the dull parts 2–4×, and add a zoom keyframe on each number.
  5. Run auto-captions, then fix the physics words (SAM, Muse Spark, SciPy, g).
- **Audio:** record the voice-over separately after the cut, in a quiet room, with the phone mic 20 cm away. No music, or quiet royalty-free music under −25 dB.
- **Export:** 1080p MP4 at 30 fps, under 3:00. Upload to YouTube as **Unlisted** (or Public). Check it plays logged out.

## Before recording

1. **Dry run the whole flow once**, the day before, with the same clips and settings.
2. Check the app is online:
   - open https://physicslens-lab.vercel.app/app; if it does not load, run `CLOUDFLARED=~/.local/bin/cloudflared deploy/tunnel.sh --publish`.
3. The budget is `PHYSICSLENS_MAX_USD=10` in `.env`. The pendulum run with SAM and the agent costs a few cents, and repeated takes are cached at $0.
4. Browser:
   - hide bookmarks (Cmd+Shift+B);
   - close other tabs;
   - zoom 100%;
   - dark system theme;
   - notifications off (Focus mode).
5. Use only the clips listed below. **Never show the OSP Tracker samples (BallDrop): they have no license.**

## Clips and credits (end card and video description)

- Pendulum: *Fadenpendelgeschwindigkeit (2 grad)*, Mathe-AC Lernwerkstatt Aachen (Rhetos), Wikimedia Commons, CC BY-SA 4.0.
  - File: `samples/commons/pendulum_clip2.mp4` (a crop of it).
  - https://commons.wikimedia.org/wiki/File:Fadenpendelgeschwindigkeit_(2_grad).webm
- Overlaid throw: *Ballistische Kurve*, Mathe-AC Lernwerkstatt Aachen (Rhetos), Wikimedia Commons, CC BY-SA 4.0.
  - File: `samples/open/Ballistische_Kurve.webm`.
  - https://commons.wikimedia.org/wiki/File:Ballistische_Kurve.webm
- PhysicsLens is a hackathon project, not endorsed by Meta.

## If you have only 30 minutes

Record three takes:

1. the pendulum run with the agent (0:20–1:40);
2. the "no measurement" clip (1:40–2:05);
3. the landing page (0:08–0:20 and 2:45–3:00).

Open on the first frame of the pendulum take, add the voice-over, and end on the URL.
