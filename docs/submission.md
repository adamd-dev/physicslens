# Submission: Devpost draft and readiness

Deadline: **2026-11-04, 9:00 PM ET**. Code freeze: end of **2026-11-01**. Judging (indie track): creativity, technical
ambition, and whether it works.

## Readiness checklist

| # | Item | Status | When |
|---|---|---|---|
| 1 | Repo public, MIT, homepage and topics | done | |
| 2 | CI green (ruff, mypy, 342 pytest, 12 vitest) | done | |
| 3 | Landing page live: https://physicslens-lab.vercel.app | done | |
| 4 | App live at `/app` (tunnel from the Mac, `PHYSICSLENS_MAX_USD=10`) | done, check it again | 11-03, 11-04 |
| 5 | Code freeze: no commits to `main` except docs and the video link | to do | end of 11-01 |
| 6 | Dry run of `docs/demo-script.md` with the same clips and settings | to do | 11-02 |
| 7 | Record, edit, upload the video (Unlisted, under 3:00); check it plays logged out | to do | 11-02 – 11-03 |
| 8 | Paste the video link into the README (top) and Devpost | to do | after 7 |
| 9 | Fill in Devpost from the draft below; pick the theme | to do | 11-03 |
| 10 | Submit, then open the public submission page and check every link | to do | 11-04, before 6 PM ET (3 h margin) |
| 11 | Keep the Mac awake and on power during judging (11-04 – 11-08); `/app` is down when it sleeps | to do | 11-04 – 11-08 |

If the app is down while judges look, the video and the README's local setup are the fallback. Say so in the Devpost
"Try it out" field.

## Devpost draft

**Name:** PhysicsLens

**Tagline:** Measure gravity and a pendulum's length from a phone video, with an AI agent that refuses numbers it
cannot check.

**Theme:** Build for Your World. Physics lab work, as a student, is where the stopwatch errors come from. Change this
if "Push the Frontier" fits your entry better.

**Links:**

- Live: https://physicslens-lab.vercel.app (app at `/app`)
- Code: https://github.com/adamd-dev/physicslens
- Video: *(paste after upload)*

### Inspiration

School physics labs measure g with a stopwatch and a falling ball. Human reaction time is about 0.2 s, on a fall that
lasts less than a second. Every phone films at 30 to 240 frames a second. The measurement is already in the video; it
only needs to be read out, and checked.

### What it does

1. You upload a video of a ball drop, a throw or a pendulum, with something of known length in view.
2. PhysicsLens tracks the ball in every frame, fits the physics with SciPy, and reports:
   - g for a drop;
   - the landing point for a throw;
   - the length for a pendulum.
   Each value comes with an uncertainty.
3. A Meta Muse Spark agent checks the fit before you see a number. When the fit cannot be trusted, the result is **no
   measurement**, with the reason. Two examples:
   - a ball that hardly moves;
   - a video with marks drawn over the flight.
4. Every run is saved to History and replays exactly as it was seen.

### How we built it

- **Meta SAM 3.1** (Meta Model API) segments and follows the ball from one box, when the simple trackers lose it.
  Without a box, a motion tracker finds the moving blob whose path falls, flies or swings.
- **Meta Muse Spark** (`muse-spark-1.3`) runs as a tool-calling agent:
  - it describes the track;
  - it calls the fit and check tools;
  - it tries trimmed windows;
  - it accepts or rejects the fit.
  The numbers always come from SciPy, never from the model's text.
- **The backend:** Python with OpenCV and SciPy, and a FastAPI server that streams every analysis step to the browser.
- **The frontend:** a React + Vite web app, and a pre-rendered landing page with an interactive fit and a pendulum
  simulator.
- **The tests:** 342 Python tests on synthetic clips, with no API calls, plus web tests. They all run in CI.
- **Spending:** capped by a budget. Model answers are cached, so a repeated run costs $0.

### Challenges we ran into

- **Video timestamps lie.** Container clocks drift, and slowed edits repeat each frame several times. PhysicsLens
  repairs the clock and drops the copies, so a 5× slow-motion edit still gives the right g.
- **Trackers lock onto the wrong object.** A gate rejects fits that fit almost anything, and asks for a box instead.
- **Openly licensed physics videos are rare,** and most have overlays, cuts or perspective. That is why the refusal
  path matters as much as the measurement.
- **Hosting without a payment card:** the app runs from a laptop behind a Cloudflare tunnel, with a public mode:
  - upload caps;
  - per-visitor rate limits;
  - runs that stop when the visitor leaves;
  - a fallback to the plain fit when the budget is used up.

### Accomplishments that we're proud of

- **Within 1–2% on real clips:**
  - g 9.73 m/s² against 9.81;
  - a pendulum length of 0.590 m against 0.60 m on the tape.
- **An agent that says "no measurement"** instead of inventing one.

### What we learned

A measurement tool is only as good as its refusals. Letting the model choose and check, but never compute, kept every
number reproducible.

### What's next

- Phone capture in the browser, with a live framing guide.
- More experiments: inclined planes, collisions.
- A teacher view that compares a class's runs.

### Built with

Meta SAM 3.1 · Meta Muse Spark · Python · SciPy · OpenCV · FastAPI · React · TypeScript · Vite · Tailwind CSS ·
Vercel · Cloudflare Tunnel

### Credits

- Pendulum and throw clips: Mathe-AC Lernwerkstatt Aachen (Rhetos), Wikimedia Commons, CC BY-SA 4.0.
- A hackathon project, not endorsed by Meta.
