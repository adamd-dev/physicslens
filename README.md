# PhysicsLens

**Measure gravity with your phone.** Film a ball drop, a throw or a pendulum. PhysicsLens tracks the ball, fits the
physics with SciPy, and a Meta Muse Spark agent checks the fit before you see a number.

- **Landing page:** https://physicslens-lab.vercel.app
- **Live app:** https://physicslens-lab.vercel.app/app (runs from the author's machine during the judging period; see
  [Run it as a public site](#run-it-as-a-public-site)). If it does not answer, the machine is offline: the demo video
  shows the full flow, and [Run it on your computer](#run-it-on-your-computer) takes five commands.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026). All code here was written during the hackathon.

## How it works

1. **Film:** a phone held still, side-on, with something of known length (a ruler, a sheet of A4) in the ball's plane.
2. **Track:** a motion or colour tracker finds the ball in every frame. Without a box, the ball is found from how it
   moves (`track/autodetect.py`: the moving blob whose path falls, flies or swings). **Meta SAM 3.1** segments and
   follows the ball from one box when the simple trackers lose it.
3. **Fit:** SciPy fits free fall, a projectile or a damped swing, and reports g, the landing point or the pendulum
   length with an uncertainty from the fit and from reading the scale. Fits a hand could have made (nothing falls,
   flies or swings) give **no measurement**, not a number.
4. **Check:** a **Meta Muse Spark** agent calls the fit and check tools, rejects fits that fail the checks, and explains
   what it kept. The numbers always come from SciPy, never from the model's text.

Every analysis is saved to **History** with its full event stream, so a past run reopens exactly as it was seen: video
overlay, charts, result card and the agent's steps.

## Meta models used

| Model | Where | What for |
|---|---|---|
| Muse Spark (`muse-spark-1.3`, Meta Model API) | `physicslens/llm/meta_client.py`, `physicslens/agent/` | The checking agent: tool calls to fit and check, final answer |
| SAM 3.1 (`sam-3.1`, Meta Model API) | `physicslens/llm/sam_client.py`, `physicslens/track/sam.py` | Ball segmentation and tracking from one box |

No other model runs at runtime.

## Results

Two clips, not a benchmark. ± is the fit (and the scale reading where stated); perspective is not included.

| Clip | Quantity | Measured | Reference | Error |
|---|---|---|---|---|
| BallDrop (OSP Tracker sample) · motion tracker | g | 9.728 m/s² | 9.81 m/s² | −0.8% |
| BallDrop · SAM 3.1 box | g | 9.653 m/s² | 9.81 m/s² | −1.6% |
| Wikimedia Commons pendulum · SAM 3.1 + fit (flagged, so no ± is given) | length L | 0.590 m | 0.60 m (tape) | −1.6% |

The main remaining error on BallDrop is depth: the ball falls in front of the ruler's plane.

## Run it on your computer

Needs Python 3.11+ and Node 20.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[ui,dev]"
(cd web && npm ci && npm run build)
cp .env.example .env          # add MODEL_API_KEY for SAM 3.1 and the Muse agent
.venv/bin/physicslens-ui      # http://127.0.0.1:8000
```

Without a key, the motion and colour trackers and the SciPy fits still work. Spending stops at `PHYSICSLENS_MAX_USD`
(default $5), counted in `costs.csv`; API answers are cached under `cache/`.

**Command line:**

```bash
.venv/bin/python cli.py video.mp4 --px-per-m 400                    # auto-detect the ball, fit, report
.venv/bin/python cli.py video.mp4 --px-per-m 400 --tracker sam --sam-box 194,40,209,55 --agent
```

Exit codes: 0 measured and plausible, 1 measured but flagged, 2 tracking failed, 3 setup or API failure, 4 no
measurement. Slow-motion clips need their real frame rate: `--capture-fps 240`.

## Run it as a public site

`deploy/tunnel.sh --publish` builds the app, starts it in public mode and opens a Cloudflare quick tunnel (no account),
then points the landing page's `/app` link at it. Public mode shows a notice that uploads are public, shares one
History, caps uploads, keeps the newest videos, limits each visitor per hour, lets only the owner delete runs
(`PHYSICSLENS_ADMIN_TOKEN`), stops a run when its visitor leaves, and falls back to the plain fit when the budget is
used up. The link works while the machine and the script run.

`Dockerfile`, `render.yaml` and `deploy/cloudrun.sh` deploy the same app to a container host (Hugging Face, Render,
Google Cloud Run); they are kept for that, but unused, since each of those hosts asks for a payment card.

The landing page (`web/landing.html`) is pre-rendered at build time and deployed on Vercel from `web/`.

## Development

```bash
.venv/bin/pytest -q                       # Python tests (synthetic clips, no API calls)
.venv/bin/ruff check . && .venv/bin/mypy physicslens cli.py
(cd web && npm test && npm run build)     # Home page maths, type check, build
```

| Path | What |
|---|---|
| `physicslens/ingest.py` | Video loading with real timestamps, windows, frame and width limits |
| `physicslens/track/` | Motion, colour and SAM 3.1 trackers; ball auto-detect |
| `physicslens/physics/` | Models and least-squares fits |
| `physicslens/agent/` | The Muse Spark agent loop and its tools |
| `physicslens/pipeline.py` | Tracking, rest trimming, the wrong-object gate |
| `physicslens/server.py` | FastAPI: uploads, the analysis event stream, History |
| `web/` | React app (Home, Analyze, History) and the landing page |

## Credits

- Pendulum clip: [“Fadenpendelgeschwindigkeit (2 grad)”](https://commons.wikimedia.org/wiki/File:Fadenpendelgeschwindigkeit_(2_grad).webm) by Mathe-AC Lernwerkstatt Aachen (Rhetos), Wikimedia Commons, CC BY-SA 4.0.
- Overlaid throw: [“Ballistische Kurve”](https://commons.wikimedia.org/wiki/File:Ballistische_Kurve.webm) by Mathe-AC Lernwerkstatt Aachen (Rhetos), Wikimedia Commons, CC BY-SA 4.0. Download both into `samples/` to try them.
- BallDrop: Open Source Physics Tracker sample (used for local testing only; not redistributed).
- A hackathon project, not endorsed by Meta.

## License

MIT, see [LICENSE](LICENSE).
