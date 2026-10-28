"""PhysicsLens prototype CLI: video in, measured physics out.

Example (Tracker sample, interlaced 30 fps, 1 m stick = 400 px):
    python cli.py samples/tracker/mechanics_videos/BallDrop.mp4 \
        --px-per-m 400 --roi 150,60,260,480 --min-area 60 --max-area 120 --max-jump 60 --fields auto

Without --model, every model is fitted and the plausible one with the lowest
error wins. This deterministic baseline is what the agent loop must beat.
"""

import argparse
import dataclasses
import json
import os
import sys
import warnings

import numpy as np

from physicslens.agent.loop import run_agent
from physicslens.agent.tools import Toolbox, TrackData
from physicslens.calibrate import Calibration
from physicslens.ingest import load_video
from physicslens.llm.meta_client import LLMError, MetaClient
from physicslens.physics import models
from physicslens.track.color import track_color
from physicslens.track.motion import field_order, track_motion

FITS = {"free_fall": "fit_free_fall", "projectile": "fit_projectile", "pendulum": "fit_pendulum"}


def _numbers(text: str, n: int, cast=float):
    values = [cast(v) for v in text.split(",")]
    if len(values) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers, got {text!r}")
    return tuple(values)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    scale = p.add_mutually_exclusive_group(required=True)
    scale.add_argument("--px-per-m", type=float, help="pixels per metre in the plane of motion")
    scale.add_argument("--ref", type=lambda s: _numbers(s, 5), help="x1,y1,x2,y2,length_m of a reference object")
    p.add_argument("--tracker", choices=["motion", "color"], default="motion")
    p.add_argument("--hsv-lower", type=lambda s: _numbers(s, 3, int), default=(0, 0, 200))
    p.add_argument("--hsv-upper", type=lambda s: _numbers(s, 3, int), default=(180, 60, 255))
    p.add_argument("--fields", choices=["none", "auto", "top", "bottom"], default="none", help="interlaced video")
    p.add_argument("--roi", type=lambda s: _numbers(s, 4, int), help="x0,y0,x1,y1 search area in full-frame pixels")
    p.add_argument("--min-area", type=int, default=10)
    p.add_argument("--max-area", type=int, help="largest blob in px (default: 2000 for motion, no limit for color)")
    p.add_argument("--max-jump", type=float, help="max px between samples before a point is rejected")
    p.add_argument("--capture-fps", type=float, help="real capture rate of slow-motion clips, e.g. 240")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float)
    p.add_argument("--model", choices=["auto", *FITS], default="auto")
    p.add_argument("--axis", choices=["x", "y"], default="x", help="pendulum swing axis")
    p.add_argument("--height", type=float, help="drop height in m for a landing prediction (projectile)")
    p.add_argument("--plot", help="save trajectory and residual plot to this PNG")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    p.add_argument("--agent", action="store_true", help="let the Muse agent choose, check and report (needs MODEL_API_KEY)")
    p.add_argument("--transcript", help="save the agent's messages and tool calls to this JSON file")
    args = p.parse_args(argv)
    if args.transcript and not args.agent:
        p.error("--transcript needs --agent")
    if args.agent and args.json:
        p.error("--json is not supported with --agent yet")
    if args.agent and args.model != "auto":
        p.error("--model picks the model by hand; drop it or drop --agent")
    return args


def make_client():
    """The runtime model; tests swap this for a scripted MockClient."""
    return MetaClient.from_env()


def report_agent(args, data: TrackData, box: Toolbox, video_warnings) -> int:
    """Exit codes: 0 accepted and plausible, 1 accepted but flagged, 3 setup/API failure, 4 no answer."""
    for w in video_warnings:
        print(f"  warning: {w.message}", file=sys.stderr)
    question = "Measure the physics in this tracked trajectory."
    if args.height:
        question += f" The launch edge is {args.height} m above the landing surface; if this is a projectile, predict where it lands."
    try:
        result = run_agent(make_client(), box, question)
    except (LLMError, OSError, ValueError) as exc:
        print(f"agent failed: {exc}", file=sys.stderr)
        return 3
    if args.transcript:
        # Saved before the verdict: a run with no accepted answer is the one worth reading.
        try:
            text = json.dumps([dataclasses.asdict(m) for m in result.transcript], indent=2, ensure_ascii=False, default=str)
            # Owner-only: prompts and tool output stay on this machine's account.
            with os.fdopen(os.open(args.transcript, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="utf-8") as fh:
                fh.write(text)
        except (OSError, TypeError, ValueError) as exc:
            print(f"  warning: transcript not saved: {exc}", file=sys.stderr)
    if result.fit_id is None:
        print(f"agent gave no accepted answer after {result.steps} steps: {result.summary}", file=sys.stderr)
        return 4
    # Measured values first: they come from the tools, the summary comes from the model.
    fit = box.fits[result.fit_id]
    status = "ok" if result.check["ok"] else "; ".join(result.check["problems"])
    print(f"agent chose {result.fit_id} ({fit.model}) in {result.steps} steps: r2={fit.r2:.5f} [{status}]")
    for name, value in fit.params.items():
        print(f"  {name} = {value:.4f} ± {fit.stderr[name]:.4f}")
    print(f"\n  model summary (written by the model, not verified): {result.summary}")
    if not result.check["ok"]:
        print(f"WARNING: the model accepted a fit that failed checks: {status}", file=sys.stderr)
    if args.plot:
        plot(args.plot, data, box, {"fit_id": result.fit_id, "model": fit.model}, args.axis)
    return 0 if result.check["ok"] else 1


def track(args, video):
    if args.tracker == "color":
        return "none", track_color(
            video, args.hsv_lower, args.hsv_upper, args.min_area, args.max_jump, max_area=args.max_area, roi=args.roi
        )
    kw = dict(min_area=args.min_area, max_area=args.max_area or 2000, max_jump_px=args.max_jump, roi=args.roi)
    if args.fields == "auto":
        return field_order(video, **kw)
    fields = None if args.fields == "none" else args.fields
    return args.fields, track_motion(video, fields=fields, **kw)


def run_fits(box: Toolbox, args) -> list[dict]:
    names = list(FITS) if args.model == "auto" else [args.model]
    results = []
    for name in names:
        kwargs = {"axis": args.axis} if name == "pendulum" else {}
        summary = box.call(FITS[name], kwargs)
        if "error" not in summary:
            summary["check"] = box.check_plausibility(summary["fit_id"])
        results.append({"name": name, **summary})
    return results


MIN_VX = 0.1  # m/s; slower horizontal drift is camera tilt or tracking noise, not a throw


def choose(results: list[dict]) -> dict | None:
    """Prefer the simplest plausible model; RMSE across different models is not comparable."""
    fitted = {r["name"]: r for r in results if "error" not in r}
    plausible = {name: r for name, r in fitted.items() if r["check"]["ok"]}
    projectile = plausible.get("projectile")
    if projectile:
        vx, sx = projectile["params"]["vx"], projectile["stderr"]["vx"]
        if abs(vx) > MIN_VX and abs(vx) > 3 * sx:
            return projectile
    for name in ("free_fall", "projectile", "pendulum"):
        if name in plausible:
            return plausible[name]
    return min(fitted.values(), key=lambda r: r["rmse_m"]) if fitted else None


def plot(path, data: TrackData, box: Toolbox, best: dict, axis: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = box.fits[best["fit_id"]]
    signal = data.x if best["model"] == "damped_oscillation" and axis == "x" else data.y
    residuals = result.residuals[-len(signal):]  # projectile stores x then y residuals
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, height_ratios=[3, 1])
    top.plot(data.t, signal, "o", ms=4, label="tracked")
    top.plot(data.t, signal - residuals, "-", label=f"{best['model']} fit")
    top.set_ylabel("position (m, y down)" if signal is data.y else "position (m)")
    top.legend()
    bottom.axhline(0, color="gray", lw=0.8)
    bottom.plot(data.t, residuals * 1000, "o", ms=3)
    bottom.set_ylabel("residual (mm)")
    bottom.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(argv=None) -> int:
    args = parse_args(argv)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        video = load_video(args.video, args.start, args.end, args.capture_fps)
    calibration = Calibration.from_reference(args.px_per_m, 1.0) if args.px_per_m else Calibration.from_points(
        args.ref[:2], args.ref[2:4], args.ref[4]
    )
    try:
        fields, tr = track(args, video)
    except ValueError as exc:
        print(f"tracking failed: {exc}", file=sys.stderr)
        return 2
    # Shift time to the first tracked sample so the toolbox window covers the whole track.
    data = TrackData(tr.t - tr.t[0], calibration.to_meters(tr.x), calibration.to_meters(tr.y))
    box = Toolbox(data)
    if args.agent:
        return report_agent(args, data, box, caught)
    results = run_fits(box, args)
    best = choose(results)

    report = {
        "video": args.video,
        "frames": len(video.frames),
        "fps": round(video.fps, 3),
        "timestamp_source": video.timestamp_source,
        "fields": fields,
        "tracked_samples": len(tr),
        "px_per_m": round(calibration.px_per_m, 2),
        "warnings": [str(w.message) for w in caught],
        "fits": results,
        "best": best["fit_id"] if best else None,
    }
    if best and args.height and best["model"] == "projectile":
        report["landing"] = box.call("predict_landing", {"fit_id": best["fit_id"], "height_m": args.height})

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{args.video}: {len(tr)}/{report['frames']} samples tracked "
              f"(fps {report['fps']}, timestamps {video.timestamp_source}, fields {fields})")
        for w in report["warnings"]:
            print(f"  warning: {w}")
        for r in results:
            if "error" in r:
                print(f"  {r['name']:<11} error: {r['error']}")
                continue
            mark = "*" if best and r["fit_id"] == best["fit_id"] else " "
            status = "ok" if r["check"]["ok"] else "; ".join(r["check"]["problems"])
            print(f" {mark}{r['name']:<11} r2={r['r2']:.5f} rmse={r['rmse_m'] * 1000:.2f} mm  [{status}]")
        if best:
            p, s = best["params"], best["stderr"]
            if "g" in p:
                err = 100 * (p["g"] - models.G) / models.G
                print(f"\n  g = {p['g']:.3f} ± {s['g']:.3f} m/s²  ({err:+.1f}% vs {models.G})")
            if "length_m" in best:
                period = 2 * np.pi / p["omega"]
                print(f"\n  pendulum length = {best['length_m']:.3f} m  (period {period:.3f} s)")
            if "vx" in p:
                print(f"  vx = {p['vx']:.3f} ± {s['vx']:.3f} m/s")
        if "landing" in report:
            land = report["landing"]
            print(f"  landing distance = {land['landing_distance_m']:.3f} ± {land['uncertainty_m']:.3f} m")
    if args.plot and best:
        plot(args.plot, data, box, best, args.axis)
        if not args.json:
            print(f"  plot saved to {args.plot}")
    return 0 if best and best["check"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
