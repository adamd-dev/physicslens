"""Video in, track and fits out: the steps shared by the CLI and the web API."""

import warnings
from dataclasses import dataclass

import numpy as np

from .agent.tools import Toolbox, TrackData
from .calibrate import Calibration
from .ingest import Video, load_video
from .track.base import Track
from .track.color import track_color
from .track.motion import field_order, track_motion

FITS = {"free_fall": "fit_free_fall", "projectile": "fit_projectile", "pendulum": "fit_pendulum"}
MIN_VX = 0.1  # m/s; slower horizontal drift is camera tilt or tracking noise, not a throw
MOTION_MAX_AREA = 2000


@dataclass
class TrackSettings:
    tracker: str = "motion"  # "motion" | "color"
    hsv_lower: tuple[int, int, int] = (0, 0, 200)
    hsv_upper: tuple[int, int, int] = (180, 60, 255)
    fields: str = "none"  # "none" | "auto" | "top" | "bottom"
    roi: tuple[int, int, int, int] | None = None
    min_area: int = 10
    max_area: int | None = None  # None: MOTION_MAX_AREA for motion, no limit for color
    max_jump: float | None = None


def load(
    path: str, start: float = 0.0, end: float | None = None, capture_fps: float | None = None, max_width: int | None = None
) -> tuple[Video, list[str]]:
    """Load the window; ingest warnings (repaired timestamps, slow motion) come back as text."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        video = load_video(path, start, end, capture_fps, max_width)
    return video, [str(w.message) for w in caught]


def track(video: Video, s: TrackSettings) -> tuple[str, Track]:
    """Returns (field mode used, track in pixels); raises ValueError when the object is not found."""
    if s.tracker == "color":
        return "none", track_color(video, s.hsv_lower, s.hsv_upper, s.min_area, s.max_jump, max_area=s.max_area, roi=s.roi)
    kw = dict(min_area=s.min_area, max_area=s.max_area or MOTION_MAX_AREA, max_jump_px=s.max_jump, roi=s.roi)
    if s.fields == "auto":
        return field_order(video, **kw)
    fields = None if s.fields == "none" else s.fields
    return s.fields, track_motion(video, fields=fields, **kw)


def to_track_data(tr: Track, calibration: Calibration) -> TrackData:
    # Shift time to the first tracked sample so the toolbox window covers the whole track.
    return TrackData(tr.t - tr.t[0], calibration.to_meters(tr.x), calibration.to_meters(tr.y))


def run_fits(box: Toolbox, model: str = "auto", axis: str = "x") -> list[dict]:
    names = list(FITS) if model == "auto" else [model]
    results = []
    for name in names:
        kwargs = {"axis": axis} if name == "pendulum" else {}
        summary = box.call(FITS[name], kwargs)
        if "error" not in summary:
            summary["check"] = box.check_plausibility(summary["fit_id"])
        results.append({"name": name, **summary})
    return results


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
    # Nothing passed: R² is scale-free, so it compares a 1 m drop with a 5 mm sideways wobble fairly.
    # r2 is None when it was not finite.
    return max(fitted.values(), key=lambda r: -np.inf if r["r2"] is None else r["r2"]) if fitted else None


def fitted_xy(box: Toolbox, fit_id: str) -> dict[str, list[float]]:
    """Fitted position in metres at the samples the fit used; the unfitted axis keeps the tracked value.

    Multiply by px_per_m to draw it over the video: TrackData metres have no offset.
    """
    n, model, axis, t0 = box.windows[fit_id]
    data, residuals = box.data, box.fits[fit_id].residuals
    start = int(np.searchsorted(data.t, t0 - 1e-9))
    idx = slice(start, start + n)
    x, y = data.x[idx].copy(), data.y[idx].copy()
    if model == "projectile":  # residuals hold x then y
        x -= residuals[:n]
        y -= residuals[n:]
    elif model == "free_fall" or (model == "pendulum" and axis == "y"):
        y -= residuals
    else:
        x -= residuals
    return {"t": data.t[idx].tolist(), "x": x.tolist(), "y": y.tolist()}
