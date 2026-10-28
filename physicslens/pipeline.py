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
# A still ball (in the hand before release, on the floor after landing) moves less than REST_RADIUS_M
# for at least MIN_REST_S. Shorter still runs are the slow first frames of real motion.
REST_RADIUS_M = 0.005
MIN_REST_S = 0.03
MIN_MOVING = 3  # fewer moving samples than this: nothing moved, leave the track for the checks to reject


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


def _still_run(t: np.ndarray, x: np.ndarray, y: np.ndarray, radius_px: float) -> int:
    """Length of the leading run of samples within radius_px of the first one, if it lasts MIN_REST_S."""
    far = np.hypot(x - x[0], y - y[0]) > radius_px
    n = int(np.argmax(far)) if far.any() else len(t)
    return n if n >= 2 and abs(t[n - 1] - t[0]) >= MIN_REST_S else 0


def rest_samples(tr: Track, px_per_m: float) -> tuple[int, int]:
    """(leading, trailing) samples where the object sits still: held before release, lying after landing.

    They are not free motion. Fitting them bends the curve through a pause and biases g low.
    """
    if len(tr) < 2:
        return 0, 0
    radius = REST_RADIUS_M * px_per_m
    lead = _still_run(tr.t, tr.x, tr.y, radius)
    trail = _still_run(tr.t[::-1], tr.x[::-1], tr.y[::-1], radius)
    if len(tr) - lead - trail < MIN_MOVING:
        return 0, 0
    return lead, trail


def trim_rest(tr: Track, px_per_m: float) -> tuple[Track, list[str]]:
    """Drop the still samples at both ends; the notes say what went, so nothing changes silently."""
    lead, trail = rest_samples(tr, px_per_m)
    if not lead and not trail:
        return tr, []
    notes = []
    if lead:
        notes.append(f"dropped {lead} samples at rest before the motion ({tr.t[lead - 1] - tr.t[0]:.2f} s, "
                     "e.g. the ball still in the hand)")
    if trail:
        notes.append(f"dropped {trail} samples at rest after the motion ({tr.t[-1] - tr.t[-trail]:.2f} s, "
                     "e.g. the ball lying on the floor)")
    end = len(tr) - trail
    return Track(tr.t[lead:end], tr.x[lead:end], tr.y[lead:end], tr.source), notes


def release_question(notes: list[str]) -> str:
    """One sentence for the agent, so its summary does not blame calibration for a pause it never saw."""
    if not notes:
        return ""
    return " Samples where the ball was at rest (" + "; ".join(notes) + ") were already removed from the track."


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
