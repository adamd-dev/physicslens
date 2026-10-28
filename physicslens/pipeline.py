"""Video in, track and fits out: the steps shared by the CLI and the web API."""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from .agent.tools import Toolbox, TrackData
from .calibrate import Calibration
from .ingest import Video, load_video
from .llm.meta_client import LLMError
from .track.base import Track
from .track.color import track_color
from .track.motion import field_order, track_motion
from .track.sam import Segmenter, track_sam

FITS = {"free_fall": "fit_free_fall", "projectile": "fit_projectile", "pendulum": "fit_pendulum"}
MIN_VX = 0.1  # m/s; slower horizontal drift is camera tilt or tracking noise, not a throw
MOTION_MAX_AREA = 2000
# A still ball (in the hand before release, on the floor after landing) stays within the rest radius
# for at least MIN_REST_S. The radius never drops below REST_RADIUS_PX: tracking jitter is 1-2 px, and
# interlaced fields move a still ball half a line. Shorter still runs are the slow first frames of motion.
REST_RADIUS_M = 0.005
REST_RADIUS_PX = 4.0
MIN_REST_S = 0.08
MIN_MOVING = 8  # fewer moving samples than this: too little left to fit honestly, keep the track as is


@dataclass
class TrackSettings:
    tracker: str = "motion"  # "motion" | "color" | "sam"
    hsv_lower: tuple[int, int, int] = (0, 0, 200)
    hsv_upper: tuple[int, int, int] = (180, 60, 255)
    fields: str = "none"  # "none" | "auto" | "top" | "bottom"
    roi: tuple[int, int, int, int] | None = None
    min_area: int = 10
    max_area: int | None = None  # None: MOTION_MAX_AREA for motion, no limit for color
    max_jump: float | None = None
    sam_box: tuple[int, int, int, int] | None = None  # around the object on the first frame; wins over the prompt
    sam_prompt: str = "ball"


def _sam_client() -> Segmenter:
    from .llm.sam_client import SamClient

    return SamClient.from_env()


def load(
    path: str, start: float = 0.0, end: float | None = None, capture_fps: float | None = None, max_width: int | None = None
) -> tuple[Video, list[str]]:
    """Load the window; ingest warnings (repaired timestamps, slow motion) come back as text."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        video = load_video(path, start, end, capture_fps, max_width)
    return video, [str(w.message) for w in caught]


def track(video: Video, s: TrackSettings, sam: Callable[[], Segmenter] = _sam_client) -> tuple[str, Track, list[str]]:
    """Returns (field mode used, track in pixels, notes); raises ValueError when the object is not found.

    When SAM 3.1 cannot track (no key, API error, budget, nothing found) the motion tracker takes over
    and a note says why: the result is never silently from another tracker.
    """
    if s.tracker == "sam":
        try:
            prompt = None if s.sam_box else s.sam_prompt
            return "none", track_sam(video, sam(), box=s.sam_box, prompt=prompt, roi=s.roi), []
        except (LLMError, ValueError, OSError) as exc:
            note = f"SAM 3.1 could not track ({exc}); used the motion tracker instead"
            try:
                fields, tr, _ = track(video, replace(s, tracker="motion"))
            except ValueError as motion_exc:
                raise ValueError(f"{motion_exc} ({note})") from exc
            return fields, tr, [note]
    if s.tracker == "color":
        return "none", track_color(video, s.hsv_lower, s.hsv_upper, s.min_area, s.max_jump, max_area=s.max_area, roi=s.roi), []
    kw = dict(min_area=s.min_area, max_area=s.max_area or MOTION_MAX_AREA, max_jump_px=s.max_jump, roi=s.roi)
    if s.fields == "auto":
        return (*field_order(video, **kw), [])
    fields = None if s.fields == "none" else s.fields
    return s.fields, track_motion(video, fields=fields, **kw), []


def _still_run(t: np.ndarray, x: np.ndarray, y: np.ndarray, radius_px: float) -> int:
    """Length of the leading run of samples near where the track starts, if it lasts MIN_REST_S
    and the object never comes back there.

    The anchor is the median of the first 3 samples, so one bad first centroid cannot hide a hold.
    NaN counts as moving: an unknown position never extends a still run.
    """
    ax, ay = np.median(x[:3]), np.median(y[:3])
    far = ~(np.hypot(x - ax, y - ay) <= radius_px)
    far[0] = False  # judged by the samples after it, through the median anchor
    n = int(np.argmax(far)) if far.any() else len(t)
    if n < 2 or abs(t[n - 1] - t[0]) < MIN_REST_S:
        return 0
    # A held or landed ball leaves for good; a pendulum comes back to its turning point every swing.
    return 0 if (~far[n:]).any() else n


def rest_samples(tr: Track, px_per_m: float) -> tuple[int, int]:
    """(leading, trailing) samples where the object sits still: held before release, lying after landing.

    They are not free motion. Fitting them bends the curve through a pause and biases g low.
    """
    if len(tr) < MIN_MOVING:
        return 0, 0
    radius = max(REST_RADIUS_M * px_per_m, REST_RADIUS_PX)
    lead = _still_run(tr.t, tr.x, tr.y, radius)
    trail = _still_run(tr.t[::-1], tr.x[::-1], tr.y[::-1], radius)
    if len(tr) - lead - trail < MIN_MOVING:
        return 0, 0
    return lead, trail


@dataclass
class Trim:
    track: Track  # the samples kept for fitting, original times
    lead: int  # still samples dropped at the start
    trail: int  # still samples dropped at the end
    notes: list[str]


def trim_rest(tr: Track, px_per_m: float) -> Trim:
    """Drop the still samples at both ends; the notes say what went, so nothing changes silently."""
    lead, trail = rest_samples(tr, px_per_m)
    if not lead and not trail:
        return Trim(tr, 0, 0, [])
    notes = []
    if lead:
        notes.append(f"dropped {lead} samples at rest before the motion ({tr.t[lead - 1] - tr.t[0]:.2f} s: "
                     "the ball held before release)")
    if trail:
        notes.append(f"dropped {trail} samples at rest after the motion ({tr.t[-1] - tr.t[-trail]:.2f} s: "
                     "the ball lying still after landing)")
    end = len(tr) - trail
    return Trim(Track(tr.t[lead:end], tr.x[lead:end], tr.y[lead:end], tr.source), lead, trail, notes)


def agent_question(height_m: float | None = None, rest_notes: Sequence[str] = ()) -> str:
    """The agent's task; shared by the CLI and the API so both send the same text (and hit the same cache)."""
    question = "Measure the physics in this tracked trajectory."
    if height_m:
        question += f" The launch edge is {height_m} m above the landing surface; if this is a projectile, predict where it lands."
    if rest_notes:
        # So the summary does not blame calibration for a pause the agent never saw.
        question += " Samples where the ball was at rest (" + "; ".join(rest_notes) + ") were already removed from the track."
    return question


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
    # Nothing passed: fewest failed checks first, then R², which is scale-free (RMSE in mm would compare
    # a 1 m drop with a 5 mm sideways wobble). r2 is None when it was not finite.
    def rank(r: dict) -> tuple[int, float]:
        return len(r["check"]["problems"]), np.inf if r["r2"] is None else -r["r2"]

    return min(fitted.values(), key=rank) if fitted else None


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
