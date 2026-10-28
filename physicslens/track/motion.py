"""Motion tracker: subtract the median background, follow the moving blob.

Static objects that look like the ball (a white metre stick, a white wall)
vanish in the difference image. Handles interlaced video by tracking each
field separately, which doubles the sample rate.
"""

import cv2
import numpy as np

from ..ingest import Video
from .base import Track
from .color import MIN_POINTS, pick_blob


def _sized_blobs(mask: np.ndarray, min_area: int, max_area: int) -> np.ndarray:
    """Drop blobs outside [min_area, max_area], e.g. a moving arm much bigger than the ball."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    good = [i for i in range(1, n) if min_area <= stats[i, cv2.CC_STAT_AREA] <= max_area]
    return np.isin(labels, good).astype(np.uint8)


# Pixels "moving" in more than this share of frames are compression flicker on static
# edges (e.g. metre-stick ticks); a real moving object covers any pixel only briefly.
FLICKER_SHARE = 0.3


def _samples(video: Video, fields: str | None, threshold: int):
    """Yield (time, moving_mask, y_scale, y_offset) for each frame or field."""
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in video.frames], np.int16)
    moving = (np.abs(gray - np.median(gray, axis=0)) > threshold).astype(np.uint8)
    moving &= (moving.mean(axis=0) <= FLICKER_SHARE).astype(np.uint8)
    if fields is None:
        for mask, t in zip(moving, video.timestamps):
            yield t, mask, 1, 0
        return
    # Each field is half a frame interval apart; "bottom" means odd rows were exposed first.
    order = (1, 0) if fields == "bottom" else (0, 1)
    half_dt = 0.5 / video.fps
    for mask, t in zip(moving, video.timestamps):
        for k, parity in enumerate(order):
            yield t + k * half_dt, mask[parity::2], 2, parity


def track_motion(
    video: Video,
    min_area: int = 10,
    max_area: int = 2000,
    threshold: int = 40,
    fields: str | None = None,
    max_jump_px: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
) -> Track:
    """Track the moving object. `fields` is None, "top" or "bottom" (which field comes first).

    `roi` = (x0, y0, x1, y1) in full-frame pixels keeps hands and people out of the search.
    """
    if fields not in (None, "top", "bottom"):
        raise ValueError("fields must be None, 'top' or 'bottom'")
    ts, xs, ys = [], [], []
    previous = None
    for t, moving, y_scale, y_offset in _samples(video, fields, threshold):
        if roi is not None:
            x0, y0, x1, y1 = roi
            inside = np.zeros_like(moving)
            rows = slice(max(0, (y0 - y_offset + y_scale - 1) // y_scale), max(0, (y1 - y_offset) // y_scale + 1))
            inside[rows, x0:x1] = 1
            moving &= inside
        # Areas are given in full-frame pixels; a field has half the rows.
        mask = _sized_blobs(moving, min_area / y_scale, max_area / y_scale)
        # Compare in field coordinates so the continuity check uses like with like.
        prev_field: tuple[float, float] | None = None if previous is None else (previous[0], (previous[1] - y_offset) / y_scale)
        c = pick_blob(mask, min_area / y_scale, prev_field)
        if c is None:
            continue
        previous = (c[0], c[1] * y_scale + y_offset)
        ts.append(t)
        xs.append(previous[0])
        ys.append(previous[1])
    t_arr, x_arr, y_arr = np.array(ts), np.array(xs), np.array(ys)
    if max_jump_px is not None and len(t_arr) > 1:
        t_arr, x_arr, y_arr = _longest_segment(t_arr, x_arr, y_arr, max_jump_px, max_gap_s=3.5 / video.fps)
    if len(t_arr) < MIN_POINTS:
        raise ValueError(f"moving object found in only {len(t_arr)} samples; adjust threshold or area limits")
    return Track(t_arr, x_arr, y_arr, source="motion" if fields is None else f"motion-{fields}")


def _longest_segment(ts, xs, ys, max_jump_px: float, max_gap_s: float):
    """Keep the longest run without a position jump or time gap.

    Rejecting points one by one can lock onto a wrong early blob (a hand, a
    flickering edge); splitting and keeping the longest run lets the real
    trajectory win by length.
    """
    breaks = (np.hypot(np.diff(xs), np.diff(ys)) > max_jump_px) | (np.diff(ts) > max_gap_s)
    starts = np.r_[0, np.flatnonzero(breaks) + 1]
    ends = np.r_[starts[1:], len(ts)]
    k = int(np.argmax(ends - starts))
    return ts[starts[k] : ends[k]], xs[starts[k] : ends[k]], ys[starts[k] : ends[k]]


def roughness(track: Track) -> float:
    """Mean squared second difference of position; small for smooth physical motion."""
    return float(np.mean(np.diff(track.x, 2) ** 2 + np.diff(track.y, 2) ** 2))


def is_interlaced(field_track: Track, frame_dt: float) -> bool:
    """True when the two fields of a frame show the object in different places.

    Progressive video split into fields repeats each position; interlaced video
    moves about half as far between fields as between frames.
    """
    t, x, y = field_track.t, field_track.x, field_track.y
    step = np.hypot(np.diff(x), np.diff(y))
    # The second field is stamped half a frame later, so flooring maps both fields to their frame.
    same_frame = np.diff(np.floor(t / frame_dt + 1e-6)) == 0
    if same_frame.sum() < 3 or (~same_frame).sum() < 3:
        return False
    return float(np.median(step[same_frame])) > 0.25 * float(np.median(step[~same_frame]))


def field_order(video: Video, **kwargs) -> tuple[str, Track]:
    """Return ("none" | "top" | "bottom", track) for possibly interlaced video.

    The wrong field order makes the object jump back and forth every half frame,
    so the smoother trajectory wins; no physics model needed.
    """
    candidates = {order: track_motion(video, fields=order, **kwargs) for order in ("top", "bottom")}
    best = min(candidates, key=lambda order: roughness(candidates[order]))
    if not is_interlaced(candidates[best], 1.0 / video.fps):
        return "none", track_motion(video, fields=None, **kwargs)
    return best, candidates[best]
