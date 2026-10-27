"""Fallback tracker: HSV color threshold, blob nearest the previous position.

Used when SAM 3.1 is unavailable or too slow, and as a sanity check against it.
"""

import cv2
import numpy as np

from ..ingest import Video
from .base import Track

MIN_POINTS = 5
_KERNEL = np.ones((3, 3), np.uint8)


def color_mask(frame: np.ndarray, hsv_lower, hsv_upper) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = np.array(hsv_lower, np.uint8), np.array(hsv_upper, np.uint8)
    if lower[0] <= upper[0]:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # Red wraps around hue 0/180: take [lower_h, 180] and [0, upper_h].
        mask = cv2.inRange(hsv, lower, np.array([180, upper[1], upper[2]], np.uint8))
        mask |= cv2.inRange(hsv, np.array([0, lower[1], lower[2]], np.uint8), upper)
    # Opening removes single-pixel specks that would otherwise pass as the ball.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)


def pick_blob(mask: np.ndarray, min_area: int, previous: tuple[float, float] | None) -> tuple[float, float] | None:
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    candidates = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not candidates:
        return None
    if previous is None:
        best = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    else:
        best = min(candidates, key=lambda i: np.hypot(*(centroids[i] - previous)))
    return float(centroids[best][0]), float(centroids[best][1])


def track_color(video: Video, hsv_lower, hsv_upper, min_area: int = 20, max_jump_px: float | None = None) -> Track:
    """Track one colored object; frames where it is lost or jumps too far are dropped, not guessed."""
    ts, xs, ys = [], [], []
    previous = None
    for frame, t in zip(video.frames, video.timestamps):
        c = pick_blob(color_mask(frame, hsv_lower, hsv_upper), min_area, previous)
        if c is None:
            continue
        if previous is not None and max_jump_px is not None and np.hypot(c[0] - previous[0], c[1] - previous[1]) > max_jump_px:
            continue
        previous = c
        ts.append(t)
        xs.append(c[0])
        ys.append(c[1])
    if len(ts) < MIN_POINTS:
        raise ValueError(f"object found in only {len(ts)} of {len(video.frames)} frames; check the HSV range")
    return Track(np.array(ts), np.array(xs), np.array(ys), source="color")
