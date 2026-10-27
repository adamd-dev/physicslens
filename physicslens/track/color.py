"""Fallback tracker: HSV color threshold, largest blob centroid.

Used when SAM 3.1 is unavailable or too slow, and as a sanity check against it.
"""

import cv2
import numpy as np

from ..ingest import Video
from .base import Track


def centroid(mask: np.ndarray) -> tuple[float, float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    m = cv2.moments(blob)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def track_color(video: Video, hsv_lower, hsv_upper) -> Track:
    lower, upper = np.array(hsv_lower, np.uint8), np.array(hsv_upper, np.uint8)
    ts, xs, ys = [], [], []
    for frame, t in zip(video.frames, video.timestamps):
        mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower, upper)
        c = centroid(mask)
        if c is not None:
            ts.append(t)
            xs.append(c[0])
            ys.append(c[1])
    return Track(np.array(ts), np.array(xs), np.array(ys), source="color")
