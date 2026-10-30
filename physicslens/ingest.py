"""Read video frames with real timestamps.

Phone videos often have variable frame timing, so timestamps come from the
container when they look sane and fall back to index / fps otherwise.

Slow-motion files (e.g. Samsung 240 fps) are written to play back at 30 fps, so
their container clock runs slow. Pass `capture_fps` to rescale to real time.
"""

import math
import warnings
from dataclasses import dataclass

import cv2
import numpy as np

# Container clock may disagree with the declared fps by this much before we distrust it.
MAX_CLOCK_MISMATCH = 0.2


@dataclass
class Video:
    frames: list[np.ndarray]
    timestamps: np.ndarray  # real seconds, zero at the first kept frame
    fps: float  # real capture rate
    width: int
    height: int
    timestamp_source: str  # "container" or "index"
    time_scale: float  # real seconds per playback second
    scale: float = 1.0  # decoded pixels per source pixel (< 1 when max_width shrank the frames)


def _repair(ts: np.ndarray) -> np.ndarray | None:
    """Fix isolated duplicate / backwards stamps by interpolation; None if hopeless."""
    idx = np.arange(len(ts))
    good = np.r_[True, np.diff(ts) > 0]
    good &= ts >= np.maximum.accumulate(ts)
    if good.sum() < max(2, len(ts) // 2):
        return None
    fixed = np.interp(idx, idx[good], ts[good])
    return fixed if np.all(np.diff(fixed) > 0) else None


def load_video(
    path: str,
    start_s: float = 0.0,
    end_s: float | None = None,
    capture_fps: float | None = None,
    max_width: int | None = None,
    max_frames: int | None = None,
) -> Video:
    """Load frames in the half-open real-time window [start_s, end_s).

    `max_width` shrinks wider frames on load: a 1080p60 phone clip is ~6 MB per frame in memory.
    `max_frames`: a longer window is refused as soon as it is exceeded, before the rest is decoded.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    playback_fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(playback_fps) or playback_fps <= 0:
        playback_fps = 30.0
    time_scale = playback_fps / capture_fps if capture_fps else 1.0
    source_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    scale = max_width / source_width if max_width and source_width > max_width else 1.0

    frames: list[np.ndarray] = []
    container_ts: list[float] = []
    index_ts: list[float] = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_container = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 * time_scale
        t_index = i / playback_fps * time_scale
        i += 1
        # Window on index time: container stamps are not trusted until checked below.
        if t_index < start_s:
            continue
        if end_s is not None and t_index >= end_s:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if max_frames is not None and len(frames) >= max_frames:
            cap.release()
            raise ValueError(f"the window has more than {max_frames} frames ({max_frames * time_scale / playback_fps:.1f} s): "
                             "set a start and end around the motion")
        frames.append(frame)
        container_ts.append(t_container)
        index_ts.append(t_index)
    cap.release()
    if not frames:
        raise ValueError(f"no frames in window [{start_s}, {end_s}) of {path}")

    expected_dt = time_scale / playback_fps
    timestamps, source = np.asarray(index_ts), "index"
    if len(frames) > 1:
        repaired = _repair(np.asarray(container_ts))
        if repaired is not None:
            median_dt = float(np.median(np.diff(repaired)))
            if abs(median_dt - expected_dt) / expected_dt <= MAX_CLOCK_MISMATCH:
                timestamps, source = repaired, "container"
                dt = np.diff(repaired)
                if capture_fps and np.std(dt) / median_dt > 0.2:
                    warnings.warn("frame spacing is uneven: slow-motion clip may contain a speed ramp")
        if source == "index":
            warnings.warn("container timestamps unusable; falling back to index / fps")

    height, width = frames[0].shape[:2]  # decoded shape, which already reflects rotation
    return Video(frames, timestamps - timestamps[0], 1.0 / expected_dt, width, height, source, time_scale, scale)
