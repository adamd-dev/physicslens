"""Read video frames with real timestamps.

Phone videos often have variable frame timing, so timestamps come from the
container when they look sane and fall back to index / fps otherwise.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Video:
    frames: list[np.ndarray]
    timestamps: np.ndarray
    fps: float
    width: int
    height: int


def load_video(path: str, start_s: float = 0.0, end_s: float | None = None) -> Video:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames, container_ts = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        container_ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()

    index_ts = np.arange(len(frames)) / fps
    container_ts = np.asarray(container_ts)
    monotonic = len(container_ts) > 1 and np.all(np.diff(container_ts) > 0)
    timestamps = container_ts - container_ts[0] if monotonic else index_ts

    keep = timestamps >= start_s
    if end_s is not None:
        keep &= timestamps <= end_s
    kept_frames = [f for f, k in zip(frames, keep) if k]
    return Video(kept_frames, timestamps[keep], fps, width, height)
