"""SAM 3.1 tracker: Meta's video segmentation model follows the object; the box centre is its position.

The clip sent is re-encoded from the frames already loaded (window and resize applied), so SAM's
frame N is exactly video.timestamps[N] and its pixels are the decoded frame's pixels. A text prompt
finds every match (a small ball is often missed); a box drawn around the object on the first frame
follows that one object.
"""

import re
import tempfile
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from ..ingest import Video
from .base import Track

MIN_POINTS = 5
Box = tuple[int, int, int, int]
_FRAME = re.compile(r"^<(\d+)f>")
_BOX = re.compile(r"(\d+)<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);")


class Segmenter(Protocol):
    def segment(self, clip: bytes, prompt: str, direction: str = "both") -> str: ...


def encode_clip(video: Video) -> bytes:
    """MPEG-4 (accepted by SAM, unlike WebM) of exactly the loaded frames."""
    h, w = video.frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "clip.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(video.fps, 1.0), (w, h))
        if not writer.isOpened():
            raise ValueError("OpenCV cannot write an MP4 clip for SAM")
        for frame in video.frames:
            writer.write(frame)
        writer.release()
        return path.read_bytes()


def box_prompt(box, width: int, height: int) -> str:
    """Object prompt on frame 0: inclusive integer box in the clip's pixels, clipped to the frame."""
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, x2 = sorted((min(max(x1, 0), width - 1), min(max(x2, 0), width - 1)))
    y1, y2 = sorted((min(max(y1, 0), height - 1), min(max(y2, 0), height - 1)))
    return f"<0f>1<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height}|>"


def parse_frames(text: str) -> dict[int, dict[int, Box]]:
    """{frame: {object id: box}}. Lines come out of order, frames without objects have no line,
    and ids are not dense, so everything is keyed, never positional."""
    frames: dict[int, dict[int, Box]] = {}
    for line in text.splitlines():
        head = _FRAME.match(line)
        if not head:
            continue
        objects = {int(m[1]): (int(m[2]), int(m[3]), int(m[4]), int(m[5])) for m in _BOX.finditer(line)}
        if objects:
            frames.setdefault(int(head[1]), {}).update(objects)
    return frames


def _centre(b: Box) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def pick_object(frames: dict[int, dict[int, Box]], roi=None) -> int | None:
    """The object seen in the most frames (inside the search area, if any), then the one that moves most:
    a ball lying in the background is seen as often as the moving one but does not move."""
    paths: dict[int, list[tuple[float, float]]] = {}
    for objects in frames.values():
        for oid, b in objects.items():
            cx, cy = _centre(b)
            if roi is None or (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                paths.setdefault(oid, []).append((cx, cy))
    if not paths:
        return None

    def rank(oid: int) -> tuple[int, float]:
        xy = np.array(paths[oid])
        return len(xy), float(np.hypot(*np.ptp(xy, axis=0)))

    return max(paths, key=rank)


def track_sam(video: Video, client: Segmenter, box=None, prompt: str | None = None, roi=None) -> Track:
    """Track one object with SAM 3.1; raises ValueError (with what to try) when it is not followed."""
    h, w = video.frames[0].shape[:2]
    if box is not None:
        request, direction, wanted = box_prompt(box, w, h), "forward", "the boxed object"
    elif prompt and prompt.strip():
        request, direction, wanted = prompt.strip(), "both", repr(prompt.strip())
    else:
        raise ValueError("SAM needs a box or a text prompt")
    frames = parse_frames(client.segment(encode_clip(video), request, direction))
    frames = {n: objects for n, objects in frames.items() if n < len(video.frames)}
    oid = pick_object(frames, roi)
    if oid is None:
        raise ValueError(f"SAM found no {wanted} in the clip; draw a box around the ball on the first frame")
    ns = sorted(n for n, objects in frames.items() if oid in objects)
    if len(ns) < MIN_POINTS:
        raise ValueError(f"SAM followed {wanted} in only {len(ns)} frames; need at least {MIN_POINTS}")
    centres = np.array([_centre(frames[n][oid]) for n in ns])
    return Track(video.timestamps[ns], centres[:, 0], centres[:, 1], "sam")
