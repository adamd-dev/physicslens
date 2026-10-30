"""SAM 3.1 tracker: Meta's video segmentation model follows the object; the box centre is its position.

The clip sent is re-encoded from the frames already loaded (window and resize applied), so SAM's
frame N is video.timestamps[N] and its coordinates are the decoded frame's (MPEG-4 is lossy, the
geometry is not changed). A text prompt finds every match (a small ball is often missed); a box
drawn around the object on the first frame follows that one object.
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
MAX_SAM_FRAMES = 900  # 30 s at 30 fps, $0.18: longer clips need a start/end window first
MIN_MOTION_PX = 5.0  # a box centre that wanders less than this is a still object, not the one thrown
Box = tuple[int, int, int, int]
_FRAME = re.compile(r"^<(\d+)f>")
_BOX = re.compile(r"(\d+)<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);")


class SamTrackError(ValueError):
    """SAM answered, but the answer cannot be used as a track (nothing found, too few frames, bad mapping)."""


class Segmenter(Protocol):
    def segment(self, clip: bytes, prompt: str, direction: str = "both", frames: int | None = None) -> str: ...


def encode_clip(video: Video) -> bytes:
    """MPEG-4 (accepted by SAM, unlike WebM) of exactly the loaded frames."""
    h, w = video.frames[0].shape[:2]
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), max(video.fps, 1.0), (w, h))
            if not writer.isOpened():
                raise SamTrackError("OpenCV cannot write an MP4 clip for SAM")
            for frame in video.frames:
                writer.write(frame)
            writer.release()
            return path.read_bytes()
    except cv2.error as exc:
        raise SamTrackError(f"could not encode the clip for SAM: {exc}") from None


# A noun phrase only: "<" and "|" would let raw SAM prompt tokens through.
TEXT_PROMPT_PATTERN = r"^[A-Za-z0-9 ,.'-]+$"
MAX_TEXT_PROMPT = 80


def box_prompt(box, width: int, height: int, frame: int = 0) -> str:
    """Object prompt on `frame`: inclusive integer box in the clip's pixels, clipped to the frame."""
    x1, y1, x2, y2 = (round(v) for v in box)
    x1, x2 = sorted((min(max(x1, 0), width - 1), min(max(x2, 0), width - 1)))
    y1, y2 = sorted((min(max(y1, 0), height - 1), min(max(y2, 0), height - 1)))
    return f"<{frame}f>1<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={width};h={height}|>"


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


def _iou(a, b) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pick_object(frames: dict[int, dict[int, Box]], roi=None, prompt_box=None, prompt_frame: int = 0) -> int | None:
    """Which SAM object is the ball.

    With a box prompt: the object that best overlaps the box on the prompt's frame (or, if SAM
    returned nothing there, the nearest frame it did return: the ball moves between frames).
    With a text prompt: objects that move beat still ones (a ball lying in the background is seen
    in every frame but never moves), then the one seen in the most frames inside the search area.
    """
    if prompt_box is not None:
        if not frames:
            return None
        first = frames[min(frames, key=lambda i: (abs(i - prompt_frame), i))]
        best = max(first, key=lambda oid: _iou(first[oid], prompt_box))
        # Nothing where the user boxed: say so rather than follow some other object.
        return best if _iou(first[best], prompt_box) > 0 else None
    paths: dict[int, list[tuple[float, float]]] = {}
    for objects in frames.values():
        for oid, b in objects.items():
            cx, cy = _centre(b)
            if roi is None or (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                paths.setdefault(oid, []).append((cx, cy))
    if not paths:
        return None

    def rank(oid: int) -> tuple[bool, int, float]:
        xy = np.array(paths[oid])
        extent = float(np.hypot(*np.ptp(xy, axis=0)))
        return extent >= MIN_MOTION_PX, len(xy), extent

    return max(paths, key=rank)


def track_sam(video: Video, client: Segmenter, box=None, prompt: str | None = None, roi=None, box_frame: int = 0) -> Track:
    """Track one object with SAM 3.1; raises SamTrackError (with what to try) when it is not followed.

    `box_frame`: the frame the box is on. After frame 0, SAM tracks both ways from it (tested 2026-10-29).
    """
    n = len(video.frames)
    if n > MAX_SAM_FRAMES:
        raise SamTrackError(f"the window has {n} frames; SAM takes at most {MAX_SAM_FRAMES}: set a start and end")
    h, w = video.frames[0].shape[:2]
    if box is not None:
        direction = "forward" if box_frame == 0 else "both"
        request, wanted = box_prompt(box, w, h, box_frame), "the boxed object"
    elif prompt and prompt.strip():
        request, direction, wanted = prompt.strip(), "both", repr(prompt.strip())
    else:
        raise SamTrackError("SAM needs a box or a text prompt")
    frames = parse_frames(client.segment(encode_clip(video), request, direction, frames=n))
    if frames and max(frames) >= n:
        # Our frame N is SAM's frame N only if it decoded exactly our frames; anything else would mistime every point.
        raise SamTrackError(f"SAM returned frame {max(frames)}, beyond the clip's {n} frames: the frame mapping is off")
    oid = pick_object(frames, roi, prompt_box=box, prompt_frame=box_frame)
    if oid is None:
        raise SamTrackError(f"SAM found no {wanted} in the clip; draw a box around the ball on the first frame")
    ns = sorted(i for i, objects in frames.items() if oid in objects)
    if len(ns) < MIN_POINTS:
        raise SamTrackError(f"SAM followed {wanted} in only {len(ns)} frames; need at least {MIN_POINTS}")
    centres = np.array([_centre(frames[i][oid]) for i in ns])
    return Track(video.timestamps[ns], centres[:, 0], centres[:, 1], "sam")
