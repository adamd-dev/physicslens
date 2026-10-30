"""Find the ball without a box: the moving blob whose path falls, flies or swings like a real object.

Every moving blob in every frame (or field) is a candidate. Candidates are linked into paths by
predicting each path's next position from its last three samples (constant acceleration), so a
falling ball and a twitching hand beside it stay separate paths. Each long enough path is fitted
with the same models the agent uses. Of the paths a real object could have made, with a clean fit,
the one that travels farthest wins: a ball crosses the frame, a hand only twitches.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..agent.tools import MIN_R2, Toolbox, TrackData, could_be_physics
from ..ingest import Video
from .base import MAX_SPEED_M_S, Track
from .motion import _samples, is_interlaced, moving_masks

MIN_PATH = 8  # samples; fewer cannot tell a fall from a hand
MAX_GAP_S = 0.25  # a path may go unseen this long (occlusion, a merged blob) before it ends
NEW_PATH_GAP = 2  # samples a one-sample path waits for its second blob: flicker must not grow paths
LINK_PX = 8.0  # how far a blob may be from a path's prediction, plus a share of the predicted step
LINK_SHARE = 0.5
MAX_BLOBS = 20  # largest blobs per sample: bounds the linking cost on a busy clip
BOX_PAD_PX = 2
WHOLE_SHARE = 0.8  # of the path's median box area: smaller first blobs are crescents, not the whole ball
MIN_EXTENT_BOXES = 3  # a path must travel at least this many ball sizes to be worth fitting
MAX_PATHS_FITTED = 10  # farthest first: bounds the fitting time on a busy clip
MAX_SWING_FITS = 3  # paths that get the slow pendulum fits when no fall or throw fits them
FIELDS = ("auto", "none", "top", "bottom")
# Cheap ballistic fits first: a pendulum fit on a falling path can take a minute and never wins.
_BALLISTIC: tuple[tuple[str, dict[str, str]], ...] = (("fit_free_fall", {}), ("fit_projectile", {}))
_SWING: tuple[tuple[str, dict[str, str]], ...] = (("fit_pendulum", {"axis": "x"}), ("fit_pendulum", {"axis": "y"}))


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]  # around the ball on `frame`, full-frame pixels
    frame: int  # index, in the loaded window, of the frame the box is on: where the path starts
    track: Track  # the path in full-frame pixels, usable as the motion track
    fields: str  # "none" | "top" | "bottom"
    model: str  # the model that fits the path
    r2: float  # how well that model fits the path (not a confidence that it is the ball)


@dataclass
class _Path:
    t: list[float]
    x: list[float]
    y: list[float]
    boxes: list[tuple[int, int, int, int]]
    frames: list[int]

    def predict(self, t: float) -> tuple[float, float, float]:
        """(x, y, predicted step length) at time t; the step is infinite when there is no motion to extend."""
        if len(self.t) >= 3:
            t0, t1, t2 = self.t[-3:]
            if t0 < t1 < t2:
                # Quadratic through the last three samples (Lagrange form, no fitting).
                w0 = (t - t1) * (t - t2) / ((t0 - t1) * (t0 - t2))
                w1 = (t - t0) * (t - t2) / ((t1 - t0) * (t1 - t2))
                w2 = (t - t0) * (t - t1) / ((t2 - t0) * (t2 - t1))
                px = w0 * self.x[-3] + w1 * self.x[-2] + w2 * self.x[-1]
                py = w0 * self.y[-3] + w1 * self.y[-2] + w2 * self.y[-1]
                return px, py, math.hypot(px - self.x[-1], py - self.y[-1])
        if len(self.t) >= 2 and self.t[-1] > self.t[-2]:
            k = (t - self.t[-1]) / (self.t[-1] - self.t[-2])
            px, py = self.x[-1] + k * (self.x[-1] - self.x[-2]), self.y[-1] + k * (self.y[-1] - self.y[-2])
            return px, py, math.hypot(px - self.x[-1], py - self.y[-1])
        return self.x[-1], self.y[-1], math.inf


def _blobs(mask: np.ndarray, y_scale: int, y_offset: int, min_area: float, max_area: float):
    """(x, y, box) of the largest blobs in full-frame pixels; areas are given in full-frame pixels."""
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    found = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] * y_scale
        if not min_area <= area <= max_area:
            continue
        x0, y0, w, h = (int(v) for v in stats[i, :4])
        box = (x0, y0 * y_scale + y_offset, x0 + w - 1, (y0 + h - 1) * y_scale + y_offset)
        found.append((area, float(centroids[i][0]), float(centroids[i][1]) * y_scale + y_offset, box))
    found.sort(key=lambda b: -b[0])
    return [(x, y, box) for _, x, y, box in found[:MAX_BLOBS]]


def _paths(video: Video, moving: np.ndarray, fields: str | None, px_per_m: float, min_area: float, max_area: float) -> list[_Path]:
    per_frame = 1 if fields is None else 2
    sample_dt = 1.0 / (video.fps * per_frame)
    live: list[_Path] = []
    done: list[_Path] = []
    for k, (t, mask, y_scale, y_offset) in enumerate(_samples(video, fields, 0, moving)):
        # End paths unseen for too long; a one-sample path gets only NEW_PATH_GAP samples.
        still = []
        for p in live:
            limit = NEW_PATH_GAP * sample_dt if len(p.t) == 1 else MAX_GAP_S
            if t - p.t[-1] <= limit + 1e-9:
                still.append(p)
            elif len(p.t) >= MIN_PATH:
                done.append(p)
        live = still
        blobs = _blobs(mask, y_scale, y_offset, min_area, max_area)
        used_blobs: set[int] = set()
        if live and blobs:
            pred = np.array([p.predict(t) for p in live])
            last = np.array([p.t[-1] for p in live])
            xy = np.array([(bx, by) for bx, by, _ in blobs])
            dist = np.hypot(pred[:, None, 0] - xy[None, :, 0], pred[:, None, 1] - xy[None, :, 1])
            fresh = np.isinf(pred[:, 2])
            step = np.where(fresh, 0.0, pred[:, 2])
            gate = np.where(fresh, LINK_PX + MAX_SPEED_M_S * px_per_m * (t - last), LINK_PX + LINK_SHARE * step)
            ii, jj = np.nonzero(dist <= gate[:, None])
            # Closest pairs first, each path and blob used once.
            used_paths: set[int] = set()
            for n in np.argsort(dist[ii, jj], kind="stable"):
                i, j = int(ii[n]), int(jj[n])
                if i in used_paths or j in used_blobs:
                    continue
                used_paths.add(i)
                used_blobs.add(j)
                p, (bx, by, box) = live[i], blobs[j]
                p.t.append(t)
                p.x.append(bx)
                p.y.append(by)
                p.boxes.append(box)
                p.frames.append(k // per_frame)
        for j, (bx, by, box) in enumerate(blobs):
            if j not in used_blobs:
                live.append(_Path([t], [bx], [by], [box], [k // per_frame]))
    return done + [p for p in live if len(p.t) >= MIN_PATH]


def _extent(p: _Path) -> float:
    return float(np.hypot(np.ptp(p.x), np.ptp(p.y)))


def _size(p: _Path) -> float:
    """Median blob width in pixels: the ball's size, or a hand's."""
    return float(np.median([b[2] - b[0] + 1 for b in p.boxes]))


def _first_whole(p: _Path) -> int:
    """Index of the first sample whose blob is the ball's full size.

    Where a ball was held still is masked as flicker, so just after release only a crescent of it
    moves: a box around that crescent misses the ball's centre.
    """
    areas = [(b[2] - b[0] + 1) * (b[3] - b[1] + 1) for b in p.boxes]
    full = WHOLE_SHARE * float(np.median(areas))
    return next((i for i, a in enumerate(areas) if a >= full), 0)


def _score(p: _Path, px_per_m: float, swing: bool) -> tuple[str, float] | None:
    """(model, R²) of the best physically possible fit of the path, or None if no real object moves so.

    `swing`: also try the (slow) pendulum fits when no fall or throw fits.
    """
    t = np.array(p.t)
    box = Toolbox(TrackData(t - t[0], np.array(p.x) / px_per_m, np.array(p.y) / px_per_m))
    best = None
    for fits in (_BALLISTIC, _SWING) if swing else (_BALLISTIC,):
        for name, kwargs in fits:
            summary = box.call(name, kwargs)
            if "error" in summary or not could_be_physics(summary) or summary["r2"] < MIN_R2:
                continue
            if best is None or summary["r2"] > best[1]:
                best = (summary["model"], float(summary["r2"]))
        if best is not None:
            return best
    return None


def _detect(video: Video, moving: np.ndarray, fields: str | None, px_per_m: float, min_area: float, max_area: float) -> Detection | None:
    paths = [p for p in _paths(video, moving, fields, px_per_m, min_area, max_area) if _extent(p) >= MIN_EXTENT_BOXES * _size(p)]
    paths.sort(key=_extent, reverse=True)
    # Swings only in whole frames: split fields of a progressive clip break into fragments, and each
    # fragment would get the slow pendulum fits. A swing is found in whole frames as well.
    swings_left = MAX_SWING_FITS if fields is None else 0
    for p in paths[:MAX_PATHS_FITTED]:
        # Farthest first: the first path that fits is the one that travels farthest.
        scored = _score(p, px_per_m, swing=swings_left > 0)
        swings_left -= 1
        if scored is None:
            continue
        model, r2 = scored
        h, w = video.frames[0].shape[:2]
        i = _first_whole(p)
        x0, y0, x1, y1 = p.boxes[i]
        box = (max(0, x0 - BOX_PAD_PX), max(0, y0 - BOX_PAD_PX), min(w - 1, x1 + BOX_PAD_PX), min(h - 1, y1 + BOX_PAD_PX))
        source = "motion" if fields is None else f"motion-{fields}"
        return Detection(box, p.frames[i], Track(np.array(p.t), np.array(p.x), np.array(p.y), source=source), fields or "none", model, r2)
    return None


def _track_extent(tr: Track) -> float:
    return float(np.hypot(np.ptp(tr.x), np.ptp(tr.y)))


def find_ball(
    video: Video,
    px_per_m: float,
    fields: str = "auto",
    min_area: float = 10,
    max_area: float = 2000,
    threshold: int = 40,
    roi: tuple[int, int, int, int] | None = None,
) -> Detection | None:
    """The ball's path and a box around it where it starts moving; None when nothing moves like a real object.

    `px_per_m` is needed to tell a fall (g near 9.81) from a hand. `fields`: "auto" also tries both field
    orders and keeps a field path only if the video is interlaced. `roi` limits the search to an area.
    """
    if fields not in FIELDS:
        raise ValueError(f"fields must be one of {FIELDS}")
    if not px_per_m > 0:
        raise ValueError("px_per_m must be positive")
    moving = moving_masks(video, threshold)  # once: the median background is the expensive part
    if roi is not None:
        x0, y0, x1, y1 = roi
        inside = np.zeros(moving.shape[1:], np.uint8)
        inside[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1] = 1
        moving &= inside
    if fields != "auto":
        return _detect(video, moving, None if fields == "none" else fields, px_per_m, min_area, max_area)
    # The wrong field order zig-zags; its path can still pass as a swing, so a field path must also look
    # interlaced. Fields or whole frames: the path that travels farthest wins (a flickering edge in one
    # field must not beat the throw seen in whole frames).
    split = [d for d in (_detect(video, moving, order, px_per_m, min_area, max_area) for order in ("top", "bottom"))
             if d and is_interlaced(d.track, 1.0 / video.fps)]
    frames = _detect(video, moving, None, px_per_m, min_area, max_area)
    found = [*split, *([frames] if frames else [])]
    return max(found, key=lambda d: (_track_extent(d.track), d.r2)) if found else None
