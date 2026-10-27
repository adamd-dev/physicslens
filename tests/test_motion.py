"""Motion tracker on synthetic clips with the traps seen in real Tracker videos."""

import cv2
import numpy as np
import pytest

from physicslens.ingest import load_video
from physicslens.physics import models
from physicslens.physics.fit import fit_free_fall
from physicslens.track.motion import field_order, track_motion

FPS = 30
PX_PER_M = 400
SIZE = (320, 480)


def ball_y(t):
    return 40 + PX_PER_M * models.free_fall(t, 0, 0, models.G)


def draw(frame, y, x=160, rows=slice(None)):
    layer = frame.copy()
    cv2.circle(layer, (x, int(round(y))), 6, (255, 255, 255), -1)
    frame[rows] = layer[rows]


def write(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, SIZE)
    for f in frames:
        writer.write(f)
    writer.release()


def blank(with_stick=True):
    frame = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    if with_stick:
        frame[20:460, 240:250] = 255  # white metre stick: same color as the ball
    return frame


def g_of(track):
    return fit_free_fall(track.t - track.t[0], track.y / PX_PER_M).params["g"]


def test_static_white_stick_is_ignored(tmp_path):
    frames = []
    for i in range(10):
        f = blank()
        draw(f, ball_y(i / FPS))
        frames.append(f)
    write(tmp_path / "p.avi", frames)
    order, track = field_order(load_video(str(tmp_path / "p.avi")), min_area=40, max_area=400, max_jump_px=80)
    assert order == "none"  # progressive video must not be split into fields
    assert np.all(np.abs(track.x - 160) < 2)
    assert g_of(track) == pytest.approx(models.G, rel=0.05)


@pytest.mark.parametrize("first", ["top", "bottom"])
def test_interlaced_field_order_is_detected(tmp_path, first):
    early, late = (slice(0, None, 2), slice(1, None, 2)) if first == "top" else (slice(1, None, 2), slice(0, None, 2))
    frames = []
    for i in range(10):
        f = blank()
        draw(f, ball_y(i / FPS), rows=early)
        draw(f, ball_y((i + 0.5) / FPS), rows=late)
        frames.append(f)
    write(tmp_path / "i.avi", frames)
    order, track = field_order(load_video(str(tmp_path / "i.avi")), min_area=40, max_area=400, max_jump_px=80)
    assert order == first
    assert len(track) == 20  # two samples per frame
    assert g_of(track) == pytest.approx(models.G, rel=0.05)


def test_early_wrong_blob_loses_to_longest_segment(tmp_path):
    frames = []
    for i in range(14):
        f = blank(with_stick=False)
        if i < 3:
            draw(f, 400, x=40)  # a hand twitching in a corner before the drop
        else:
            draw(f, ball_y((i - 3) / FPS))
        frames.append(f)
    write(tmp_path / "h.avi", frames)
    track = track_motion(load_video(str(tmp_path / "h.avi")), min_area=40, max_area=400, max_jump_px=80)
    assert len(track) == 11
    assert np.all(track.x > 100)
