"""Real-video failure modes the clean synthetic tests miss."""

import cv2
import numpy as np
import pytest

from physicslens.calibrate import Calibration
from physicslens.ingest import Video, _repair, load_video
from physicslens.physics import models
from physicslens.physics.fit import fit_free_fall, fit_projectile
from physicslens.track.color import color_mask, track_color

PX_PER_M = 300
SHIFT = 4  # cv2 sub-pixel fixed point


def write_drop(path, capture_fps, playback_fps, n_frames, hidden=(), distractor=False, size=(640, 480)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), playback_fps, size)
    scale = 2**SHIFT
    for i in range(n_frames):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        if distractor:
            cv2.circle(frame, (80, 400), 7, (255, 255, 255), -1)
        if i not in hidden:
            y_px = 50 + PX_PER_M * models.free_fall(i / capture_fps, 0, 0, models.G)
            cv2.circle(frame, (320 * scale, int(round(y_px * scale))), 12 * scale, (255, 255, 255), -1, cv2.LINE_AA, SHIFT)
        writer.write(frame)
    writer.release()


def measure_g(video, **track_kwargs):
    track = track_color(video, (0, 0, 200), (180, 60, 255), **track_kwargs)
    return fit_free_fall(track.t, Calibration.from_reference(PX_PER_M, 1.0).to_meters(track.y)).params["g"]


def test_slow_motion_needs_capture_fps(tmp_path):
    path = tmp_path / "slowmo.avi"
    write_drop(path, capture_fps=240, playback_fps=30, n_frames=96)  # 0.4 s real
    assert measure_g(load_video(str(path), capture_fps=240)) == pytest.approx(models.G, rel=0.02)
    # Without the override the clock runs 8x slow and g comes out 64x too small.
    assert measure_g(load_video(str(path))) == pytest.approx(models.G / 64, rel=0.05)


def test_window_is_rezeroed(tmp_path):
    path = tmp_path / "drop.avi"
    write_drop(path, capture_fps=60, playback_fps=60, n_frames=30)
    video = load_video(str(path), start_s=0.1, end_s=0.4)
    assert video.timestamps[0] == 0.0
    assert len(video.frames) == 18  # frames 6..23 in [0.1, 0.4)


def test_hidden_ball_and_distractor_do_not_corrupt_g(tmp_path):
    path = tmp_path / "occluded.avi"
    write_drop(path, capture_fps=60, playback_fps=60, n_frames=27, hidden=range(10, 14), distractor=True)
    video = load_video(str(path))
    # Without gating the tracker would jump to the distractor while the ball is hidden.
    assert measure_g(video, max_jump_px=80) == pytest.approx(models.G, rel=0.02)


def two_blob_video():
    """A small white ball moving right, and a large white board fixed on the left."""
    frames = []
    for i in range(8):
        f = np.zeros((120, 200, 3), np.uint8)
        f[10:110, 5:45] = 255  # board: 4000 px
        cv2.circle(f, (80 + 10 * i, 60), 6, (255, 255, 255), -1)
        frames.append(f)
    return Video(frames, np.arange(8) / 30, 30.0, 200, 120, "index", 1.0)


def test_color_tracker_respects_max_area():
    track = track_color(two_blob_video(), (0, 0, 200), (180, 60, 255), max_area=1000)
    assert track.x[0] == pytest.approx(80, abs=1)


def test_color_tracker_respects_roi():
    track = track_color(two_blob_video(), (0, 0, 200), (180, 60, 255), roi=(60, 0, 200, 120))
    assert np.all(track.x > 60)


def test_red_hue_wraps_around_zero():
    frame = np.zeros((20, 20, 3), np.uint8)
    frame[5:15, 5:15] = (0, 0, 255)  # pure red in BGR, hue 0
    assert color_mask(frame, (170, 100, 100), (10, 255, 255)).sum() > 0


def test_tracker_reports_missing_object(tmp_path):
    path = tmp_path / "empty.avi"
    write_drop(path, capture_fps=60, playback_fps=60, n_frames=10, hidden=range(10))
    with pytest.raises(ValueError, match="check the HSV range"):
        track_color(load_video(str(path)), (0, 0, 200), (180, 60, 255))


def test_repair_fixes_duplicate_stamps():
    ts = np.array([0.0, 1 / 60, 1 / 60, 3 / 60, 4 / 60])
    np.testing.assert_allclose(_repair(ts), np.arange(5) / 60)


def test_projectile_r2_with_tiny_vx():
    t = np.arange(0, 0.4, 1 / 60)
    rng = np.random.default_rng(3)
    x = models.linear(t, 0.0, 0.001) + rng.normal(0, 0.001, len(t))
    y = models.free_fall(t, 0.0, 0.0, models.G) + rng.normal(0, 0.001, len(t))
    assert fit_projectile(t, x, y).r2 > 0.99
