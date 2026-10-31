"""Edited clips slowed down by repeating each frame: the copies carry no motion and stall the motion tracker."""

import warnings

import cv2
import numpy as np
import pytest

from physicslens.calibrate import Calibration
from physicslens.ingest import load_video
from physicslens.physics import models
from physicslens.physics.fit import fit_free_fall
from physicslens.track.motion import track_motion
from tests.test_robustness import PX_PER_M, write_drop


def write_slowed_drop(path, real_fps=25, repeat=5, n_real=12):
    """A drop filmed at `real_fps`, slowed `repeat` times: every frame shown `repeat` times, played at `real_fps`."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), real_fps, (640, 480))
    for i in range(n_real):
        frame = np.zeros((480, 640, 3), np.uint8)
        y = 40 + PX_PER_M * models.free_fall(i / real_fps, 0, 0, models.G)
        cv2.circle(frame, (320, round(y)), 10, (255, 255, 255), -1)
        for _ in range(repeat):
            writer.write(frame)
    writer.release()
    return str(path)


def test_repeated_frames_are_dropped_and_said(tmp_path):
    path = write_slowed_drop(tmp_path / "slowed.avi")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Played at 25 fps, each frame 5 times: 5x slow motion, so the real capture rate is 125 fps.
        video = load_video(path, capture_fps=125)
    assert len(video.frames) == 12
    assert np.allclose(np.diff(video.timestamps), 1 / 25, atol=1e-3)  # the real frame interval
    assert any("repeated" in str(w.message) for w in caught)


def test_a_slowed_edit_then_gives_the_right_g(tmp_path):
    video = load_video(write_slowed_drop(tmp_path / "slowed.avi"), capture_fps=125)
    tr = track_motion(video, min_area=50)
    g = fit_free_fall(tr.t, Calibration.from_reference(PX_PER_M, 1.0).to_meters(tr.y)).params["g"]
    assert g == pytest.approx(models.G, rel=0.03)


def test_an_ordinary_clip_keeps_every_frame(tmp_path):
    write_drop(tmp_path / "d.avi", 60, 60, 27, hold_frames=9)  # a held ball repeats only a few frames
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        video = load_video(str(tmp_path / "d.avi"))
    assert len(video.frames) == 27 and not any("repeated" in str(w.message) for w in caught)


def test_a_still_clip_is_not_mistaken_for_a_slowed_one(tmp_path):
    path = tmp_path / "still.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (640, 480))
    for _ in range(9):
        writer.write(np.zeros((480, 640, 3), np.uint8))
    writer.release()
    assert len(load_video(str(path)).frames) == 9
