"""End to end on a synthetic video: draw a falling ball, track it, measure g."""

import cv2
import numpy as np
import pytest

from physicslens.calibrate import Calibration
from physicslens.ingest import load_video
from physicslens.physics import models
from physicslens.physics.fit import fit_free_fall
from physicslens.track.color import track_color

FPS = 60
PX_PER_M = 300
SUBPIXEL_SHIFT = 4  # cv2 fixed-point coordinates: value * 2**shift


def write_falling_ball(path, duration_s=0.45, size=(640, 480)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, size)
    scale = 2**SUBPIXEL_SHIFT
    for t in np.arange(0, duration_s, 1 / FPS):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        y_px = 50 + PX_PER_M * models.free_fall(t, 0, 0, models.G)
        center = (int(320 * scale), int(round(y_px * scale)))
        cv2.circle(frame, center, 12 * scale, (255, 255, 255), -1, cv2.LINE_AA, SUBPIXEL_SHIFT)
        writer.write(frame)
    writer.release()


def test_synthetic_drop_measures_g(tmp_path):
    path = tmp_path / "drop.avi"
    write_falling_ball(path)

    video = load_video(str(path))
    track = track_color(video, hsv_lower=(0, 0, 200), hsv_upper=(180, 60, 255))
    assert len(track) == len(video.frames)

    calibration = Calibration.from_reference(PX_PER_M, 1.0)
    result = fit_free_fall(track.t, calibration.to_meters(track.y))
    assert result.params["g"] == pytest.approx(models.G, rel=0.02)


def test_calibration_from_points():
    calibration = Calibration.from_points((0, 0), (300, 400), length_m=0.5)
    assert calibration.px_per_m == pytest.approx(1000)
    assert calibration.to_meters([1000])[0] == pytest.approx(1.0)


def test_calibration_rejects_nonpositive():
    with pytest.raises(ValueError):
        Calibration.from_reference(0, 1.0)


def test_load_video_missing_file():
    with pytest.raises(FileNotFoundError):
        load_video("does-not-exist.mp4")
