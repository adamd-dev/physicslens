"""End to end on a synthetic video: draw a falling ball, track it, measure g."""

import cv2
import numpy as np
import pytest

from physicslens.calibrate import Calibration
from physicslens.ingest import load_video
from physicslens.physics import models
from physicslens.physics.fit import fit_free_fall
from physicslens.pipeline import choose
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
        center = (int(320 * scale), round(y_px * scale))
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


def test_choose_falls_back_to_best_r2_not_smallest_rmse():
    # A 1 m drop that fails a check vs a pendulum fit on a few mm of sideways wobble.
    flagged = {"ok": False, "problems": ["g outside plausible"]}
    results = [
        {"name": "free_fall", "fit_id": "fit_1", "r2": 0.998, "rmse_m": 0.0147, "check": flagged},
        {"name": "pendulum", "fit_id": "fit_2", "r2": 0.982, "rmse_m": 0.0005, "check": flagged},
        {"name": "projectile", "fit_id": "fit_3", "r2": None, "rmse_m": 0.0001, "check": flagged},
    ]
    assert choose(results)["name"] == "free_fall"


def test_choose_prefers_fewest_failed_checks():
    # BallTossOut: a pendulum with a 63 s period fits the straight x motion with R² 0.99999,
    # but fails four checks; the projectile fails only the runs test (perspective).
    results = [
        {"name": "free_fall", "fit_id": "fit_1", "r2": 0.99997, "rmse_m": 0.00135,
         "check": {"ok": False, "problems": ["runs", "sideways"]}},
        {"name": "projectile", "fit_id": "fit_2", "r2": 0.99993, "rmse_m": 0.00257,
         "check": {"ok": False, "problems": ["runs"]}},
        {"name": "pendulum", "fit_id": "fit_3", "r2": 0.99999, "rmse_m": 0.0012,
         "check": {"ok": False, "problems": ["runs", "cycles", "extrapolated", "length"]}},
    ]
    assert choose(results)["name"] == "projectile"
