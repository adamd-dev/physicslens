"""Motion tracking has a default jump limit from physics: nothing real moves faster than MAX_SPEED_M_S."""

import cv2
import numpy as np
import pytest

import cli
from physicslens import pipeline, server
from physicslens.ingest import load_video
from physicslens.physics import models
from physicslens.pipeline import TrackSettings, track
from tests.test_robustness import PX_PER_M
from tests.test_server import analyze, client  # noqa: F401  (client is a fixture)


def drop_after_a_passer_by(path, n_frames=27, fps=60, passer_frames=5):
    """A big blob crosses the bottom left for the first frames, then only the ball (x=320) moves.

    Without a limit the track starts on the passer-by and then jumps about 400 px to the ball.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    for i in range(n_frames):
        frame = np.zeros((480, 640, 3), np.uint8)
        if i < passer_frames:
            cv2.circle(frame, (40 + 25 * i, 420), 22, (255, 255, 255), -1)
        y_px = 50 + PX_PER_M * models.free_fall(i / fps, 0, 0, models.G)
        cv2.circle(frame, (320, round(y_px)), 12, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def test_without_a_scale_there_is_no_default_limit(tmp_path):
    # Documents the failure: the passer-by starts the track.
    _, tr, _ = track(load_video(drop_after_a_passer_by(tmp_path / "p.avi")), TrackSettings("motion"))
    assert tr.x.min() < 200


@pytest.mark.parametrize("fields", ["none", "bottom"])
def test_the_scale_gives_a_default_limit_that_drops_the_passer_by(tmp_path, fields):
    video = load_video(drop_after_a_passer_by(tmp_path / "p.avi"))
    _, tr, _ = track(video, TrackSettings("motion", fields=fields, px_per_m=PX_PER_M))
    assert np.all(np.abs(tr.x - 320) < 5) and len(tr) >= 20


def test_the_default_limit_follows_the_speed_and_the_sample_rate():
    # 60 fps, 300 px/m: a sample apart, MAX_SPEED_M_S covers this many pixels.
    assert pipeline.default_max_jump(PX_PER_M, 60.0, fields=False) == pytest.approx(pipeline.MAX_SPEED_M_S * PX_PER_M / 60)
    assert pipeline.default_max_jump(PX_PER_M, 60.0, fields=True) == pytest.approx(pipeline.MAX_SPEED_M_S * PX_PER_M / 120)


def test_a_real_fall_is_never_cut_by_the_default():
    # A 3 m drop ends at about 7.7 m/s: well inside the limit even at 30 fps.
    v = np.sqrt(2 * models.G * 3.0)
    assert v * PX_PER_M / 30 < pipeline.default_max_jump(PX_PER_M, 30.0, fields=False)


def test_an_explicit_max_jump_wins_over_the_default(tmp_path):
    video = load_video(drop_after_a_passer_by(tmp_path / "p.avi"))
    _, tr, _ = track(video, TrackSettings("motion", max_jump=10_000, px_per_m=PX_PER_M))
    assert tr.x.min() < 200


def test_the_api_passes_the_scale_in_decoded_pixels():
    s = server.Settings().scaled(0.5, px_per_m=400.0)
    assert s.px_per_m == pytest.approx(200.0)


def test_api_motion_run_drops_the_passer_by(client, tmp_path):  # noqa: F811
    path = drop_after_a_passer_by(tmp_path / "p.avi")
    with open(path, "rb") as f:
        video_id = client.post("/api/videos", files={"file": ("p.avi", f, "video/x-msvideo")}).json()["video_id"]
    stream = dict(analyze(client, video_id, settings={"tracker": "motion"}, model="free_fall"))
    assert stream["result"]["params"]["g"] == pytest.approx(models.G, rel=0.03)


def test_cli_motion_run_drops_the_passer_by(tmp_path, capsys):
    path = drop_after_a_passer_by(tmp_path / "p.avi")
    assert cli.main([path, "--px-per-m", str(PX_PER_M), "--model", "free_fall"]) == 0
    g = float(capsys.readouterr().out.split("g = ")[1].split()[0])
    assert g == pytest.approx(models.G, rel=0.03)
