"""Motion tracking has a default jump limit from physics: nothing real moves faster than MAX_SPEED_M_S."""

import cv2
import numpy as np
import pytest

import cli
from physicslens import server
from physicslens.ingest import load_video
from physicslens.physics import models
from physicslens.pipeline import TrackSettings, track
from tests.test_robustness import PX_PER_M, write_drop
from tests.test_server import analyze, client  # noqa: F401  (client is a fixture)


def drop_after_a_passer_by(path, n_frames=27, fps=60, passer_frames=5, hidden=()):
    """A big blob crosses the bottom left for the first frames, then only the ball (x=320) moves.

    Without a limit the track starts on the passer-by and then jumps about 400 px to the ball.
    hidden: frames where the ball is not drawn.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    for i in range(n_frames):
        frame = np.zeros((480, 640, 3), np.uint8)
        if i < passer_frames:
            cv2.circle(frame, (40 + 25 * i, 420), 22, (255, 255, 255), -1)
        y_px = 50 + PX_PER_M * models.free_fall(i / fps, 0, 0, models.G)
        if i not in hidden:
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


def test_the_default_drops_the_passer_by_and_keeps_a_ball_hidden_for_a_while(tmp_path):
    # Hidden for 6 frames (behind a hand, out of the light): a speed limit, unlike a fixed jump
    # and gap limit, allows the longer step after the gap and keeps both parts of the fall.
    video = load_video(drop_after_a_passer_by(tmp_path / "h.avi", hidden=range(10, 16)))
    _, tr, _ = track(video, TrackSettings("motion", px_per_m=PX_PER_M))
    # Auto-detect (scale known, no search area) follows the ball from frame 0, across the gap.
    assert np.all(np.abs(tr.x - 320) < 5) and len(tr) == 21


def test_an_explicit_max_jump_wins_over_the_default(tmp_path):
    # The explicit limit keeps its time-gap rule, so it cuts the hidden fall where the default does not.
    path = tmp_path / "hidden.avi"
    write_drop(path, capture_fps=60, playback_fps=60, n_frames=27, hidden=range(10, 16))
    video = load_video(str(path))
    _, by_default, _ = track(video, TrackSettings("motion", px_per_m=PX_PER_M))
    _, explicit, _ = track(video, TrackSettings("motion", max_jump=10_000, px_per_m=PX_PER_M))
    assert (len(by_default), len(explicit)) == (21, 11)


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
