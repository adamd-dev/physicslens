"""SAM never tracks blind: no box means no API call, and a failed box run falls back to motion at the ball."""

import cv2
import numpy as np
import pytest

import cli
from physicslens import server
from physicslens.ingest import load_video
from physicslens.physics import models
from physicslens.pipeline import TrackSettings, track
from tests.test_agent_gate import wandering_video
from tests.test_robustness import PX_PER_M
from tests.test_sam import FakeSam, drop_video
from tests.test_server import analyze, client, upload_drop  # noqa: F401  (client is a fixture)

BALL_BOX = (300, 30, 340, 70)  # around the ball of tests.test_robustness.write_drop on frame 0


def never_called():
    raise AssertionError("SAM must not be called without a box")


def drop_with_a_bigger_mover(path, n_frames=27, fps=60, hold_frames=0):
    """The ball falls at x=320 while a bigger blob (a hand, a head) circles around x=130.

    hold_frames: the ball sits still for that many frames first, so the mover is the only thing moving.
    """
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    for i in range(n_frames):
        frame = np.zeros((480, 640, 3), np.uint8)
        a = 0.25 * i
        cv2.circle(frame, (130 + round(100 * np.cos(a)), 250 + round(150 * np.sin(a))), 22, (255, 255, 255), -1)
        y_px = 50 + PX_PER_M * models.free_fall(max(i - hold_frames, 0) / fps, 0, 0, models.G)
        cv2.circle(frame, (320, round(y_px)), 12, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def test_sam_without_a_box_does_not_track_blind(tmp_path):
    with pytest.raises(ValueError, match="box around the ball"):
        track(load_video(str(drop_video(tmp_path))), TrackSettings("sam"), sam=never_called)


def test_a_failed_text_prompt_does_not_fall_back_blind(tmp_path):
    # Without a box nothing says where the ball is, so a motion fallback could follow anyone.
    with pytest.raises(ValueError, match="box around the ball"):
        track(load_video(str(drop_video(tmp_path))), TrackSettings("sam", sam_prompt="ball"), sam=lambda: FakeSam(""))


def test_unseeded_motion_follows_the_bigger_mover(tmp_path):
    # The failure this guards against: without the box, the largest moving blob wins.
    video = load_video(drop_with_a_bigger_mover(tmp_path / "two.avi"))
    _, tr, _ = track(video, TrackSettings("motion"))
    assert np.median(tr.x) < 200


@pytest.mark.parametrize("fields", ["none", "bottom"])
def test_fallback_after_a_failed_box_run_starts_at_the_ball(tmp_path, fields):
    video = load_video(drop_with_a_bigger_mover(tmp_path / "two.avi"))
    _, tr, notes = track(video, TrackSettings("sam", sam_box=BALL_BOX, fields=fields), sam=lambda: FakeSam(""))
    assert tr.source.startswith("motion") and "motion tracker" in notes[0]
    assert np.all(np.abs(tr.x - 320) < 5)


def test_fallback_waits_for_the_ball_while_only_something_else_moves(tmp_path):
    # While the ball is held still, the mover is the nearest (and only) blob: it must not start the track.
    video = load_video(drop_with_a_bigger_mover(tmp_path / "held.avi", n_frames=36, hold_frames=9))
    _, tr, _ = track(video, TrackSettings("sam", sam_box=BALL_BOX), sam=lambda: FakeSam(""))
    assert np.all(np.abs(tr.x - 320) < 5)


def test_the_box_seeds_only_the_sam_fallback(tmp_path):
    # tracker="motion" with a stray sam_box from the API keeps plain motion behaviour.
    video = load_video(drop_with_a_bigger_mover(tmp_path / "two.avi"))
    _, tr, _ = track(video, TrackSettings("motion", sam_box=BALL_BOX))
    assert np.median(tr.x) < 200


def test_api_sam_without_a_box_is_an_error_and_costs_nothing(client, tmp_path, monkeypatch):  # noqa: F811
    # Nothing in this clip moves like a ball, so auto-detect finds no box either.
    monkeypatch.setattr(server, "make_sam_client", never_called)
    with open(wandering_video(tmp_path / "w.avi"), "rb") as f:
        video_id = client.post("/api/videos", files={"file": ("w.avi", f, "video/x-msvideo")}).json()["video_id"]
    stream = dict(analyze(client, video_id, settings={"tracker": "sam"}))
    assert "result" not in stream
    assert stream["error"]["stage"] == "tracking" and "box around the ball" in stream["error"]["message"]


def test_api_default_settings_send_no_text_prompt():
    assert server.Settings().sam_prompt is None


@pytest.mark.parametrize("prompt", ["", "<0f>1<|box;x1=0;y1=0;x2=5;y2=5;w=9;h=9|>", "x" * 81])
def test_cli_rejects_a_prompt_the_api_would_reject(tmp_path, prompt, capsys):
    with pytest.raises(SystemExit):
        cli.main([str(drop_video(tmp_path)), "--px-per-m", "300", "--tracker", "sam", "--sam-prompt", prompt])
    assert "--sam-prompt must be" in capsys.readouterr().err


def test_cli_sam_without_a_box_fails_before_any_call(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_sam_client", never_called)
    assert cli.main([wandering_video(tmp_path / "w.avi"), "--px-per-m", "300", "--tracker", "sam"]) == 2
    assert "box around the ball" in capsys.readouterr().err
