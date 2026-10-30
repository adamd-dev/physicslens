"""Auto-detect: the ball is the moving blob whose path falls, flies or swings like a real object."""

from pathlib import Path

import cv2
import numpy as np
import pytest

import cli
from physicslens import server
from physicslens.ingest import load_video
from physicslens.physics import models
from physicslens.pipeline import TrackSettings, track
from physicslens.track.autodetect import find_ball
from tests.test_agent_gate import wandering_video
from tests.test_ball_guards import drop_with_a_bigger_mover
from tests.test_robustness import PX_PER_M
from tests.test_sam import FakeSam, drop_text
from tests.test_server import analyze, client  # noqa: F401  (client is a fixture)

BALLDROP = Path(__file__).parent.parent / "samples" / "tracker" / "mechanics_videos" / "BallDrop.mp4"


def never_called():
    raise AssertionError("SAM must not be called")


def held_drop_beside_a_hand(path, n_frames=36, fps=60, hold_frames=9):
    """The ball sits still under a twitching hand, then falls; the hand keeps twitching where it was."""
    rng = np.random.default_rng(4)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    for i in range(n_frames):
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.circle(frame, (330 + int(rng.integers(-8, 8)), 20 + int(rng.integers(-6, 6))), 20, (255, 255, 255), -1)
        y_px = 60 + PX_PER_M * models.free_fall(max(i - hold_frames, 0) / fps, 0, 0, models.G)
        cv2.circle(frame, (320, round(y_px)), 10, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def pendulum_video(path, n_frames=150, fps=60, length_m=0.5, amplitude_px=150):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    omega = np.sqrt(models.G / length_m)
    for i in range(n_frames):
        frame = np.zeros((480, 640, 3), np.uint8)
        x = 320 + amplitude_px * np.sin(omega * i / fps)
        cv2.circle(frame, (round(x), 300), 12, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def test_finds_the_ball_beside_a_bigger_mover(tmp_path):
    det = find_ball(load_video(drop_with_a_bigger_mover(tmp_path / "two.avi")), PX_PER_M)
    assert det is not None and det.model in ("free_fall", "projectile")
    cx, cy = centre(det.box)
    assert abs(cx - 320) < 6 and abs(cy - (50 + PX_PER_M * models.free_fall(det.frame / 60, 0, 0, models.G))) < 6
    assert np.all(np.abs(det.track.x - 320) < 5)


def test_the_box_is_on_the_frame_where_the_fall_starts(tmp_path):
    det = find_ball(load_video(held_drop_beside_a_hand(tmp_path / "held.avi")), PX_PER_M)
    assert det is not None and 8 <= det.frame <= 12  # released at frame 9
    cx, cy = centre(det.box)
    assert abs(cx - 320) < 6 and det.box[1] < cy < det.box[3]
    assert np.all(np.abs(det.track.x - 320) < 5)


def test_finds_a_swinging_bob(tmp_path):
    det = find_ball(load_video(pendulum_video(tmp_path / "p.avi")), PX_PER_M)
    assert det is not None and det.model == "damped_oscillation"
    assert np.all(np.abs(det.track.y - 300) < 5)


def test_nothing_ball_like_is_none(tmp_path):
    assert find_ball(load_video(wandering_video(tmp_path / "w.avi")), PX_PER_M) is None


@pytest.mark.skipif(not BALLDROP.is_file(), reason="Tracker sample clip not downloaded")
def test_finds_the_ball_in_balldrop():
    # Hand-drawn box that tracks well: 194,40,209,55 on frame 0; the hand holds the ball until about frame 5.
    det = find_ball(load_video(str(BALLDROP)), 400)
    assert det is not None and det.fields in ("top", "bottom")
    assert 190 <= centre(det.box)[0] <= 212 and np.ptp(det.track.y) > 300  # follows the fall, not the hand


@pytest.mark.skipif(not BALLDROP.is_file(), reason="Tracker sample clip not downloaded")
def test_balldrop_motion_without_any_setting_measures_g():
    from physicslens.calibrate import Calibration
    from physicslens.physics.fit import fit_free_fall
    from physicslens.pipeline import trim_rest

    _, tr, notes = track(load_video(str(BALLDROP)), TrackSettings("motion", px_per_m=400))
    tr = trim_rest(tr, 400).track
    g = fit_free_fall(tr.t - tr.t[0], Calibration.from_reference(400, 1.0).to_meters(tr.y)).params["g"]
    assert g == pytest.approx(models.G, rel=0.03) and "found automatically" in notes[0]


def test_motion_without_a_search_area_follows_the_found_ball(tmp_path):
    video = load_video(drop_with_a_bigger_mover(tmp_path / "two.avi"))
    _, tr, notes = track(video, TrackSettings("motion", px_per_m=PX_PER_M))
    assert np.all(np.abs(tr.x - 320) < 5) and "found automatically" in notes[0]


def test_a_search_area_turns_auto_detect_off(tmp_path):
    video = load_video(drop_with_a_bigger_mover(tmp_path / "two.avi"))
    _, _, notes = track(video, TrackSettings("motion", px_per_m=PX_PER_M, roi=(0, 0, 640, 480)))
    assert not any("found automatically" in n for n in notes)


def test_sam_without_a_box_uses_the_found_ball(tmp_path):
    video = load_video(held_drop_beside_a_hand(tmp_path / "held.avi"))
    sam = FakeSam(drop_text(n_frames=36, hold_frames=9))
    seen = []
    _, tr, notes = track(video, TrackSettings("sam", px_per_m=PX_PER_M), sam=lambda: sam, on_detect=seen.append)
    prompt, direction = sam.calls[0][1], sam.calls[0][2]
    assert prompt.startswith(f"<{seen[0].frame}f>1<|box;") and direction == "both"
    assert tr.source == "sam" and "found automatically" in notes[0]


def test_failed_sam_on_a_found_ball_falls_back_to_its_path(tmp_path):
    video = load_video(drop_with_a_bigger_mover(tmp_path / "two.avi"))
    _, tr, notes = track(video, TrackSettings("sam", px_per_m=PX_PER_M), sam=lambda: FakeSam(""))
    assert np.all(np.abs(tr.x - 320) < 5)
    assert any("found automatically" in n for n in notes) and any("SAM 3.1 could not track" in n for n in notes)


def test_sam_without_a_box_and_no_ball_like_motion_asks_for_a_box(tmp_path):
    with pytest.raises(ValueError, match="box around the ball"):
        track(load_video(wandering_video(tmp_path / "w.avi")), TrackSettings("sam", px_per_m=PX_PER_M), sam=never_called)


def test_api_reports_the_detection_in_source_pixels(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "make_sam_client", lambda: FakeSam(""))
    path = drop_with_a_bigger_mover(tmp_path / "two.avi")
    with open(path, "rb") as f:
        video_id = client.post("/api/videos", files={"file": ("two.avi", f, "video/x-msvideo")}).json()["video_id"]
    stream = dict(analyze(client, video_id, settings={"tracker": "sam"}, model="free_fall"))
    det = stream["detection"]
    assert abs(centre(det["box"])[0] - 320) < 6 and det["model"] in ("free_fall", "projectile")
    assert stream["result"]["params"]["g"] == pytest.approx(models.G, rel=0.03)


def test_cli_sam_without_a_box_uses_the_found_ball(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_sam_client", lambda: FakeSam(""))
    path = drop_with_a_bigger_mover(tmp_path / "two.avi")
    assert cli.main([path, "--px-per-m", str(PX_PER_M), "--tracker", "sam", "--model", "free_fall"]) == 0
    assert "found automatically" in capsys.readouterr().out
