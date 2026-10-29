"""A track that no physics model can describe never reaches the agent, and the agent gives up early."""

import cv2
import numpy as np

import cli
from physicslens import server
from physicslens.agent.loop import NUDGE_STEP, run_agent
from physicslens.agent.tools import Toolbox, TrackData
from physicslens.llm.base import MockClient, Reply
from physicslens.physics import models
from physicslens.pipeline import wrong_object
from tests.test_agent_loop import call
from tests.test_robustness import PX_PER_M
from tests.test_server import analyze, client, upload_drop  # noqa: F401  (client is a fixture)


def never_called():
    raise AssertionError("the agent must not run on a track no model describes")


def wandering(n=27, fps=60, seed=3):
    """A hand-like random walk in metres: no fall, no swing."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    return TrackData(t, np.cumsum(rng.normal(0, 0.03, n)), np.cumsum(rng.normal(0, 0.03, n)))


def falling(n=27, fps=60):
    t = np.arange(n) / fps
    return TrackData(t, np.zeros(n), models.free_fall(t, 0, 0, models.G))


def wandering_video(path, n_frames=27, fps=60, seed=3):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (640, 480))
    x, y = 320.0, 240.0
    for _ in range(n_frames):
        x, y = x + rng.normal(0, 12), y + rng.normal(0, 12)
        frame = np.zeros((480, 640, 3), np.uint8)
        cv2.circle(frame, (round(x), round(y)), 14, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return str(path)


def test_a_wandering_track_is_the_wrong_object():
    why = wrong_object(wandering())
    assert why is not None and "followed something other than the ball" in why


def test_a_real_fall_passes_the_gate():
    assert wrong_object(falling()) is None


def test_a_flagged_but_physical_track_still_reaches_the_agent():
    # Slightly off g with noise: checks may fail, but it is still a fall the agent can reason about.
    data = falling()
    data.y = data.y * 1.04 + np.random.default_rng(0).normal(0, 0.002, len(data.t))
    assert wrong_object(data) is None


def test_slow_motion_without_capture_fps_is_caught_with_a_hint():
    # Played 8x slow: g comes out 64x too small, a fall no real object makes.
    data = falling()
    data.t = data.t * 8
    why = wrong_object(data)
    assert why is not None and "frame rate" in why


def test_the_gate_leaves_the_toolbox_alone():
    box = Toolbox(wandering())
    wrong_object(box.data)
    assert box.fits == {}  # the agent's fit ids start at fit_1


def test_api_skips_the_agent_on_the_wrong_object(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "make_client", never_called)
    path = wandering_video(tmp_path / "w.avi")
    with open(path, "rb") as f:
        video_id = client.post("/api/videos", files={"file": ("w.avi", f, "video/x-msvideo")}).json()["video_id"]
    stream = dict(analyze(client, video_id, settings={"tracker": "motion"}, agent=True))
    assert "result" not in stream and "cost" not in stream
    assert stream["error"]["stage"] == "no measurement" and stream["error"]["steps"] == 0
    assert "followed something other than the ball" in stream["error"]["message"]


def test_api_still_runs_the_agent_on_a_real_fall(client, tmp_path, monkeypatch):  # noqa: F811
    scripted = MockClient([Reply("", [call(1, "fit_free_fall"), call(2, "final_answer", fit_id="fit_1", summary="g")])])
    monkeypatch.setattr(server, "make_client", lambda: scripted)
    stream = dict(analyze(client, upload_drop(client, tmp_path), agent=True))
    assert stream["result"]["source"] == "agent"


def test_cli_skips_the_agent_on_the_wrong_object(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_client", never_called)
    path = wandering_video(tmp_path / "w.avi")
    assert cli.main([path, "--px-per-m", str(PX_PER_M), "--agent"]) == 4
    assert "followed something other than the ball" in capsys.readouterr().err


def stuck_replies(n):
    """A model that keeps refitting and never decides."""
    return [Reply("", [call(i, "fit_free_fall")]) for i in range(1, n + 1)]


def test_the_agent_is_told_to_give_up_after_nudge_step_steps():
    give_up = Reply("", [call(99, "final_answer", summary="the track is a wandering hand; box the ball")])
    client = MockClient([*stuck_replies(NUDGE_STEP), give_up])
    result = run_agent(client, Toolbox(wandering()))
    assert result.no_measurement and result.steps == NUDGE_STEP + 1
    nudges = [m for m in client.requests[-1] if m.role == "user" and "final_answer without fit_id" in (m.content or "")]
    assert len(nudges) == 1
    assert not any("final_answer without fit_id" in (m.content or "") for m in client.requests[NUDGE_STEP - 1] if m.role == "user")


def test_no_nudge_while_a_fit_passes():
    client = MockClient([*stuck_replies(NUDGE_STEP), Reply("", [call(99, "final_answer", fit_id="fit_1", summary="g")])])
    result = run_agent(client, Toolbox(falling()))
    assert result.fit_id == "fit_1"
    assert not any("final_answer without fit_id" in (m.content or "") for m in client.requests[-1] if m.role == "user")
