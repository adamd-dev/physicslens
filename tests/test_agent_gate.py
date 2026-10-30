"""A track that no physics model can describe never reaches the agent, and the agent gives up early."""

import itertools

import cv2
import numpy as np
import pytest

import cli
from physicslens import pipeline, server
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
    """A tracked drop: 0.5 mm of tracking noise, as a real tracker gives."""
    t = np.arange(n) / fps
    noise = np.random.default_rng(1).normal(0, 0.0005, (2, n))
    return TrackData(t, noise[0], models.free_fall(t, 0, 0, models.G) + noise[1])


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


def test_a_fit_without_a_finite_g_does_not_crash_the_gate(monkeypatch):
    # Summaries turn a non-finite g into None; the gate must still answer, and such a fit is no physics.
    bad = {"name": "free_fall", "fit_id": "fit_1", "r2": 0.95, "params": {"g": None}, "check": {"ok": False, "problems": []}}
    monkeypatch.setattr(pipeline, "run_fits", lambda *a, **k: [bad])
    why = wrong_object(falling())
    assert why is not None and "free_fall R² 0.95)" in why


def test_a_swing_along_y_reaches_the_agent():
    t = np.arange(240) / 60
    theta = 0.2 * np.cos(np.sqrt(models.G / 0.5) * t)
    noise = np.random.default_rng(2).normal(0, 0.0005, len(t))
    data = TrackData(t, np.zeros(len(t)) + noise, 0.5 * np.sin(theta) + noise)  # x and y swapped, as a rotated phone films it
    assert wrong_object(data) is None


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


def test_api_without_the_agent_gives_no_number_for_the_wrong_object(client, tmp_path):  # noqa: F811
    # The least-bad fit of a hand's path is still a hand: no result, the same answer as with the agent.
    path = wandering_video(tmp_path / "w.avi")
    with open(path, "rb") as f:
        video_id = client.post("/api/videos", files={"file": ("w.avi", f, "video/x-msvideo")}).json()["video_id"]
    stream = dict(analyze(client, video_id, settings={"tracker": "motion"}))
    assert "result" not in stream
    assert stream["error"]["stage"] == "no measurement" and "followed something other than the ball" in stream["error"]["message"]


def test_cli_without_the_agent_gives_no_number_for_the_wrong_object(tmp_path, capsys):
    path = wandering_video(tmp_path / "w.avi")
    assert cli.main([path, "--px-per-m", str(PX_PER_M)]) == 4
    out = capsys.readouterr()
    assert "followed something other than the ball" in out.err and "g =" not in out.out


def test_cli_json_says_why_there_is_no_measurement(tmp_path, capsys):
    import json

    path = wandering_video(tmp_path / "w.avi")
    assert cli.main([path, "--px-per-m", str(PX_PER_M), "--json"]) == 4
    report = json.loads(capsys.readouterr().out)
    assert report["best"] is None and "followed something other than the ball" in report["no_measurement"]


def test_no_fit_at_all_is_no_measurement_not_a_pass():
    # Too few samples for any model: the agent must not be paid to find that out.
    why = wrong_object(wandering(n=5))
    assert why is not None and "no model could be fitted" in why


@pytest.mark.parametrize(("model", "axis", "reused"), [("auto", "x", True), ("auto", "y", False), ("free_fall", "x", False)])
def test_only_full_auto_fits_are_reused(model, axis, reused):
    results = [{"name": "free_fall"}]
    assert (pipeline.reusable(results, model, axis) is results) is reused


def test_a_forced_model_on_a_real_swing_is_flagged_not_blocked(tmp_path, capsys):
    # The gate asks whether anything physical moves, not whether the forced model fits it.
    from tests.test_autodetect import pendulum_video

    path = pendulum_video(tmp_path / "p.avi")
    assert cli.main([path, "--px-per-m", str(PX_PER_M), "--model", "free_fall"]) == 1


def test_the_gate_reuses_fits_it_is_given(monkeypatch):
    box = Toolbox(falling())
    results = pipeline.run_fits(box)
    calls = []
    original = Toolbox.call
    monkeypatch.setattr(Toolbox, "call", lambda self, name, args: calls.append((name, args.get("axis"))) or original(self, name, args))
    assert wrong_object(falling(), results=results) is None
    assert calls == [("fit_pendulum", "y")]  # only the fit run_fits does not make


def stuck_replies(n):
    """A model that keeps refitting and never decides."""
    return [Reply("", [call(i, "fit_free_fall")]) for i in range(1, n + 1)]


def test_the_agent_is_told_to_give_up_after_nudge_step_steps():
    give_up = Reply("", [call(99, "final_answer", summary="the track is a wandering hand; box the ball")])
    llm = MockClient([*stuck_replies(NUDGE_STEP), give_up])
    result = run_agent(llm, Toolbox(wandering()))
    assert result.no_measurement and result.steps == NUDGE_STEP + 1
    nudges = [m for m in llm.requests[-1] if m.role == "user" and "final_answer without fit_id" in (m.content or "")]
    assert len(nudges) == 1
    assert not any("final_answer without fit_id" in (m.content or "") for m in llm.requests[NUDGE_STEP - 1] if m.role == "user")


def test_no_nudge_before_any_fit_was_tried():
    looks = [Reply("", [call(i, "describe_track")]) for i in range(1, NUDGE_STEP + 1)]
    llm = MockClient([*looks, Reply("", [call(99, "final_answer", summary="the track is a wandering hand; box the ball")])])
    run_agent(llm, Toolbox(wandering()))
    assert not any("final_answer without fit_id" in (m.content or "") for m in llm.requests[-1] if m.role == "user")


def test_the_nudge_joins_a_pending_user_turn():
    # A text-only reply at step 6 adds "Continue ..."; the nudge must not make two user turns in a row.
    llm = MockClient([*stuck_replies(NUDGE_STEP - 1), Reply("thinking", []),
                      Reply("", [call(99, "final_answer", summary="the track is a wandering hand; box the ball")])])
    run_agent(llm, Toolbox(wandering()))
    roles = [m.role for m in llm.requests[-1]]
    assert not any(a == b == "user" for a, b in itertools.pairwise(roles))
    assert "final_answer without fit_id" in llm.requests[-1][-1].content


def test_no_nudge_while_a_fit_passes():
    llm = MockClient([*stuck_replies(NUDGE_STEP), Reply("", [call(99, "final_answer", fit_id="fit_1", summary="g")])])
    result = run_agent(llm, Toolbox(falling()))
    assert result.fit_id == "fit_1"
    assert not any("final_answer without fit_id" in (m.content or "") for m in llm.requests[-1] if m.role == "user")
