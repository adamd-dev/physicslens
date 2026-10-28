"""Web API end to end on a synthetic drop: upload, analyze, stream events."""

import json

import pytest
from fastapi.testclient import TestClient

from physicslens import server
from physicslens.llm.base import MockClient, Reply
from physicslens.physics import models
from tests.test_robustness import PX_PER_M, write_drop

SETTINGS = {"tracker": "color"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(server, "COSTS", tmp_path / "costs.csv")
    return TestClient(server.app)


def upload_drop(client, tmp_path) -> str:
    path = tmp_path / "drop.avi"
    write_drop(path, capture_fps=60, playback_fps=60, n_frames=27)
    with path.open("rb") as f:
        response = client.post("/api/videos", files={"file": ("drop.avi", f, "video/x-msvideo")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert (body["width"], body["height"], body["frames"]) == (640, 480, 27)
    return body["video_id"]


def events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def analyze(client, video_id, **extra):
    body = {"video_id": video_id, "scale": {"px_per_m": PX_PER_M}, "settings": SETTINGS, **extra}
    response = client.post("/api/analyze", json=body)
    assert response.status_code == 200, response.text
    return events(response)


def test_analyze_streams_track_fits_and_result(client, tmp_path):
    stream = analyze(client, upload_drop(client, tmp_path))
    names = [name for name, _ in stream]
    assert [n for n in names if n != "status"] == ["video", "track", "fits", "result", "done"]
    track = dict(stream)["track"]
    assert track["x_px"][0] == pytest.approx(320, abs=1)  # source pixels, for the overlay
    result = dict(stream)["result"]
    assert result["model"] == "free_fall" and result["source"] == "deterministic"
    assert result["params"]["g"] == pytest.approx(models.G, rel=0.02)
    assert len(result["curve"]["t_play"]) == len(result["curve"]["y_px"]) == len(track["t"])


def test_calibration_from_two_points(client, tmp_path):
    scale = {"p1": [0, 0], "p2": [0, PX_PER_M], "length_m": 1.0}
    response = client.post("/api/analyze", json={"video_id": upload_drop(client, tmp_path), "scale": scale, "settings": SETTINGS})
    assert dict(events(response))["result"]["params"]["g"] == pytest.approx(models.G, rel=0.02)


def upload_held_drop(client, tmp_path) -> str:
    path = tmp_path / "held.avi"
    write_drop(path, capture_fps=60, playback_fps=60, n_frames=39, hold_frames=12)
    with path.open("rb") as f:
        return client.post("/api/videos", files={"file": ("held.avi", f, "video/x-msvideo")}).json()["video_id"]


def test_ball_held_before_release_is_dropped_and_shown(client, tmp_path):
    stream = analyze(client, upload_held_drop(client, tmp_path))
    track = dict(stream)["track"]
    assert "before the motion" in track["notes"][0]
    rest = track["rest"]  # still drawn on the video, in grey
    assert len(rest["t_play"]) >= 12 and len(rest["x_px"]) == len(rest["y_px"]) == len(rest["t_play"])
    assert rest["t_play"][-1] < track["t_play"][0]
    result = dict(stream)["result"]
    assert result["params"]["g"] == pytest.approx(models.G, rel=0.02) and result["check"]["ok"]
    assert result["curve"]["t_play"][0] == pytest.approx(track["t_play"][0])


def test_release_detection_can_be_turned_off(client, tmp_path):
    stream = analyze(client, upload_held_drop(client, tmp_path), detect_release=False)
    track = dict(stream)["track"]
    assert track["notes"] == [] and track["rest"]["t_play"] == []
    assert len(track["t"]) == 39


def test_agent_is_told_about_the_dropped_samples(client, tmp_path, monkeypatch):
    scripted = MockClient([Reply("", [call(1, "final_answer", summary="test run, no measurement wanted here")])])
    monkeypatch.setattr(server, "make_client", lambda: scripted)
    analyze(client, upload_held_drop(client, tmp_path), agent=True)
    question = scripted.requests[0][1].content
    assert "at rest" in question and "already removed" in question


def call(i, name, **arguments):
    return {"id": f"c{i}", "name": name, "arguments": json.dumps(arguments)}


def test_agent_messages_stream_live(client, tmp_path, monkeypatch):
    scripted = MockClient([
        Reply("Looks like a drop.", [call(1, "fit_free_fall")]),
        Reply("", [call(2, "final_answer", fit_id="fit_1", summary="g is about 9.8")]),
    ])
    monkeypatch.setattr(server, "make_client", lambda: scripted)
    stream = analyze(client, upload_drop(client, tmp_path), agent=True)
    roles = [data["role"] for name, data in stream if name == "agent_message"]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool"]
    result = dict(stream)["result"]
    assert result["source"] == "agent" and result["summary"] == "g is about 9.8" and result["steps"] == 2
    assert dict(stream)["cost"]["run_usd"] == 0


def test_agent_without_answer_ends_with_error(client, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "make_client", lambda: MockClient([Reply("thinking")] * 12))
    stream = analyze(client, upload_drop(client, tmp_path), agent=True)
    assert dict(stream)["error"]["stage"] == "agent"
    assert stream[-1][0] == "done"


def test_agent_can_report_no_measurement(client, tmp_path, monkeypatch):
    reason = "only 3 points tracked; draw a search area around the ball"
    monkeypatch.setattr(server, "make_client", lambda: MockClient([Reply("", [call(1, "final_answer", summary=reason)])]))
    stream = analyze(client, upload_drop(client, tmp_path), agent=True)
    error = dict(stream)["error"]
    assert error["stage"] == "no measurement" and error["message"] == reason
    assert "result" not in dict(stream) and stream[-1][0] == "done"


def test_tracking_failure_is_an_error_event(client, tmp_path):
    stream = analyze(client, upload_drop(client, tmp_path), settings={"tracker": "color", "hsv_lower": [0, 200, 0], "hsv_upper": [10, 255, 50]})
    assert dict(stream)["error"]["stage"] == "tracking"


@pytest.mark.parametrize("video_id", ["0" * 32, "../../etc/passwd", "abc"])
def test_unknown_or_malicious_ids_are_404(client, video_id):
    assert client.get(f"/api/videos/{video_id}/frame").status_code == 404
    body = {"video_id": video_id, "scale": {"px_per_m": 100}}
    assert client.post("/api/analyze", json=body).status_code == 404


def test_upload_rejects_wrong_type_and_oversize(client, tmp_path, monkeypatch):
    assert client.post("/api/videos", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 415
    assert client.post("/api/videos", files={"file": ("fake.mp4", b"not a video", "video/mp4")}).status_code == 415
    monkeypatch.setattr(server, "MAX_UPLOAD_BYTES", 10)
    assert client.post("/api/videos", files={"file": ("big.mp4", b"x" * 100, "video/mp4")}).status_code == 413
    assert not any((tmp_path / "uploads").iterdir())  # rejected files are not kept


def test_frame_endpoint_returns_jpeg(client, tmp_path):
    response = client.get(f"/api/videos/{upload_drop(client, tmp_path)}/frame", params={"t": 0.1})
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"
