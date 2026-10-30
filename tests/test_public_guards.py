"""A public site stays usable for everyone: no stranger can wipe History, stall the one run slot, flood it, or
break the demo by using up the budget."""

import threading
import time

import pytest

from physicslens import server
from physicslens.llm.meta_client import BudgetExceededError
from tests.test_server import analyze, client, upload_drop  # noqa: F401  (client is a fixture)


@pytest.fixture
def public(monkeypatch):
    monkeypatch.setattr(server, "PUBLIC", True)


def test_on_a_public_site_only_the_admin_deletes_runs(client, tmp_path, public, monkeypatch):  # noqa: F811
    run_id = analyze(client, upload_drop(client, tmp_path))[0][1]["id"]
    assert client.get("/api/config").json()["can_delete"] is False
    assert client.delete(f"/api/runs/{run_id}").status_code == 403
    monkeypatch.setattr(server, "ADMIN_TOKEN", "s3cret-token")
    assert client.delete(f"/api/runs/{run_id}", headers={"X-Admin-Token": "wrong"}).status_code == 403
    assert client.delete(f"/api/runs/{run_id}", headers={"X-Admin-Token": "s3cret-token"}).status_code == 204


def test_locally_anyone_deletes(client, tmp_path):  # noqa: F811
    run_id = analyze(client, upload_drop(client, tmp_path))[0][1]["id"]
    assert client.get("/api/config").json()["can_delete"] is True
    assert client.delete(f"/api/runs/{run_id}").status_code == 204


def test_a_stopped_run_ends_and_frees_the_slot(client, tmp_path, monkeypatch):  # noqa: F811
    def slow(req, emit):
        for _ in range(200):  # 10 s unless cancelled
            emit("status", {"stage": "tracking"})
            time.sleep(0.05)

    monkeypatch.setattr(server, "_analyze", slow)
    video = upload_drop(client, tmp_path)

    def stop_when_running():
        for _ in range(100):
            if server._cancels:
                run_id = next(iter(server._cancels))
                assert client.post(f"/api/runs/{run_id}/cancel").status_code == 202
                return
            time.sleep(0.02)

    stopper = threading.Thread(target=stop_when_running)
    stopper.start()
    started = time.monotonic()
    stream = analyze(client, video)
    stopper.join()
    assert time.monotonic() - started < 5
    assert stream[-2] == ("error", {"stage": "cancelled", "message": "stopped before it finished"})
    assert stream[-1][0] == "done" and not server._cancels
    monkeypatch.setattr(server, "_analyze", lambda req, emit: None)
    assert analyze(client, video)[-1][0] == "done"  # the slot is free again


def test_cancelling_an_unknown_run_is_404(client):  # noqa: F811
    assert client.post(f"/api/runs/{'a' * 32}/cancel").status_code == 404


def test_a_public_site_limits_each_visitor(client, tmp_path, public, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "RATE_ANALYZE", 2)
    monkeypatch.setattr(server, "_hits", {})
    video = upload_drop(client, tmp_path)
    me = {"CF-Connecting-IP": "203.0.113.7"}
    body = {"video_id": video, "scale": {"px_per_m": 300}, "settings": {"tracker": "color"}}
    assert client.post("/api/analyze", json=body, headers=me).status_code == 200
    assert client.post("/api/analyze", json=body, headers=me).status_code == 200
    third = client.post("/api/analyze", json=body, headers=me)
    assert third.status_code == 429 and "try again" in third.json()["detail"]
    other = {"CF-Connecting-IP": "198.51.100.4"}
    assert client.post("/api/analyze", json=body, headers=other).status_code == 200  # someone else is not blocked


def test_uploads_are_limited_too(client, tmp_path, public, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "RATE_UPLOAD", 1)
    monkeypatch.setattr(server, "_hits", {})
    upload_drop(client, tmp_path)
    with open(tmp_path / "drop.avi", "rb") as f:
        assert client.post("/api/videos", files={"file": ("drop.avi", f, "video/x-msvideo")}).status_code == 429


def test_no_limit_locally(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "RATE_ANALYZE", 1)
    monkeypatch.setattr(server, "_hits", {})
    video = upload_drop(client, tmp_path)
    for _ in range(3):
        assert analyze(client, video)[-1][0] == "done"


def test_a_used_up_budget_falls_back_to_the_plain_fit(client, tmp_path, monkeypatch):  # noqa: F811
    def broke(*a, **k):
        raise BudgetExceededError("logged spend $10.0000 reached the $10.00 limit")

    monkeypatch.setattr(server, "make_client", lambda: None)
    monkeypatch.setattr(server, "run_agent", broke)
    stream = dict(analyze(client, upload_drop(client, tmp_path), agent=True))
    assert stream["result"]["source"] == "deterministic" and "error" not in stream
    assert "budget" in stream["note"]["message"]
