"""History: every analysis is saved with its events, listed newest first, reopened and deleted."""

import json
import os

import pytest

from physicslens import server
from tests.test_agent_gate import wandering_video
from tests.test_server import analyze, client, upload_drop  # noqa: F401  (client is a fixture)


def upload(client, path, name):  # noqa: F811
    with open(path, "rb") as f:
        return client.post("/api/videos", files={"file": (name, f, "video/x-msvideo")}).json()["video_id"]


def test_a_run_is_saved_and_listed(client, tmp_path):  # noqa: F811
    stream = analyze(client, upload_drop(client, tmp_path))
    assert stream[0][0] == "run"  # the id comes first, so the UI can link to it while the run goes on
    run_id = stream[0][1]["id"]
    runs = client.get("/api/runs").json()
    assert [r["id"] for r in runs] == [run_id]
    summary = runs[0]
    assert summary["status"] == "ok" and summary["model"] == "free_fall" and summary["tracker"] == "color"
    assert summary["video_name"] == "drop.avi" and summary["agent"] is False
    assert summary["key"]["name"] == "g" and summary["key"]["value"] == pytest.approx(9.81, rel=0.02)
    assert summary["key"]["unit"] == "m/s²" and summary["key"]["err"] > 0


def test_a_saved_run_replays_its_stream(client, tmp_path):  # noqa: F811
    stream = analyze(client, upload_drop(client, tmp_path))
    record = client.get(f"/api/runs/{stream[0][1]['id']}").json()
    # The same events in the same order: the UI replays them through the live handler.
    assert [e["event"] for e in record["events"]] == [name for name, _ in stream]
    assert record["request"]["settings"]["tracker"] == "color" and record["video_id"]


def test_failed_runs_are_saved_as_no_measurement(client, tmp_path):  # noqa: F811
    analyze(client, upload(client, wandering_video(tmp_path / "w.avi"), "hand.avi"), settings={"tracker": "motion"})
    summary = client.get("/api/runs").json()[0]
    assert summary["status"] == "none" and summary["key"] is None
    assert summary["error"]["stage"] == "no measurement"


def test_runs_are_listed_newest_first(client, tmp_path):  # noqa: F811
    video = upload_drop(client, tmp_path)
    first = analyze(client, video)[0][1]["id"]
    second = analyze(client, video)[0][1]["id"]
    assert [r["id"] for r in client.get("/api/runs").json()] == [second, first]


def test_deleting_a_run_keeps_its_video(client, tmp_path):  # noqa: F811
    video = upload_drop(client, tmp_path)
    run_id = analyze(client, video)[0][1]["id"]
    assert client.delete(f"/api/runs/{run_id}").status_code == 204
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    # A path with slashes never reaches the handler (405 from the static files): nothing is deleted either way.
    assert client.delete(f"/api/runs/{run_id}").status_code in (404, 405)
    assert client.get("/api/runs").json() == []
    assert client.get(f"/api/videos/{video}/file").status_code == 200  # other runs may use it


@pytest.mark.parametrize("run_id", ["..%2F..%2Fcosts", "abc", "0" * 31 + "g", "0" * 32])
def test_unknown_or_malicious_run_ids_are_404(client, run_id):  # noqa: F811
    assert client.get(f"/api/runs/{run_id}").status_code == 404
    # A path with slashes never reaches the handler (405 from the static files): nothing is deleted either way.
    assert client.delete(f"/api/runs/{run_id}").status_code in (404, 405)


def test_run_files_are_private_and_a_broken_one_is_skipped(client, tmp_path):  # noqa: F811
    run_id = analyze(client, upload_drop(client, tmp_path))[0][1]["id"]
    path = server.RUNS / f"{run_id}.json"
    assert os.stat(path).st_mode & 0o777 == 0o600  # prompts, costs and file names stay on this account
    (server.RUNS / ("f" * 32 + ".json")).write_text("{not json")
    assert [r["id"] for r in client.get("/api/runs").json()] == [run_id]


def test_the_summary_of_a_pendulum_is_its_length():
    events = [{"event": "track", "data": {"source": "motion"}},
              {"event": "result", "data": {"model": "pendulum", "params": {"omega": 4.0}, "stderr_total": {}, "length_m": 0.59,
                                           "period_s": 1.54, "check": {"ok": False}, "deviation_pct": {}}},
              {"event": "cost", "data": {"run_usd": 0.0123, "total_usd": 1.0}}]
    s = server.summarize(events, {"settings": {"tracker": "motion"}, "agent": True, "start": 0.5})
    assert s["status"] == "flagged" and s["key"] == {"name": "L", "value": 0.59, "err": None, "unit": "m"}
    assert s["cost_usd"] == 0.0123 and s["agent"] is True and s["thumb_t"] == 0.5


def test_a_run_record_is_json(client, tmp_path):  # noqa: F811
    run_id = analyze(client, upload_drop(client, tmp_path))[0][1]["id"]
    record = json.loads((server.RUNS / f"{run_id}.json").read_text())
    assert record["id"] == run_id and record["summary"]["id"] == run_id and record["created_at"]


def test_a_failing_save_never_locks_the_server(client, tmp_path, monkeypatch):  # noqa: F811
    # Any error while saving must still end the stream and free the one-run lock.
    def broken(*a, **k):
        raise KeyError("ok")

    monkeypatch.setattr(server, "summarize", broken)
    video = upload_drop(client, tmp_path)
    assert analyze(client, video)[-1][0] == "done"
    assert analyze(client, video)[-1][0] == "done"  # not 409


def test_the_list_reads_only_summaries(client, tmp_path, monkeypatch):  # noqa: F811
    run_id = analyze(client, upload_drop(client, tmp_path))[0][1]["id"]
    (server.RUNS / f"{run_id}.json").write_text("{}")  # the full record is not needed for the list
    assert [r["id"] for r in client.get("/api/runs").json()] == [run_id]
    assert client.delete(f"/api/runs/{run_id}").status_code == 204
    assert client.get("/api/runs").json() == [] and not list(server.RUNS.iterdir())


def test_a_summary_without_a_date_is_skipped(client, tmp_path):  # noqa: F811
    run_id = analyze(client, upload_drop(client, tmp_path))[0][1]["id"]
    (server.RUNS / ("e" * 32 + ".sum.json")).write_text(json.dumps({"id": "e" * 32}))
    (server.RUNS / ("d" * 32 + ".sum.json")).write_text(json.dumps(["not", "a", "dict"]))
    assert [r["id"] for r in client.get("/api/runs").json()] == [run_id]


def test_odd_results_still_summarize():
    events = [{"event": "result", "data": {"model": "spring", "params": {"g": 9.8}, "stderr_total": {}, "check": {}}}]
    s = server.summarize(events, {"settings": {"tracker": "sam"}})
    assert s["status"] == "flagged" and s["key"]["name"] == "g" and s["tracker"] == "sam"


def test_a_long_file_name_is_cut(client, tmp_path):  # noqa: F811
    from tests.test_robustness import write_drop

    write_drop(tmp_path / "d.avi", 60, 60, 27)
    video = upload(client, tmp_path / "d.avi", "x" * 500 + ".avi")
    assert len(client.get(f"/api/videos/{video}/info").json()["name"]) <= 200


def test_video_info_rebuilds_what_the_stage_needs(client, tmp_path):  # noqa: F811
    video = upload_drop(client, tmp_path)
    info = client.get(f"/api/videos/{video}/info").json()
    assert info["video_id"] == video and info["name"] == "drop.avi" and info["width"] == 640 and info["fps"] > 0
    assert client.get(f"/api/videos/{'0' * 32}/info").status_code == 404
