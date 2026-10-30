"""An answer without a fit or without its check is reported as no answer, never a crash."""

import pytest

import cli
from physicslens import server
from physicslens.agent.loop import AgentResult
from tests.test_cli_agent import args, drop_video
from tests.test_server import analyze, client, upload_drop  # noqa: F401  (client is a fixture)

# Accepted fit with no plausibility check: the type allows it, so callers must refuse it.
UNCHECKED = AgentResult("fit_1", "g is about 9.8", None, 2)


def test_cli_refuses_an_answer_without_a_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_client", lambda: None)
    monkeypatch.setattr(cli, "run_agent", lambda *a, **k: UNCHECKED)
    assert cli.main(args(drop_video(tmp_path))) == 4
    assert "no accepted answer" in capsys.readouterr().err


def test_cli_without_any_fit_exits_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_fits", lambda *a, **k: [{"name": "free_fall", "error": "too few points"}])
    assert cli.main(args(drop_video(tmp_path))[:-1]) == 4  # no fit, no measurement
    out = capsys.readouterr()
    assert "error: too few points" in out.out and "no measurement possible" in out.err


def test_api_refuses_an_answer_without_a_check(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(server, "make_client", lambda: None)
    monkeypatch.setattr(server, "run_agent", lambda *a, **k: UNCHECKED)
    stream = dict(analyze(client, upload_drop(client, tmp_path), agent=True))
    assert "result" not in stream
    assert stream["error"]["stage"] == "agent"


@pytest.mark.parametrize("missing", ["min_area", "max_area", "max_jump"])
def test_default_settings_are_complete(missing):
    assert missing in server.AnalyzeRequest(video_id="x", scale={"px_per_m": 1}).settings.model_dump()
