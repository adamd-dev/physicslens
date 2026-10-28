"""cli.py --agent end to end on a synthetic drop, with a scripted model."""

import json

import pytest

import cli
from physicslens.llm.base import MockClient, Reply
from physicslens.llm.meta_client import LLMConfigError
from tests.test_robustness import PX_PER_M, write_drop


def call(i, name, **arguments):
    return {"id": f"c{i}", "name": name, "arguments": json.dumps(arguments)}


def drop_video(tmp_path):
    path = tmp_path / "drop.avi"
    write_drop(path, capture_fps=60, playback_fps=60, n_frames=27)
    return str(path)


def args(path):
    return [path, "--px-per-m", str(PX_PER_M), "--tracker", "color", "--agent"]


def test_agent_flag_runs_the_loop(tmp_path, monkeypatch, capsys):
    scripted = MockClient([
        Reply("", [call(1, "describe_track")]),
        Reply("", [call(2, "fit_free_fall"), call(3, "check_plausibility", fit_id="fit_1")]),
        Reply("", [call(4, "final_answer", fit_id="fit_1", summary="g is about 9.8 m/s²")]),
    ])
    monkeypatch.setattr(cli, "make_client", lambda: scripted)
    assert cli.main(args(drop_video(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "in 3 steps" in out
    assert "g is about 9.8" in out
    assert "g = 9.8" in out


def test_transcript_saves_every_message(tmp_path, monkeypatch):
    scripted = MockClient([
        Reply("", [call(1, "fit_free_fall")]),
        Reply("", [call(2, "final_answer", fit_id="fit_1", summary="drop")]),
    ])
    monkeypatch.setattr(cli, "make_client", lambda: scripted)
    out = tmp_path / "run.json"
    assert cli.main(args(drop_video(tmp_path)) + ["--transcript", str(out)]) == 0
    messages = json.loads(out.read_text())
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert messages[2]["tool_calls"][0]["name"] == "fit_free_fall"


def test_transcript_saved_when_agent_gives_up(tmp_path, monkeypatch):
    scripted = MockClient([Reply("thinking", [])] * 12)
    monkeypatch.setattr(cli, "make_client", lambda: scripted)
    out = tmp_path / "run.json"
    assert cli.main(args(drop_video(tmp_path)) + ["--transcript", str(out)]) == 4
    assert json.loads(out.read_text())[2]["content"] == "thinking"
    assert out.stat().st_mode & 0o777 == 0o600


def test_no_measurement_is_reported_not_a_number(tmp_path, monkeypatch, capsys):
    reason = "the hand was tracked, not the ball; draw a search area"
    monkeypatch.setattr(cli, "make_client", lambda: MockClient([Reply("", [call(1, "final_answer", summary=reason)])]))
    assert cli.main(args(drop_video(tmp_path))) == 4
    captured = capsys.readouterr()
    assert f"no measurement possible: {reason}" in captured.err
    assert "g =" not in captured.out


def test_unwritable_transcript_only_warns(tmp_path, monkeypatch, capsys):
    scripted = MockClient([Reply("", [call(1, "fit_free_fall")]), Reply("", [call(2, "final_answer", fit_id="fit_1", summary="drop")])])
    monkeypatch.setattr(cli, "make_client", lambda: scripted)
    assert cli.main(args(drop_video(tmp_path)) + ["--transcript", str(tmp_path / "missing" / "run.json")]) == 0
    assert "transcript not saved" in capsys.readouterr().err


def test_transcript_needs_agent(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([drop_video(tmp_path), "--px-per-m", "100", "--transcript", str(tmp_path / "run.json")])
    assert exc.value.code == 2
    assert "--transcript needs --agent" in capsys.readouterr().err


def test_agent_without_key_fails_cleanly(tmp_path, monkeypatch, capsys):
    def missing():
        raise LLMConfigError("MODEL_API_KEY is not set")

    monkeypatch.setattr(cli, "make_client", missing)
    assert cli.main(args(drop_video(tmp_path))) == 3
    assert "MODEL_API_KEY is not set" in capsys.readouterr().err
