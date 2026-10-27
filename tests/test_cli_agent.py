"""cli.py --agent end to end on a synthetic drop, with a scripted model."""

import json

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


def test_agent_without_key_fails_cleanly(tmp_path, monkeypatch, capsys):
    def missing():
        raise LLMConfigError("MODEL_API_KEY is not set")

    monkeypatch.setattr(cli, "make_client", missing)
    assert cli.main(args(drop_video(tmp_path))) == 3
    assert "MODEL_API_KEY is not set" in capsys.readouterr().err
