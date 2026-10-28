"""Model-controlled input must never crash the loop; the CLI must make outcomes obvious."""

import json

import numpy as np
import pytest

import cli
from physicslens.agent.loop import run_agent
from physicslens.agent.tools import Toolbox, TrackData
from physicslens.llm.base import MockClient, Reply
from tests.test_agent_loop import pendulum_box
from tests.test_cli_agent import args, drop_video


def raw_call(i, name, arguments):
    return {"id": f"c{i}", "name": name, "arguments": arguments}


@pytest.mark.parametrize("arguments", ["[]", "null", "3", '"str"'])
def test_non_object_arguments_become_errors(arguments):
    client = MockClient([
        Reply("", [raw_call(1, "final_answer", arguments), raw_call(2, "fit_free_fall", arguments)]),
        Reply("", [raw_call(3, "fit_pendulum", '{"axis": "x"}')]),
        Reply("", [raw_call(4, "final_answer", '{"fit_id": "fit_1", "summary": "ok"}')]),
    ])
    result = run_agent(client, pendulum_box())
    errors = [json.loads(m.content) for m in client.requests[1] if m.role == "tool"]
    assert all("error" in e for e in errors)
    assert result.fit_id == "fit_1"


def test_unhashable_names_and_ids_become_errors():
    box = pendulum_box()
    assert "error" in box.call(["fit_free_fall"], {})
    client = MockClient([Reply("", [raw_call(1, "final_answer", '{"fit_id": ["fit_1"], "summary": "x"}')])])
    assert run_agent(client, box, max_steps=1).fit_id is None


def test_unexpected_numeric_exception_becomes_error(monkeypatch):
    box = pendulum_box()
    monkeypatch.setattr(box, "fit_free_fall", lambda **_: (_ for _ in ()).throw(FloatingPointError("overflow")))
    assert box.call("fit_free_fall", {}) == {"error": "FloatingPointError: overflow"}


def test_describe_track_on_tiny_track_is_an_error():
    t = np.array([0.0])
    assert "error" in Toolbox(TrackData(t, t, t)).call("describe_track", {})


def test_cli_reports_no_answer_with_its_own_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_client", lambda: MockClient([Reply("thinking")] * 12))
    assert cli.main(args(drop_video(tmp_path))) == 4
    assert "no accepted answer" in capsys.readouterr().err


def test_cli_flags_accepted_problem_fit_on_stderr(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "make_client", lambda: MockClient([
        Reply("", [raw_call(1, "fit_pendulum", '{"axis": "y"}')]),
        Reply("", [raw_call(2, "final_answer", '{"fit_id": "fit_1", "summary": "flagged, but y is all we tracked", "accept_flagged": true}')]),
    ]))
    assert cli.main(args(drop_video(tmp_path))) == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    out = captured.out
    assert out.index("model summary") > out.index("fit_1")  # measured values first, model text labelled


def test_cli_passes_drop_height_to_agent(tmp_path, monkeypatch):
    scripted = MockClient([Reply("thinking")] * 12)
    monkeypatch.setattr(cli, "make_client", lambda: scripted)
    cli.main(args(drop_video(tmp_path)) + ["--height", "0.75"])
    assert "0.75" in scripted.requests[0][1].content


def test_cli_rejects_json_with_agent(tmp_path):
    with pytest.raises(SystemExit):
        cli.main(args(drop_video(tmp_path)) + ["--json"])
