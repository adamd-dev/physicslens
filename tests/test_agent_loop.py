"""Agent loop against scripted replies: the loop, not the model, is under test."""

import json

import numpy as np

from physicslens.agent.loop import run_agent
from physicslens.agent.tools import Toolbox, TrackData
from physicslens.llm.base import MockClient, Reply
from physicslens.physics import models

FPS = 60


def call(i, name, **arguments):
    return {"id": f"c{i}", "name": name, "arguments": json.dumps(arguments)}


def pendulum_box():
    t = np.arange(0, 6, 1 / FPS)
    rng = np.random.default_rng(0)
    x = models.damped_oscillation(t, 0.08, 0.05, np.sqrt(models.G / 0.5), 0.3, 0.2) + rng.normal(0, 0.001, len(t))
    return Toolbox(TrackData(t, x, np.full_like(t, 0.6)))


def test_agent_recovers_from_wrong_model():
    client = MockClient([
        Reply("", [call(1, "describe_track")]),
        Reply("Looks like a drop.", [call(2, "fit_free_fall"), call(3, "check_plausibility", fit_id="fit_1")]),
        Reply("Not a drop; x oscillates.", [call(4, "fit_pendulum", axis="x"), call(5, "check_plausibility", fit_id="fit_2")]),
        Reply("", [call(6, "final_answer", fit_id="fit_2", summary="Pendulum length 0.50 m")]),
    ])
    result = run_agent(client, pendulum_box())
    assert result.fit_id == "fit_2"
    assert result.check["ok"]
    assert result.steps == 4
    # The failed check was fed back to the model before it switched models.
    tool_replies = [json.loads(m.content) for m in client.requests[2] if m.role == "tool"]
    assert any(r.get("ok") is False for r in tool_replies)


def test_flagged_fit_is_refused_without_explanation():
    client = MockClient([
        Reply("", [call(1, "fit_free_fall")]),
        Reply("", [call(2, "final_answer", fit_id="fit_1", summary="g = 0.1")]),
        Reply("", [call(3, "fit_pendulum", axis="x")]),
        Reply("", [call(4, "final_answer", fit_id="fit_2", summary="Pendulum length 0.50 m")]),
    ])
    result = run_agent(client, pendulum_box())
    refused = json.loads(client.requests[2][-1].content)
    assert "failed plausibility" in refused["error"]
    assert result.fit_id == "fit_2"


def test_flagged_fit_accepted_with_reason():
    client = MockClient([
        Reply("", [call(1, "fit_free_fall")]),
        Reply("", [call(2, "final_answer", fit_id="fit_1", summary="flagged: not a drop, x oscillates", accept_flagged=True)]),
    ])
    result = run_agent(client, pendulum_box())
    assert result.fit_id == "fit_1"
    assert not result.check["ok"]


def test_flagged_fit_needs_a_real_reason():
    client = MockClient([
        Reply("", [call(1, "fit_free_fall")]),
        Reply("", [call(2, "final_answer", fit_id="fit_1", summary="ok", accept_flagged=True)]),
    ])
    result = run_agent(client, pendulum_box(), max_steps=2)
    refused = json.loads(result.transcript[-1].content)
    assert "explain" in refused["error"]
    assert result.fit_id is None


def test_step_limit_and_bad_json():
    client = MockClient([Reply("", [{"id": "c1", "name": "fit_free_fall", "arguments": "{oops"}])] + [Reply("thinking")] * 3)
    result = run_agent(client, pendulum_box(), max_steps=4)
    assert result.fit_id is None
    assert "not valid JSON" in client.requests[1][-1].content
