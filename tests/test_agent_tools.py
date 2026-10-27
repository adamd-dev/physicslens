import numpy as np
import pytest

from physicslens.agent.tools import TOOL_NAMES, TOOL_SCHEMAS, Toolbox, TrackData
from physicslens.physics import models

FPS = 60


def noisy(values, seed):
    return values + np.random.default_rng(seed).normal(0, 0.001, len(values))


def drop_box():
    t = np.arange(0, 0.5, 1 / FPS)
    return Toolbox(TrackData(t, noisy(np.full_like(t, 0.3), 0), noisy(models.free_fall(t, 0.1, 0, models.G), 1)))


def pendulum_box():
    t = np.arange(0, 6, 1 / FPS)
    omega = np.sqrt(models.G / 0.5)
    x = noisy(models.damped_oscillation(t, 0.08, 0.05, omega, 0.3, 0.2), 2)
    return Toolbox(TrackData(t, x, noisy(np.full_like(t, 0.6), 3)))


def test_drop_fit_passes_plausibility():
    box = drop_box()
    summary = box.call("fit_free_fall", {})
    assert summary["params"]["g"] == pytest.approx(models.G, rel=0.02)
    assert box.call("check_plausibility", {"fit_id": summary["fit_id"]})["ok"]


def test_wrong_model_is_flagged():
    box = pendulum_box()
    wrong = box.call("fit_free_fall", {})
    check = box.call("check_plausibility", {"fit_id": wrong["fit_id"]})
    assert not check["ok"]
    right = box.call("fit_pendulum", {"axis": "x"})
    assert right["length_m"] == pytest.approx(0.5, rel=0.02)
    assert box.call("check_plausibility", {"fit_id": right["fit_id"]})["ok"]


def test_predict_landing_matches_closed_form():
    t = np.arange(0, 0.25, 1 / FPS)
    box = Toolbox(TrackData(t, noisy(models.linear(t, 0, 1.2), 4), noisy(models.free_fall(t, 0, 0, models.G), 5)))
    fit_id = box.call("fit_projectile", {})["fit_id"]
    pred = box.call("predict_landing", {"fit_id": fit_id, "height_m": 0.75})
    assert pred["landing_distance_m"] == pytest.approx(models.landing_distance(1.2, 0.75), rel=0.03)
    assert pred["uncertainty_m"] > 0


def test_errors_come_back_as_data():
    box = drop_box()
    assert "error" in box.call("nope", {})
    assert "error" in box.call("check_plausibility", {"fit_id": "fit_99"})
    assert "error" in box.call("fit_free_fall", {"t_start": 10.0})
    assert "error" in box.call("fit_free_fall", {"bogus": 1})


def test_describe_track_is_json_safe():
    import json

    json.dumps(pendulum_box().call("describe_track", {}))


def test_schemas_match_methods():
    assert TOOL_NAMES == {"describe_track", "fit_free_fall", "fit_projectile", "fit_pendulum", "check_plausibility", "predict_landing"}
    assert all(s["type"] == "function" for s in TOOL_SCHEMAS)
