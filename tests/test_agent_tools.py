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


def curved_toss_box():
    """TossOut-like: 21 points at 30 fps, perspective bends both axes; the full projectile fit has runs z = -4."""
    t = np.arange(21) / 30
    rng = np.random.default_rng(4)
    x = 0.3 + 1.7 * t + 0.3 * t**2 + rng.normal(0, 0.001, len(t))
    y = models.free_fall(t, 0.5, -2.3, models.G) + 0.3 * t**3 + rng.normal(0, 0.001, len(t))
    return Toolbox(TrackData(t, x, y))


def problems(box, summary):
    return box.call("check_plausibility", {"fit_id": summary["fit_id"]})["problems"]


def test_short_window_is_flagged():
    # Real TossOut run: Muse kept shrinking the window until the runs test lost power.
    box = curved_toss_box()
    assert any("residuals are structured" in p for p in problems(box, box.call("fit_projectile", {})))
    short = problems(box, box.call("fit_projectile", {"t_end": 0.37}))
    assert any("fit uses only 12/21 points" in p for p in short)


def test_window_cannot_hide_a_wrong_model():
    # On this track 11 of 15 windows with >= 80% of the points pass the runs test on their own.
    box = curved_toss_box()
    for t_start, t_end in [(0, 0.55), (1 / 30, 0.6), (2 / 30, 0.67), (0.1, 0.67)]:
        found = problems(box, box.call("fit_projectile", {"t_start": t_start, "t_end": t_end}))
        assert any("full track fails" in p for p in found), (t_start, t_end, found)


def test_edge_trim_still_passes():
    box = drop_box()
    trimmed = box.call("fit_free_fall", {"t_end": 0.46})
    assert box.call("check_plausibility", {"fit_id": trimmed["fit_id"]})["ok"]


def test_free_fall_flagged_when_ball_moves_sideways():
    # Fitting y alone would skip the projectile check on x.
    box = curved_toss_box()
    assert any("sideways" in p for p in problems(box, box.call("fit_free_fall", {})))


def test_fit_stored_without_window_info_is_still_checked():
    box = drop_box()
    box.fits["fit_x"] = box.fits[box.call("fit_free_fall", {})["fit_id"]]
    assert box.call("check_plausibility", {"fit_id": "fit_x"})["ok"]


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
