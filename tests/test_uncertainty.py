"""The reported ± covers the scale reading, not only the fit noise."""

import math

import numpy as np
import pytest

from physicslens.agent.tools import Toolbox, TrackData
from physicslens.calibrate import READ_PX, Calibration
from physicslens.physics import models
from tests.test_agent_tools import drop_box, noisy, pendulum_box
from tests.test_robustness import PX_PER_M
from tests.test_server import analyze, client, upload_drop  # noqa: F401  (client is a fixture)

FPS = 60


def test_scale_error_is_the_reading_error_over_the_reference_length():
    assert Calibration.from_points((0, 0), (0, 400), 1.0).rel_err == pytest.approx(READ_PX / 400)
    assert Calibration.from_reference(400, 1.0).rel_err == pytest.approx(READ_PX / 400)


def test_resizing_the_frames_keeps_the_scale_error():
    scaled = Calibration.from_points((0, 0), (0, 400), 1.0).scaled(0.5)
    assert scaled.px_per_m == pytest.approx(200)
    assert scaled.rel_err == pytest.approx(READ_PX / 400)


def test_g_total_error_adds_the_scale_error_in_quadrature():
    box = drop_box()
    box.scale_rel_err = 0.005
    summary = box.call("fit_free_fall", {})
    g, fit_err = summary["params"]["g"], summary["stderr"]["g"]
    assert summary["stderr_total"]["g"] == pytest.approx(math.hypot(fit_err, g * 0.005), rel=1e-3)
    assert summary["stderr_total"]["g"] > 3 * fit_err  # 1 mm noise: the scale dominates


def test_angular_parameters_do_not_depend_on_the_scale():
    box = pendulum_box()
    box.scale_rel_err = 0.01
    summary = box.call("fit_pendulum", {"axis": "x"})
    assert summary["stderr_total"]["omega"] == summary["stderr"]["omega"]
    assert summary["stderr_total"]["amplitude"] > summary["stderr"]["amplitude"]


def test_landing_error_includes_the_scale():
    t = np.arange(0, 0.25, 1 / FPS)
    data = TrackData(t, noisy(models.linear(t, 0, 1.2), 4), noisy(models.free_fall(t, 0, 0, models.G), 5))
    exact, scaled = Toolbox(data), Toolbox(data, scale_rel_err=0.01)
    a = exact.call("predict_landing", {"fit_id": exact.call("fit_projectile", {})["fit_id"], "height_m": 0.75})
    b = scaled.call("predict_landing", {"fit_id": scaled.call("fit_projectile", {})["fit_id"], "height_m": 0.75})
    assert b["fit_uncertainty_m"] == pytest.approx(a["uncertainty_m"])
    assert b["uncertainty_m"] > a["uncertainty_m"]
    assert b["uncertainty_m"] >= 0.005 * b["landing_distance_m"]  # about half the 1% scale error at least


def test_api_reports_the_total_error_and_its_parts(client, tmp_path):  # noqa: F811
    result = dict(analyze(client, upload_drop(client, tmp_path)))["result"]
    assert result["scale_rel_err"] == pytest.approx(READ_PX / PX_PER_M)
    g = result["params"]["g"]
    assert result["stderr_total"]["g"] == pytest.approx(math.hypot(result["stderr"]["g"], g * READ_PX / PX_PER_M), rel=1e-3)
    assert result["deviation_pct"]["g"] == pytest.approx(100 * (g - models.G) / models.G)
