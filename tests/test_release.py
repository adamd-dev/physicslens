"""Release detection: samples where the ball sits still (held before release, lying after landing) are not
part of the free motion and are dropped before fitting, with a note saying so."""

import numpy as np
import pytest

from physicslens.agent.tools import Toolbox
from physicslens.calibrate import Calibration
from physicslens.physics import models
from physicslens.pipeline import rest_samples, run_fits, to_track_data, trim_rest
from physicslens.track.base import Track

PX_PER_M = 400


def held_drop(fps=60, hold_s=0.2, fall_s=0.45, rest_after_s=0.0, v0=0.0, seed=0):
    """Ball held still (±0.7 px hand jitter), released from rest at hold_s, optionally lying still after."""
    rng = np.random.default_rng(seed)
    dt = 1 / fps
    t_hold = np.arange(0, hold_s - 1e-9, dt)
    t_fall = np.arange(0, fall_s, dt)
    y_fall = 0.1 + models.free_fall(t_fall, 0, v0, models.G)
    t_rest = np.arange(dt, rest_after_s + 1e-9, dt)
    t = np.concatenate([t_hold, hold_s + t_fall, hold_s + t_fall[-1] + t_rest])
    y = np.concatenate([np.full(len(t_hold), 0.1), y_fall, np.full(len(t_rest), y_fall[-1])])
    x = np.full(len(t), 0.5)
    still = np.r_[np.ones(len(t_hold), bool), np.zeros(len(t_fall), bool), np.ones(len(t_rest), bool)]
    jitter = rng.uniform(-0.7, 0.7, (2, len(t))) * still
    return Track(t, x * PX_PER_M + jitter[0], y * PX_PER_M + jitter[1], "test"), len(t_hold), len(t_rest)


def best_free_fall(tr):
    box = Toolbox(to_track_data(tr, Calibration(PX_PER_M)))
    return next(r for r in run_fits(box, "free_fall") if r["name"] == "free_fall")


def test_held_ball_is_trimmed_and_g_recovers():
    tr, n_hold, _ = held_drop()
    lead, trail = rest_samples(tr, PX_PER_M)
    # The first one or two samples after release have barely moved (0.5 g t² < 5 mm): they go too.
    assert n_hold <= lead <= n_hold + 2
    assert trail == 0
    kept, notes = trim_rest(tr, PX_PER_M)
    fit = best_free_fall(kept)
    assert fit["params"]["g"] == pytest.approx(models.G, rel=0.02)
    assert fit["check"]["ok"]
    assert len(notes) == 1 and f"{lead} samples" in notes[0]


def test_held_samples_bias_g_without_trimming():
    # The bug this fixes: the fit bends through the held points and reports a low g.
    tr, _, _ = held_drop()
    assert best_free_fall(tr)["params"]["g"] < models.G * 0.97


def test_clean_release_is_not_trimmed():
    tr, _, _ = held_drop(hold_s=0)
    assert rest_samples(tr, PX_PER_M) == (0, 0)
    kept, notes = trim_rest(tr, PX_PER_M)
    assert kept is tr and notes == []


def test_fast_camera_moving_start_is_not_trimmed():
    # 240 fps: three samples fit inside 5 mm, but 8 ms of motion is not a held ball.
    tr, _, _ = held_drop(fps=240, hold_s=0, v0=1.0)
    assert rest_samples(tr, PX_PER_M) == (0, 0)


def test_pendulum_turning_point_loses_at_most_a_few_samples():
    t = np.arange(0, 4, 1 / 30)
    omega = np.sqrt(models.G / 0.6)
    tr = Track(t, (0.5 + 0.1 * np.cos(omega * t)) * PX_PER_M, np.full(len(t), 0.9 * PX_PER_M), "test")
    lead, trail = rest_samples(tr, PX_PER_M)
    assert t[lead] - t[0] <= 0.1
    assert t[-1] - t[len(t) - 1 - trail] <= 0.1


def test_ball_lying_still_after_landing_is_trimmed():
    tr, n_hold, n_rest = held_drop(hold_s=0, rest_after_s=0.3)
    lead, trail = rest_samples(tr, PX_PER_M)
    assert lead == 0
    assert n_rest <= trail <= n_rest + 1
    kept, notes = trim_rest(tr, PX_PER_M)
    assert len(kept) == len(tr) - trail
    assert "after" in notes[0]
    assert best_free_fall(kept)["check"]["ok"]


def test_nothing_moving_is_left_alone():
    t = np.arange(0, 0.5, 1 / 60)
    tr = Track(t, np.full(len(t), 200.0), np.full(len(t), 100.0), "test")
    assert rest_samples(tr, PX_PER_M) == (0, 0)
    assert trim_rest(tr, PX_PER_M)[0] is tr


def test_kept_track_keeps_original_times():
    # The overlay maps samples back to the video clock: trimming must not shift time.
    tr, _, _ = held_drop()
    kept, _ = trim_rest(tr, PX_PER_M)
    lead = len(tr) - len(kept)
    assert np.array_equal(kept.t, tr.t[lead:])
    assert kept.source == tr.source
