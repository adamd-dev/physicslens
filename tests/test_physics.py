import numpy as np
import pytest

from physicslens.physics import models
from physicslens.physics.fit import (
    dominant_frequency,
    fit_damped_oscillation,
    fit_free_fall,
    fit_projectile,
)

FPS = 60
NOISE_M = 0.001  # 1 mm tracking noise


def noisy(values, seed=0):
    # Fresh generator per call so results don't depend on test order or -k selection.
    return values + np.random.default_rng(seed).normal(0, NOISE_M, size=len(values))


def test_free_fall_recovers_g():
    t = np.arange(0, 0.6, 1 / FPS)
    y = noisy(models.free_fall(t, 0.1, 0.0, models.G))
    result = fit_free_fall(t, y)
    assert result.params["g"] == pytest.approx(models.G, rel=0.02)
    assert result.r2 > 0.99


def test_projectile_recovers_vx_and_g():
    t = np.arange(0, 0.4, 1 / FPS)
    x = noisy(models.linear(t, 0.0, 1.2))
    y = noisy(models.free_fall(t, 0.0, 0.0, models.G), seed=1)
    result = fit_projectile(t, x, y)
    assert result.params["vx"] == pytest.approx(1.2, rel=0.02)
    assert result.params["g"] == pytest.approx(models.G, rel=0.03)


def test_damped_pendulum_recovers_length():
    length, gamma = 0.5, 0.05
    omega = np.sqrt(models.G / length - gamma**2)
    t = np.arange(0, 10, 1 / FPS)
    s = noisy(models.damped_oscillation(t, 0.08, gamma, omega, 0.3, 0.2))
    result = fit_damped_oscillation(t, s)
    estimated = models.pendulum_length(result.params["omega"], result.params["gamma"])
    assert estimated == pytest.approx(length, rel=0.02)
    assert result.params["gamma"] == pytest.approx(gamma, abs=0.01)


def test_dominant_frequency_with_uneven_timing():
    t = np.cumsum(np.random.default_rng(2).uniform(0.9, 1.1, 600) / FPS)
    s = np.sin(2 * np.pi * 2.5 * t)
    assert dominant_frequency(t, s) == pytest.approx(2.5, rel=0.01)


def test_landing_distance_closed_form():
    expected = np.sqrt(2 * 0.8 / models.G)
    assert models.landing_distance(vx=1.0, height=0.8) == pytest.approx(expected)


def test_landing_distance_with_downward_launch_is_shorter():
    flat = models.landing_distance(vx=1.0, height=0.8)
    angled = models.landing_distance(vx=1.0, height=0.8, vy=0.5)
    assert angled < flat


def test_an_oscillation_fit_gives_up_quickly_on_a_straight_line():
    # Each failed start used to run 20000 evaluations: about a minute for a path that is no swing.
    import time

    from physicslens.physics.fit import fit_damped_oscillation

    t = np.arange(40) / 60
    start = time.perf_counter()
    with pytest.raises(RuntimeError):
        fit_damped_oscillation(t, 0.05 + 0.7 * t)
    assert time.perf_counter() - start < 20
