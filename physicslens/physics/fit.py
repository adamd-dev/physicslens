"""Fit physics models to trajectories and report fit quality."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from . import models


@dataclass
class FitResult:
    model: str
    params: dict[str, float]
    stderr: dict[str, float]
    r2: float
    rmse: float
    residuals: np.ndarray


def _r2(y, y_hat) -> float:
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fit(func, names, t, y, p0, bounds=(-np.inf, np.inf)):
    popt, pcov = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=20000)
    stderr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    residuals = y - func(t, *popt)
    return (
        dict(zip(names, map(float, popt))),
        dict(zip(names, map(float, stderr))),
        residuals,
    )


def fit_free_fall(t, y) -> FitResult:
    t, y = np.asarray(t, float), np.asarray(y, float)
    a, b, c = np.polyfit(t, y, 2)
    params, stderr, res = _fit(models.free_fall, ["y0", "v0", "g"], t, y, p0=[c, b, 2 * a])
    return FitResult("free_fall", params, stderr, _r2(y, y - res), float(np.sqrt(np.mean(res**2))), res)


def fit_projectile(t, x, y) -> FitResult:
    """Uniform horizontal motion plus free fall vertically."""
    t, x, y = (np.asarray(a, float) for a in (t, x, y))
    vx_guess, x0_guess = np.polyfit(t, x, 1)
    px, sx, rx = _fit(models.linear, ["x0", "vx"], t, x, p0=[x0_guess, vx_guess])
    vertical = fit_free_fall(t, y)
    residuals = np.concatenate([rx, vertical.residuals])
    return FitResult(
        "projectile",
        {**px, "y0": vertical.params["y0"], "vy": vertical.params["v0"], "g": vertical.params["g"]},
        {**sx, "y0": vertical.stderr["y0"], "vy": vertical.stderr["v0"], "g": vertical.stderr["g"]},
        min(_r2(x, x - rx), vertical.r2),
        float(np.sqrt(np.mean(residuals**2))),
        residuals,
    )


def dominant_frequency(t, s) -> float:
    """Dominant frequency in Hz, tolerant of slightly uneven frame timing."""
    t, s = np.asarray(t, float), np.asarray(s, float)
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1], dt)
    signal = np.interp(grid, t, s)
    signal = (signal - signal.mean()) * np.hanning(len(signal))
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), dt)
    k = int(np.argmax(spectrum[1:]) + 1)
    # Parabolic interpolation around the peak bin for sub-bin precision.
    if 1 <= k < len(spectrum) - 1:
        a, b, c = spectrum[k - 1], spectrum[k], spectrum[k + 1]
        denom = a - 2 * b + c
        shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
        return float((k + shift) * (freqs[1] - freqs[0]))
    return float(freqs[k])


def fit_damped_oscillation(t, s) -> FitResult:
    t, s = np.asarray(t, float), np.asarray(s, float)
    names = ["amplitude", "gamma", "omega", "phase", "offset"]
    offset = float(np.mean(s))
    amplitude = float((np.max(s) - np.min(s)) / 2)
    omega = 2 * np.pi * dominant_frequency(t, s)
    bounds = ([0, 0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
    best = None
    for phase in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        try:
            params, stderr, res = _fit(
                models.damped_oscillation, names, t, s, p0=[amplitude, 0.0, omega, phase, offset], bounds=bounds
            )
        except RuntimeError:
            continue
        sse = float(np.sum(res**2))
        if best is None or sse < best[0]:
            best = (sse, params, stderr, res)
    if best is None:
        raise RuntimeError("damped oscillation fit did not converge")
    _, params, stderr, res = best
    return FitResult("damped_oscillation", params, stderr, _r2(s, s - res), float(np.sqrt(np.mean(res**2))), res)
