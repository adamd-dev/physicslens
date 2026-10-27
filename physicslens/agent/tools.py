"""Physics tools the agent can call.

The LLM never sees raw arrays: it picks a model and a time window, the tools
run the deterministic scipy fits and return small JSON-safe summaries.
"""

from dataclasses import dataclass, field

import numpy as np

from ..physics import fit, models

# Plausibility thresholds for a hand-held phone experiment.
MIN_R2 = 0.98
G_RANGE = (9.0, 10.6)
MAX_RUNS_Z = 3.0
MIN_CYCLES = 1.5  # an oscillation fit on less than this extrapolates the period
PENDULUM_LENGTH_RANGE = (0.05, 5.0)


@dataclass
class TrackData:
    """Trajectory in metres (y points down), as handed to the agent."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray


def _window(data: TrackData, t_start: float | None, t_end: float | None):
    keep = np.ones(len(data.t), bool)
    if t_start is not None:
        keep &= data.t >= t_start
    if t_end is not None:
        keep &= data.t <= t_end
    if keep.sum() < 8:
        raise ValueError(f"only {int(keep.sum())} points in window; need at least 8")
    # Shift time so fitted initial values describe the start of the window.
    return data.t[keep] - data.t[keep][0], data.x[keep], data.y[keep], float(data.t[keep][0])


def runs_z(residuals: np.ndarray) -> float:
    """Wald-Wolfowitz runs test z-score on residual signs.

    A large |z| means residuals come in long same-sign stretches, i.e. the
    model has the wrong shape even if R² looks fine.
    """
    signs = np.sign(residuals[residuals != 0])
    n_pos, n_neg = int((signs > 0).sum()), int((signs < 0).sum())
    n = n_pos + n_neg
    if n_pos == 0 or n_neg == 0:
        return float("inf")
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mean = 2 * n_pos * n_neg / n + 1
    var = 2 * n_pos * n_neg * (2 * n_pos * n_neg - n) / (n**2 * (n - 1))
    return float((runs - mean) / np.sqrt(var)) if var > 0 else 0.0


def _summary(fit_id: str, result: fit.FitResult, t_offset: float, n: int) -> dict:
    return {
        "fit_id": fit_id,
        "model": result.model,
        "params": {k: round(v, 5) for k, v in result.params.items()},
        "stderr": {k: round(v, 5) for k, v in result.stderr.items()},
        "r2": round(result.r2, 5),
        "rmse_m": round(result.rmse, 5),
        "window_start_s": round(t_offset, 4),
        "n_points": n,
    }


@dataclass
class Toolbox:
    data: TrackData
    fits: dict[str, fit.FitResult] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)

    def _store(self, result: fit.FitResult, t: np.ndarray, t_offset: float) -> dict:
        fit_id = f"fit_{len(self.fits) + 1}"
        self.fits[fit_id] = result
        self.durations[fit_id] = float(t[-1] - t[0])
        return _summary(fit_id, result, t_offset, len(t))

    def describe_track(self) -> dict:
        t, x, y = self.data.t, self.data.x, self.data.y
        dt = np.diff(t)
        x_span, y_span = float(np.ptp(x)), float(np.ptp(y))
        return {
            "n_points": len(t),
            "duration_s": round(float(t[-1] - t[0]), 4),
            "median_fps": round(float(1 / np.median(dt)), 2),
            "max_gap_s": round(float(dt.max()), 4),
            "x_span_m": round(x_span, 4),
            "y_span_m": round(y_span, 4),
            "x_dominant_hz": round(fit.dominant_frequency(t, x), 3) if x_span > 0 else 0.0,
            "y_monotonic_fraction": round(float(np.mean(np.diff(y) > 0)), 3),
        }

    def fit_free_fall(self, t_start: float | None = None, t_end: float | None = None) -> dict:
        t, _, y, t0 = _window(self.data, t_start, t_end)
        return self._store(fit.fit_free_fall(t, y), t, t0)

    def fit_projectile(self, t_start: float | None = None, t_end: float | None = None) -> dict:
        t, x, y, t0 = _window(self.data, t_start, t_end)
        return self._store(fit.fit_projectile(t, x, y), t, t0)

    def fit_pendulum(self, axis: str = "x", t_start: float | None = None, t_end: float | None = None) -> dict:
        t, x, y, t0 = _window(self.data, t_start, t_end)
        result = fit.fit_damped_oscillation(t, x if axis == "x" else y)
        summary = self._store(result, t, t0)
        summary["length_m"] = round(models.pendulum_length(result.params["omega"], result.params["gamma"]), 4)
        summary["cycles_observed"] = round(result.params["omega"] * self.durations[summary["fit_id"]] / (2 * np.pi), 2)
        return summary

    def check_plausibility(self, fit_id: str) -> dict:
        if fit_id not in self.fits:
            raise ValueError(f"unknown fit_id {fit_id!r}; known: {sorted(self.fits)}")
        result = self.fits[fit_id]
        problems = []
        if not np.isfinite(result.r2):
            problems.append("r2 undefined: tracked signal has no variance")
        elif result.r2 < MIN_R2:
            problems.append(f"r2 {result.r2:.4f} below {MIN_R2}")
        z = runs_z(result.residuals)
        if abs(z) > MAX_RUNS_Z:
            problems.append(f"residuals are structured (runs z={z:.1f}); model shape likely wrong")
        g = result.params.get("g")
        if g is not None and not G_RANGE[0] <= g <= G_RANGE[1]:
            problems.append(f"g={g:.3f} outside plausible {G_RANGE}; check calibration or time scale")
        if result.model == "damped_oscillation":
            cycles = result.params["omega"] * self.durations[fit_id] / (2 * np.pi)
            if cycles < MIN_CYCLES:
                problems.append(f"only {cycles:.2f} cycles observed; period is extrapolated, not measured")
            length = models.pendulum_length(result.params["omega"], result.params["gamma"])
            if not PENDULUM_LENGTH_RANGE[0] <= length <= PENDULUM_LENGTH_RANGE[1]:
                problems.append(f"pendulum length {length:.3f} m outside plausible {PENDULUM_LENGTH_RANGE}")
        return {"fit_id": fit_id, "ok": not problems, "problems": problems, "runs_z": round(z, 2)}

    def predict_landing(self, fit_id: str, height_m: float, use_measured_g: bool = True) -> dict:
        result = self.fits.get(fit_id)
        if result is None or result.model != "projectile":
            raise ValueError("predict_landing needs a projectile fit_id")
        if height_m <= 0:
            raise ValueError("height_m must be positive")
        p, s = result.params, result.stderr
        g = p["g"] if use_measured_g else models.G
        distance = models.landing_distance(p["vx"], height_m, p["vy"], g)
        # First-order uncertainty from vx only; vy and g errors are smaller for a short roll-off.
        sigma = abs(distance / p["vx"]) * s["vx"] if p["vx"] else float("nan")
        return {
            "fit_id": fit_id,
            "landing_distance_m": round(float(distance), 4),
            "uncertainty_m": round(float(sigma), 4),
            "flight_time_s": round(float(models.landing_time(height_m, p["vy"], g)), 4),
            "g_used": round(float(g), 4),
        }

    def call(self, name: str, arguments: dict) -> dict:
        """Run a tool by name; errors come back as data so the agent can recover."""
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name!r}"}
        try:
            return getattr(self, name)(**arguments)
        except (ValueError, TypeError, RuntimeError) as exc:
            return {"error": str(exc)}


_WINDOW = {
    "t_start": {"type": "number", "description": "Window start in seconds (optional)."},
    "t_end": {"type": "number", "description": "Window end in seconds (optional)."},
}


def _schema(name: str, description: str, properties: dict, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required or []},
        },
    }


TOOL_SCHEMAS = [
    _schema("describe_track", "Summary stats of the tracked trajectory. Call first.", {}),
    _schema("fit_free_fall", "Fit y(t) = y0 + v0 t + g t²/2 (y down). For drops.", _WINDOW),
    _schema("fit_projectile", "Fit uniform x motion plus free fall in y. For a ball leaving a table.", _WINDOW),
    _schema(
        "fit_pendulum",
        "Fit a damped oscillation and derive pendulum length.",
        {"axis": {"type": "string", "enum": ["x", "y"]}, **_WINDOW},
    ),
    _schema(
        "check_plausibility",
        "Check a fit: R², residual structure, physical range of g. Always call before reporting.",
        {"fit_id": {"type": "string"}},
        ["fit_id"],
    ),
    _schema(
        "predict_landing",
        "Predict horizontal landing distance from a projectile fit and a drop height.",
        {
            "fit_id": {"type": "string"},
            "height_m": {"type": "number"},
            "use_measured_g": {"type": "boolean"},
        },
        ["fit_id", "height_m"],
    ),
]
TOOL_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS}
