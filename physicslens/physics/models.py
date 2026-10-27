"""Physics models and closed-form predictions.

Units are SI. The vertical axis follows image coordinates: positive y points
down, so gravity is a positive acceleration.
"""

import numpy as np

G = 9.81


def free_fall(t, y0, v0, g):
    return y0 + v0 * t + 0.5 * g * t**2


def linear(t, x0, v):
    return x0 + v * t


def damped_oscillation(t, amplitude, gamma, omega, phase, offset):
    return amplitude * np.exp(-gamma * t) * np.cos(omega * t + phase) + offset


def pendulum_length(omega: float, gamma: float = 0.0, g: float = G) -> float:
    """Length of a simple pendulum from its damped angular frequency."""
    natural_omega_sq = omega**2 + gamma**2
    return g / natural_omega_sq


def landing_time(height: float, vy: float = 0.0, g: float = G) -> float:
    """Time to fall `height` metres starting with downward velocity `vy`."""
    return (-vy + np.sqrt(vy**2 + 2 * g * height)) / g


def landing_distance(vx: float, height: float, vy: float = 0.0, g: float = G) -> float:
    """Horizontal distance from the launch edge to the landing point."""
    return vx * landing_time(height, vy, g)
