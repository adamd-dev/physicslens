"""Convert pixel coordinates to metres using a reference object of known size."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Calibration:
    px_per_m: float

    @classmethod
    def from_reference(cls, length_px: float, length_m: float) -> "Calibration":
        if length_px <= 0 or length_m <= 0:
            raise ValueError("reference lengths must be positive")
        return cls(length_px / length_m)

    @classmethod
    def from_points(cls, p1, p2, length_m: float) -> "Calibration":
        length_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        return cls.from_reference(length_px, length_m)

    def to_meters(self, px):
        return np.asarray(px, float) / self.px_per_m
