"""Convert pixel coordinates to metres using a reference object of known size."""

from dataclasses import dataclass

import numpy as np

# 1-sigma error on a reference length read off the video, in pixels (both ends together). Every
# length, speed and g scales with px_per_m, so this error carries straight into them.
READ_PX = 2.0


@dataclass(frozen=True)
class Calibration:
    px_per_m: float
    rel_err: float = 0.0  # relative 1-sigma error of px_per_m

    @classmethod
    def from_reference(cls, length_px: float, length_m: float) -> "Calibration":
        if length_px <= 0 or length_m <= 0:
            raise ValueError("reference lengths must be positive")
        return cls(length_px / length_m, READ_PX / length_px)

    @classmethod
    def from_points(cls, p1, p2, length_m: float) -> "Calibration":
        length_px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
        return cls.from_reference(length_px, length_m)

    def scaled(self, s: float) -> "Calibration":
        """The same reading on frames resized by s: the relative error does not change."""
        return Calibration(self.px_per_m * s, self.rel_err)

    def to_meters(self, px):
        return np.asarray(px, float) / self.px_per_m
