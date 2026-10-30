"""Common track format shared by every tracker (SAM 3.1, color fallback)."""

from dataclasses import dataclass

import numpy as np

# A 10 m fall ends at 14 m/s: a blob that moved faster than this between two samples is another object.
MAX_SPEED_M_S = 15.0


@dataclass
class Track:
    """Object centroid in pixels for each frame where it was found."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    source: str

    def __len__(self) -> int:
        return len(self.t)
