"""Common track format shared by every tracker (SAM 3.1, color fallback)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    """Object centroid in pixels for each frame where it was found."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    source: str

    def __len__(self) -> int:
        return len(self.t)
