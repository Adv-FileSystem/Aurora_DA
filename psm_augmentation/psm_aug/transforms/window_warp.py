"""Window warping: stretch or squeeze a short random segment."""
import numpy as np

from ._curves import resample
from .base import Transform


class WindowWarp(Transform):
    def __init__(self, window_ratio: float = 0.1, scales=(0.5, 2.0)):
        self.window_ratio = window_ratio
        self.scales = list(scales)

    def __call__(self, x, rng):
        length = x.shape[0]
        w = max(2, int(np.ceil(self.window_ratio * length)))
        if w >= length:
            return x
        start = int(rng.integers(0, length - w + 1))
        scale = self.scales[rng.integers(len(self.scales))]
        middle = resample(x[start:start + w], max(2, int(round(w * scale))))
        out = np.concatenate([x[:start], middle, x[start + w:]], axis=0)
        return resample(out, length)
