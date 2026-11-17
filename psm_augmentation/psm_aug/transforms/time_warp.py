"""Time warping: smoothly speed up / slow down the time axis."""
import numpy as np

from ._curves import random_curve
from .base import Transform


class TimeWarp(Transform):
    def __init__(self, sigma: float = 0.2, knots: int = 4):
        self.sigma = sigma
        self.knots = knots

    def __call__(self, x, rng):
        length, n_channels = x.shape
        speed = np.clip(random_curve(length, 1, self.knots, self.sigma, rng)[:, 0], 0.1, None)
        warped = np.cumsum(speed)
        warped = (warped - warped[0]) / (warped[-1] - warped[0]) * (length - 1)
        t = np.arange(length)
        out = np.empty_like(x, dtype=np.float32)
        for c in range(n_channels):
            out[:, c] = np.interp(t, warped, x[:, c])
        return out
