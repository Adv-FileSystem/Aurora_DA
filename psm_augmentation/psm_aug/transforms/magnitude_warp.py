"""Magnitude warping with a smooth random curve."""
import numpy as np

from ._curves import random_curve
from .base import Transform


class MagnitudeWarp(Transform):
    def __init__(self, sigma: float = 0.2, knots: int = 4):
        self.sigma = sigma
        self.knots = knots

    def __call__(self, x, rng):
        curve = random_curve(x.shape[0], x.shape[1], self.knots, self.sigma, rng)
        return (x * curve).astype(np.float32)
