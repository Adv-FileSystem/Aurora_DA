"""Additive Gaussian noise."""
import numpy as np

from .base import Transform


class Jitter(Transform):
    def __init__(self, sigma: float = 0.03):
        self.sigma = sigma

    def __call__(self, x, rng):
        noise = rng.normal(0.0, self.sigma, size=x.shape)
        return (x + noise).astype(np.float32)
