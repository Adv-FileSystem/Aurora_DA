"""Per-channel random scaling."""
import numpy as np

from .base import Transform


class Scaling(Transform):
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, x, rng):
        factor = rng.normal(1.0, self.sigma, size=(1, x.shape[1]))
        return (x * factor).astype(np.float32)
