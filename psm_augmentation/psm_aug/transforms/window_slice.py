"""Window slicing: crop a sub-window and stretch it back to the original length."""
import numpy as np

from ._curves import resample
from .base import Transform


class WindowSlice(Transform):
    def __init__(self, ratio: float = 0.9):
        self.ratio = ratio

    def __call__(self, x, rng):
        length = x.shape[0]
        target = max(2, int(np.ceil(self.ratio * length)))
        if target >= length:
            return x
        start = int(rng.integers(0, length - target + 1))
        return resample(x[start:start + target], length)
