"""Segment permutation. Breaks temporal order, so use it with a low probability."""
import numpy as np

from .base import Transform


class Permutation(Transform):
    def __init__(self, max_segments: int = 5):
        self.max_segments = max_segments

    def __call__(self, x, rng):
        length = x.shape[0]
        if self.max_segments < 2 or length < 2:
            return x
        n = int(rng.integers(2, min(self.max_segments, length) + 1))
        cuts = np.sort(rng.choice(np.arange(1, length), size=n - 1, replace=False))
        segments = np.split(x, cuts)
        order = rng.permutation(len(segments))
        return np.concatenate([segments[i] for i in order], axis=0).astype(np.float32)
