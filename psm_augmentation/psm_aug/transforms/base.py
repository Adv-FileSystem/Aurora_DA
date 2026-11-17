"""Base classes for window-level transforms. A window has shape (L, C)."""
import numpy as np


class Transform:
    """Base class. Subclasses implement ``__call__(x, rng)``."""

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({params})"


class Compose(Transform):
    """Apply transforms in sequence."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x, rng):
        for t in self.transforms:
            x = t(x, rng)
        return x


class RandomApply(Transform):
    """Apply ``transform`` with probability ``p``."""

    def __init__(self, transform, p: float = 0.5):
        self.transform = transform
        self.p = p

    def __call__(self, x, rng):
        if rng.random() < self.p:
            return self.transform(x, rng)
        return x


class OneOf(Transform):
    """Apply one randomly chosen transform."""

    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x, rng):
        return self.transforms[rng.integers(len(self.transforms))](x, rng)
