"""Shared helpers: smooth random curves and resampling."""
import numpy as np


def random_curve(length: int, n_channels: int, knots: int, sigma: float, rng) -> np.ndarray:
    """Piecewise-linear random curve around 1.0, shape (length, n_channels)."""
    xs = np.linspace(0, length - 1, knots + 2)
    ys = rng.normal(1.0, sigma, size=(knots + 2, n_channels))
    t = np.arange(length)
    return np.stack([np.interp(t, xs, ys[:, c]) for c in range(n_channels)], axis=1)


def resample(x: np.ndarray, new_len: int) -> np.ndarray:
    """Linearly resample a (L, C) window to (new_len, C)."""
    old = np.linspace(0.0, 1.0, x.shape[0])
    new = np.linspace(0.0, 1.0, new_len)
    return np.stack([np.interp(new, old, x[:, c]) for c in range(x.shape[1])], axis=1).astype(np.float32)
