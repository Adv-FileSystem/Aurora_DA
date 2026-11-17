"""Sliding-window utilities."""
import numpy as np


def sliding_windows(x: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """Cut a (T, C) series into windows of shape (N, window, C)."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"expected a (T, C) array, got shape {x.shape}")
    if window > x.shape[0]:
        raise ValueError(f"window {window} is larger than series length {x.shape[0]}")
    starts = np.arange(0, x.shape[0] - window + 1, stride)
    return np.stack([x[s:s + window] for s in starts]).astype(np.float32)
