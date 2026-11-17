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


def window_labels(labels, window: int, stride: int = 1, rule: str = "any") -> np.ndarray:
    """Aggregate point labels into one label per window.

    rule: ``any`` (1 if any point is anomalous), ``last`` (label of the last point),
    ``majority`` (1 if at least half the points are anomalous).
    """
    labels = np.asarray(labels).astype(np.int64)
    w = sliding_windows(labels[:, None], window, stride)[..., 0]
    if rule == "any":
        return (w.max(axis=1) > 0).astype(np.int64)
    if rule == "last":
        return w[:, -1].astype(np.int64)
    if rule == "majority":
        return (w.mean(axis=1) >= 0.5).astype(np.int64)
    raise ValueError(f"unknown rule {rule!r}")
