"""Preprocessing: missing-value filling and min-max normalization."""
import numpy as np


def fill_missing(x: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs column-wise, back-fill leading NaNs, all-NaN columns -> 0."""
    x = np.asarray(x, dtype=np.float32).copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        valid = ~np.isnan(col)
        if valid.all():
            continue
        if not valid.any():
            x[:, j] = 0.0
            continue
        idx = np.where(valid, np.arange(len(col)), 0)
        np.maximum.accumulate(idx, out=idx)
        filled = col[idx]
        first = int(np.argmax(valid))
        filled[:first] = col[first]
        x[:, j] = filled
    return x
