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


class MinMaxScaler:
    """Per-channel min-max scaler. Fit on train, apply to train and test."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.min_ = None
        self.max_ = None

    def fit(self, x):
        x = np.asarray(x, dtype=np.float32)
        self.min_ = x.min(axis=0)
        self.max_ = x.max(axis=0)
        return self

    def transform(self, x):
        if self.min_ is None:
            raise RuntimeError("MinMaxScaler is not fitted")
        x = np.asarray(x, dtype=np.float32)
        return (x - self.min_) / (self.max_ - self.min_ + self.eps)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, x):
        x = np.asarray(x, dtype=np.float32)
        return x * (self.max_ - self.min_ + self.eps) + self.min_
