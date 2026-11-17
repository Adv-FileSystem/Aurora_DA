"""I/O helpers for the PSM dataset.

Expected layout::

    <root>/train.csv        timestamp_(min), feature_0 ... feature_24
    <root>/test.csv         timestamp_(min), feature_0 ... feature_24
    <root>/test_label.csv   timestamp_(min), label
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

TIMESTAMP_COL = "timestamp_(min)"


def load_split(root, split: str) -> pd.DataFrame:
    """Load the ``train`` or ``test`` split as a DataFrame indexed by timestamp."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    df = pd.read_csv(Path(root) / f"{split}.csv")
    if TIMESTAMP_COL in df.columns:
        df = df.set_index(TIMESTAMP_COL)
    return df


def load_labels(root) -> pd.Series:
    """Load point-wise test labels (0 normal, 1 anomaly)."""
    df = pd.read_csv(Path(root) / "test_label.csv")
    if TIMESTAMP_COL in df.columns:
        df = df.set_index(TIMESTAMP_COL)
    return df.iloc[:, 0].astype(int).rename("label")


def load_psm(root) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return ``(train, test, test_labels, feature_names)`` as numpy arrays."""
    train = load_split(root, "train")
    test = load_split(root, "test")
    labels = load_labels(root)
    if len(labels) != len(test):
        raise ValueError(f"label length {len(labels)} != test length {len(test)}")
    return (
        train.to_numpy(dtype=np.float32),
        test.to_numpy(dtype=np.float32),
        labels.to_numpy(dtype=np.int64),
        list(train.columns),
    )
