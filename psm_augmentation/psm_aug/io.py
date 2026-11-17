"""I/O helpers for the PSM dataset.

Expected layout::

    <root>/train.csv        timestamp_(min), feature_0 ... feature_24
    <root>/test.csv         timestamp_(min), feature_0 ... feature_24
    <root>/test_label.csv   timestamp_(min), label
"""
from pathlib import Path

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
