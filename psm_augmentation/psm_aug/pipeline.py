"""End-to-end augmentation pipeline for PSM."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .io import load_psm
from .preprocess import MinMaxScaler, fill_missing
from .registry import build_transform
from .transforms import mixup
from .windows import sliding_windows, window_labels


@dataclass
class AugmentResult:
    windows: np.ndarray       # (N, L, C)
    labels: np.ndarray        # (N,)
    is_augmented: np.ndarray  # (N,) bool


def prepare_windows(cfg):
    """Load PSM, preprocess it and cut it into windows. Returns ``(X, y)``."""
    train, test, test_labels, _ = load_psm(cfg["data"]["root"])
    train, test = fill_missing(train), fill_missing(test)
    if cfg.get("preprocess", {}).get("normalize", True):
        scaler = MinMaxScaler().fit(train)
        train, test = scaler.transform(train), scaler.transform(test)

    w = cfg["window"]
    if cfg["data"].get("split", "train") == "train":
        series, labels = train, np.zeros(len(train), dtype=np.int64)
    else:
        series, labels = test, test_labels
    X = sliding_windows(series, w["size"], w.get("stride", 1))
    y = window_labels(labels, w["size"], w.get("stride", 1), w.get("label_rule", "any"))
    return X, y


def augment_windows(X, y, aug_cfg, rng, clip=False):
    """Create augmented copies of windows, plus optional mixup samples."""
    transform = build_transform(aug_cfg.get("transforms", []))
    copies = int(aug_cfg.get("copies", 1))
    targets = np.arange(len(X))

    out_x, out_y = [], []
    for _ in range(copies):
        for i in targets:
            out_x.append(transform(X[i], rng))
            out_y.append(y[i])

    mix = aug_cfg.get("mixup") or {}
    normal = np.flatnonzero(y == 0)
    if mix.get("enabled", False) and len(normal) >= 2:
        n_mix = int(len(normal) * float(mix.get("ratio", 0.2)))
        for _ in range(n_mix):
            a, b = rng.choice(normal, size=2, replace=False)
            mixed, _ = mixup(X[a], X[b], rng, float(mix.get("alpha", 0.2)))
            out_x.append(mixed)
            out_y.append(0)

    if not out_x:
        return np.empty((0,) + X.shape[1:], dtype=np.float32), np.empty(0, dtype=np.int64)
    ax = np.stack(out_x).astype(np.float32)
    if clip:
        ax = np.clip(ax, 0.0, 1.0)
    return ax, np.asarray(out_y, dtype=np.int64)


def run(cfg) -> AugmentResult:
    """Run the full pipeline."""
    rng = np.random.default_rng(cfg.get("seed"))
    X, y = prepare_windows(cfg)
    aug_cfg = cfg.get("augment", {})
    normalized = cfg.get("preprocess", {}).get("normalize", True)
    clip = bool(aug_cfg.get("clip", False)) and normalized
    ax, ay = augment_windows(X, y, aug_cfg, rng, clip=clip)
    return AugmentResult(
        windows=np.concatenate([X, ax]),
        labels=np.concatenate([y, ay]),
        is_augmented=np.concatenate([np.zeros(len(X), dtype=bool), np.ones(len(ax), dtype=bool)]),
    )


def save_result(result: AugmentResult, out_dir, cfg=None) -> Path:
    """Save windows/labels as ``psm_augmented.npz`` (and the config as YAML)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "psm_augmented.npz"
    np.savez_compressed(
        path,
        windows=result.windows,
        labels=result.labels,
        is_augmented=result.is_augmented,
    )
    if cfg is not None:
        with open(out / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path
