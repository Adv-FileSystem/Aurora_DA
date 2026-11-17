"""Augmentation transforms for multivariate windows of shape (L, C)."""
from .base import Transform, Compose, RandomApply, OneOf
from .jitter import Jitter
from .scaling import Scaling
from .magnitude_warp import MagnitudeWarp
from .time_warp import TimeWarp
from .window_slice import WindowSlice

__all__ = [
    "Transform",
    "Compose",
    "RandomApply",
    "OneOf",
    "Jitter",
    "Scaling",
    "MagnitudeWarp",
    "TimeWarp",
    "WindowSlice",
]
