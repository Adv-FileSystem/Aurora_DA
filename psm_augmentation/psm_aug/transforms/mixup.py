"""Mixup between two windows (applied to normal windows in the pipeline)."""
import numpy as np


def mixup(x1: np.ndarray, x2: np.ndarray, rng, alpha: float = 0.2):
    """Return ``(mixed, lam)``; ``lam`` is kept >= 0.5 so ``x1`` dominates."""
    lam = float(rng.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    return (lam * x1 + (1.0 - lam) * x2).astype(np.float32), lam
