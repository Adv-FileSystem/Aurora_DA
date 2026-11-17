"""Simple statistics to compare original and augmented windows."""
import numpy as np


def channel_stats(x: np.ndarray):
    flat = x.reshape(-1, x.shape[-1])
    return {
        "mean": flat.mean(axis=0),
        "std": flat.std(axis=0),
        "min": flat.min(axis=0),
        "max": flat.max(axis=0),
    }


def compare(original: np.ndarray, augmented: np.ndarray):
    a, b = channel_stats(original), channel_stats(augmented)
    return {k: float(np.abs(a[k] - b[k]).mean()) for k in a}


def format_report(original: np.ndarray, augmented: np.ndarray) -> str:
    diff = compare(original, augmented)
    lines = [
        f"original windows : {len(original)}",
        f"augmented windows: {len(augmented)}",
        "mean abs difference per statistic (averaged over channels):",
    ]
    lines += [f"  {k:<5}: {v:.4f}" for k, v in diff.items()]
    return "\n".join(lines)
