import numpy as np
import pytest

from psm_aug import transforms as T

L, C = 64, 5


@pytest.fixture
def window():
    rng = np.random.default_rng(0)
    return rng.random((L, C)).astype(np.float32)


@pytest.mark.parametrize(
    "transform",
    [
        T.Jitter(0.05),
        T.Scaling(0.1),
        T.MagnitudeWarp(0.2, 4),
        T.TimeWarp(0.2, 4),
        T.WindowSlice(0.8),
        T.WindowWarp(0.1, (0.5, 2.0)),
        T.Permutation(4),
    ],
    ids=lambda t: t.__class__.__name__,
)
def test_shape_and_finite(transform, window):
    out = transform(window, np.random.default_rng(1))
    assert out.shape == window.shape
    assert np.isfinite(out).all()


def test_random_apply_p0_is_identity(window):
    t = T.RandomApply(T.Jitter(1.0), p=0.0)
    np.testing.assert_array_equal(t(window, np.random.default_rng(0)), window)


def test_mixup_lambda(window):
    rng = np.random.default_rng(2)
    mixed, lam = T.mixup(window, window[::-1].copy(), rng, alpha=0.2)
    assert mixed.shape == window.shape
    assert 0.5 <= lam <= 1.0
