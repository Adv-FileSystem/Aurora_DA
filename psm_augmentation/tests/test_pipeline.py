import numpy as np
import pandas as pd

from psm_aug.pipeline import run, save_result

N_FEAT = 4


def _write_fake_psm(root, n_train=300, n_test=200):
    rng = np.random.default_rng(0)
    cols = [f"feature_{i}" for i in range(N_FEAT)]
    for name, n in (("train", n_train), ("test", n_test)):
        df = pd.DataFrame(rng.random((n, N_FEAT)), columns=cols)
        df.insert(0, "timestamp_(min)", np.arange(n))
        df.iloc[5, 1] = np.nan
        df.to_csv(root / f"{name}.csv", index=False)
    labels = np.zeros(n_test, dtype=int)
    labels[50:70] = 1
    pd.DataFrame({"timestamp_(min)": np.arange(n_test), "label": labels}).to_csv(
        root / "test_label.csv", index=False
    )


def _cfg(root, split="train", copies=2):
    return {
        "data": {"root": str(root), "split": split},
        "seed": 0,
        "preprocess": {"normalize": True},
        "window": {"size": 20, "stride": 10, "label_rule": "any"},
        "augment": {
            "copies": copies,
            "transforms": [{"name": "jitter", "p": 1.0, "params": {"sigma": 0.01}}],
            "mixup": {"enabled": False},
        },
    }


def test_run_train_split(tmp_path):
    _write_fake_psm(tmp_path)
    res = run(_cfg(tmp_path))
    n_orig = (~res.is_augmented).sum()
    assert len(res.windows) == n_orig * 3
    assert res.windows.shape[1:] == (20, N_FEAT)
    assert np.isfinite(res.windows).all()
    assert (res.labels == 0).all()


def test_run_test_split_keeps_labels(tmp_path):
    _write_fake_psm(tmp_path)
    res = run(_cfg(tmp_path, split="test", copies=1))
    assert res.labels.sum() > 0
    orig = res.labels[~res.is_augmented]
    aug = res.labels[res.is_augmented]
    np.testing.assert_array_equal(orig, aug)


def test_save_result(tmp_path):
    _write_fake_psm(tmp_path)
    res = run(_cfg(tmp_path, copies=1))
    path = save_result(res, tmp_path / "out")
    data = np.load(path)
    assert data["windows"].shape == res.windows.shape
