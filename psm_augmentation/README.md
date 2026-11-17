# PSM Data Augmentation

Time-series data augmentation toolkit for the **PSM (Pooled Server Metrics)** anomaly-detection dataset.


## Dataset layout

```
data/PSM/
├── train.csv        # timestamp_(min), feature_0 ... feature_24
├── test.csv         # timestamp_(min), feature_0 ... feature_24
└── test_label.csv   # timestamp_(min), label
```

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# run with the default config
python scripts/augment_psm.py --root ./data/PSM

# augment the test split, 3 copies per window
python scripts/augment_psm.py --root ./data/PSM --split test --copies 3 --out ./outputs/test_aug
```

Output: `<output.dir>/psm_augmented.npz` with

| key | shape | description |
|---|---|---|
| `windows` | (N, L, C) | original + augmented windows |
| `labels` | (N,) | window label (0 normal, 1 anomaly) |
| `is_augmented` | (N,) | `True` for generated windows |

The config used for the run is saved next to it as `config.yaml`.

## Transforms

| name | description | main params |
|---|---|---|
| `jitter` | add Gaussian noise | `sigma` |
| `scaling` | multiply each channel by a random factor | `sigma` |
| `magnitude_warp` | multiply by a smooth random curve | `sigma`, `knots` |
| `time_warp` | smoothly speed up / slow down time | `sigma`, `knots` |
| `window_slice` | crop a sub-window and stretch it back | `ratio` |
| `window_warp` | stretch or squeeze a short segment | `window_ratio`, `scales` |
| `permutation` | shuffle time segments (breaks temporal order, use with care) | `max_segments` |
| mixup | blend two normal windows (`augment.mixup`) | `alpha`, `ratio` |

Each transform in `augment.transforms` is applied with probability `p`.
`augment.anomaly_policy: skip` augments normal windows only.

## Tests

```bash
pytest -q
```
