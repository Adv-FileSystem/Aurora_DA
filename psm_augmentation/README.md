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

## Tests

```bash
pytest -q
```
