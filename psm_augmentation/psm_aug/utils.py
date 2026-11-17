"""Config loading and seeding helpers."""
import random

import numpy as np
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
