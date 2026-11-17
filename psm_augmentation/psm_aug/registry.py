"""Build transform pipelines from config."""
from . import transforms as T

REGISTRY = {
    "jitter": T.Jitter,
    "scaling": T.Scaling,
    "magnitude_warp": T.MagnitudeWarp,
    "time_warp": T.TimeWarp,
    "window_slice": T.WindowSlice,
    "window_warp": T.WindowWarp,
    "permutation": T.Permutation,
}


def build_transform(cfg_list):
    """``cfg_list``: list of ``{name, p, params}`` dicts -> Compose."""
    items = []
    for item in cfg_list or []:
        name = item["name"]
        if name not in REGISTRY:
            raise KeyError(f"unknown transform {name!r}; available: {sorted(REGISTRY)}")
        t = REGISTRY[name](**(item.get("params") or {}))
        p = float(item.get("p", 1.0))
        items.append(T.RandomApply(t, p) if p < 1.0 else t)
    return T.Compose(items)
