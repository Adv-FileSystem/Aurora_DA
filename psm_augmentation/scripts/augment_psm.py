"""Command-line entry point for PSM data augmentation.

Example:
    python scripts/augment_psm.py --root ./data/PSM --copies 3
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from psm_aug.pipeline import run, save_result  # noqa: E402
from psm_aug.stats import format_report  # noqa: E402
from psm_aug.utils import load_config, set_seed  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Augment PSM time-series windows.")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--root", help="override data.root")
    p.add_argument("--split", choices=["train", "test"], help="override data.split")
    p.add_argument("--copies", type=int, help="override augment.copies")
    p.add_argument("--out", help="override output.dir")
    p.add_argument("--seed", type=int, help="override seed")
    return p.parse_args()


def apply_overrides(cfg, args):
    if args.root:
        cfg["data"]["root"] = args.root
    if args.split:
        cfg["data"]["split"] = args.split
    if args.copies is not None:
        cfg["augment"]["copies"] = args.copies
    if args.out:
        cfg["output"]["dir"] = args.out
    if args.seed is not None:
        cfg["seed"] = args.seed
    return cfg


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    set_seed(cfg.get("seed"))
    result = run(cfg)
    path = save_result(result, cfg["output"]["dir"], cfg)
    print(f"saved {len(result.windows)} windows -> {path}")
    original = result.windows[~result.is_augmented]
    augmented = result.windows[result.is_augmented]
    if len(augmented):
        print(format_report(original, augmented))


if __name__ == "__main__":
    main()
