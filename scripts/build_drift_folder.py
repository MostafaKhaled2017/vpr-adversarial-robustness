#!/usr/bin/env python3
"""Sample unlabeled images from MSLS-train city folders into one folder for the drift constraint.

Usage:
    python scripts/build_drift_folder.py CITY_DIR... --out DIR [--count 20000] [--seed 0] [--symlink]

Every image file under the CITY_DIRs (searched recursively, so the release layout
<city>/{database,query}/images/ works as is) is pooled, and --count of them are drawn without
replacement with --seed, so the same inputs and seed give the same folder. They are copied (or
linked with --symlink) into DIR as <city>_<path inside the city folder>, and DIR/sources.txt
lists where each came from. A city folder named after an MSLS validation or test city is
refused: the constraint's images must come from places the validation and test sets do not
show. DIR must be empty or absent.
"""
import argparse
import os
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from src.drift_images import list_images

# MSLS validation (cph, sf) and test cities.
HELD_OUT_CITIES = {"cph", "sf", "miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"}


def sample_images(city_dirs: Sequence[Path], count: int, seed: int) -> List[Tuple[Path, str]]:
    """``count`` (source path, output name) pairs drawn from every image of the city folders."""
    pool: List[Tuple[Path, str]] = []
    for city_dir in city_dirs:
        city_dir = Path(city_dir)
        if city_dir.name.lower() in HELD_OUT_CITIES:
            raise SystemExit(f"{city_dir} is an MSLS validation or test city")
        if not city_dir.is_dir():
            raise SystemExit(f"not a folder: {city_dir}")
        images = list_images(city_dir)
        if not images:
            raise SystemExit(f"no image files under {city_dir}")
        pool += [(path, f"{city_dir.name}_{'_'.join(path.relative_to(city_dir).parts)}") for path in images]
    if count >= len(pool):
        return pool
    chosen = np.sort(np.random.default_rng(seed).choice(len(pool), size=count, replace=False))
    return [pool[index] for index in chosen]


def write_folder(samples: Sequence[Tuple[Path, str]], out: Path, symlink: bool) -> None:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    for source, name in samples:
        if symlink:
            os.symlink(source.resolve(), out / name)
        else:
            shutil.copyfile(source, out / name)
    (out / "sources.txt").write_text("".join(f"{name}\t{source}\n" for source, name in samples), encoding="utf-8")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Sample unlabeled city images into one folder for the drift constraint.")
    parser.add_argument("city_dirs", nargs="+", type=Path, help="MSLS-train city folders")
    parser.add_argument("--out", type=Path, required=True, help="output folder (empty or absent)")
    parser.add_argument("--count", type=int, default=20000, help="images to draw (all of them if fewer)")
    parser.add_argument("--seed", type=int, default=0, help="seed of the draw")
    parser.add_argument("--symlink", action="store_true", help="link the images instead of copying them")
    args = parser.parse_args(argv)
    if args.count < 1:
        raise SystemExit("--count must be at least 1")
    samples = sample_images(args.city_dirs, args.count, args.seed)
    write_folder(samples, args.out, args.symlink)
    print(f"{len(samples)} images from {len(args.city_dirs)} city folders written to {args.out}")


if __name__ == "__main__":
    main()
