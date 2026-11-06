#!/usr/bin/env python3
"""Write the backbone of one checkpoint under the head of another to OUT.

Usage:
    python scripts/transplant.py BACKBONE_CKPT HEAD_CKPT OUT

Every "backbone.*" tensor comes from BACKBONE_CKPT, every other tensor from HEAD_CKPT, so
OUT has exactly HEAD_CKPT's keys. BACKBONE_CKPT may be a different model (for example a
token-descriptor model built on the same ViT) as long as its backbone tensors match the head
checkpoint's by name and shape; anything else is refused before OUT is written.
"""
import argparse
from collections import OrderedDict
from typing import Mapping

import torch

from src.checkpoints import (
    atomic_torch_save,
    extract_model_state_dict,
    load_trusted_checkpoint,
    normalize_state_dict_keys,
)

BACKBONE_PREFIX = "backbone."
MAX_LISTED_KEYS = 10


def load_state_dict(path: str) -> "OrderedDict[str, torch.Tensor]":
    """Model weights of a checkpoint file on the CPU, without a DataParallel prefix."""
    return normalize_state_dict_keys(extract_model_state_dict(load_trusted_checkpoint(path, map_location="cpu")))


def _listed(keys) -> str:
    keys = sorted(keys)
    more = f" and {len(keys) - MAX_LISTED_KEYS} more" if len(keys) > MAX_LISTED_KEYS else ""
    return ", ".join(keys[:MAX_LISTED_KEYS]) + more


def transplant_state_dicts(
    backbone: Mapping[str, torch.Tensor], head: Mapping[str, torch.Tensor]
) -> "tuple[OrderedDict[str, torch.Tensor], int, int]":
    """``head`` with its backbone tensors replaced by ``backbone``'s; also the two counts."""
    donated = {key for key in backbone if key.startswith(BACKBONE_PREFIX)}
    wanted = {key for key in head if key.startswith(BACKBONE_PREFIX)}
    if not wanted:
        raise SystemExit("the head checkpoint has no backbone.* tensors to replace")
    if donated - wanted:
        raise SystemExit(f"backbone tensors the head checkpoint does not have: {_listed(donated - wanted)}")
    if wanted - donated:
        raise SystemExit(f"backbone tensors missing from the backbone checkpoint: {_listed(wanted - donated)}")
    reshaped = [key for key in sorted(wanted) if tuple(backbone[key].shape) != tuple(head[key].shape)]
    if reshaped:
        key = reshaped[0]
        raise SystemExit(
            f"{len(reshaped)} backbone tensors differ in shape, e.g. {key} "
            f"{tuple(backbone[key].shape)} vs {tuple(head[key].shape)}"
        )
    mixed = OrderedDict((key, (backbone[key] if key in wanted else value).clone()) for key, value in head.items())
    return mixed, len(wanted), len(head) - len(wanted)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Backbone of one checkpoint under the head of another.")
    parser.add_argument("backbone_ckpt", help="checkpoint whose backbone.* tensors are used")
    parser.add_argument("head_ckpt", help="checkpoint whose other tensors (and key set) are used")
    parser.add_argument("out", help="output checkpoint")
    args = parser.parse_args(argv)
    mixed, from_backbone, from_head = transplant_state_dicts(load_state_dict(args.backbone_ckpt), load_state_dict(args.head_ckpt))
    atomic_torch_save({"model_state_dict": mixed}, args.out)
    print(f"{from_backbone} backbone tensors from {args.backbone_ckpt}, {from_head} head tensors from {args.head_ckpt}: {args.out}")


if __name__ == "__main__":
    main()
