#!/usr/bin/env python3
"""Write the weight average (1 - ALPHA) * A + ALPHA * B of two checkpoints to OUT.

Usage:
    python scripts/soup.py A B ALPHA OUT

A and B must hold the same model (identical keys and shapes); anything else is refused
before OUT is written. Floating-point tensors are averaged and stored in A's dtype; integer
and boolean buffers are copied from A. ALPHA = 0 reproduces A and ALPHA = 1 reproduces B.
OUT holds {"model_state_dict": ...}, which every model's load_weights accepts.
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

MAX_LISTED_KEYS = 10


def load_state_dict(path: str) -> "OrderedDict[str, torch.Tensor]":
    """Model weights of a checkpoint file on the CPU, without a DataParallel prefix."""
    return normalize_state_dict_keys(extract_model_state_dict(load_trusted_checkpoint(path, map_location="cpu")))


def check_same_layout(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> None:
    """Exit unless ``a`` and ``b`` have the same keys and shapes: two different models do not mix."""
    differing = sorted(set(a) ^ set(b))
    if differing:
        listed = ", ".join(differing[:MAX_LISTED_KEYS])
        more = f" and {len(differing) - MAX_LISTED_KEYS} more" if len(differing) > MAX_LISTED_KEYS else ""
        raise SystemExit(f"the checkpoints hold different models: {len(differing)} keys in only one of them: {listed}{more}")
    reshaped = [key for key in a if tuple(a[key].shape) != tuple(b[key].shape)]
    if reshaped:
        key = reshaped[0]
        raise SystemExit(
            f"the checkpoints hold different models: {len(reshaped)} tensors differ in shape, "
            f"e.g. {key} {tuple(a[key].shape)} vs {tuple(b[key].shape)}"
        )


def soup_state_dicts(
    a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor], alpha: float
) -> "tuple[OrderedDict[str, torch.Tensor], int]":
    """``(1 - alpha) * a + alpha * b`` per floating-point tensor, and how many were averaged."""
    check_same_layout(a, b)
    mixed = OrderedDict()
    interpolated = 0
    for key, ta in a.items():
        if not torch.is_floating_point(ta):
            mixed[key] = ta.clone()
            continue
        # Half-precision tensors are mixed in float32 so the average does not round twice.
        compute = torch.promote_types(ta.dtype, torch.float32)
        tb = b[key].to(compute)
        mixed[key] = ((1.0 - alpha) * ta.to(compute) + alpha * tb).to(ta.dtype)
        interpolated += 1
    return mixed, interpolated


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Weight average (1 - ALPHA) * A + ALPHA * B of two checkpoints.")
    parser.add_argument("a", help="first checkpoint (ALPHA = 0 reproduces it)")
    parser.add_argument("b", help="second checkpoint (ALPHA = 1 reproduces it)")
    parser.add_argument("alpha", type=float, help="mixing weight of B")
    parser.add_argument("out", help="output checkpoint")
    args = parser.parse_args(argv)
    mixed, interpolated = soup_state_dicts(load_state_dict(args.a), load_state_dict(args.b), args.alpha)
    atomic_torch_save({"model_state_dict": mixed}, args.out)
    print(f"{interpolated} float tensors interpolated, {len(mixed) - interpolated} copied from A, alpha {args.alpha:g}: {args.out}")


if __name__ == "__main__":
    main()
