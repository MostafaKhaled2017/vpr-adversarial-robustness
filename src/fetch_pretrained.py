"""Write a model's official release weights as a plain checkpoint for --resume.

    python -m src.fetch_pretrained --model dinomix --out checkpoints/dinomix_dinov2_4096.pth

The model is built by its adapter, the release is loaded with strict=True (download_weights), and
{"model_state_dict": ...} is saved. An existing --out is never overwritten.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch

from .models import get_model_adapter

MODELS = ("dinomix", "qaa")


def fetch(model_name: str, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"{out} already exists; remove it first to fetch again.")
    adapter = get_model_adapter(model_name)
    args = types.SimpleNamespace(freeze_te=None, device="cpu")
    model = adapter.build(args).model
    adapter.download_weights(model, args)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        fetch(args.model, args.out)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
