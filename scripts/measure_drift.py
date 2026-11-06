#!/usr/bin/env python3
"""Mean squared descriptor distance of a checkpoint from its model's pretrained release.

Usage:
    python scripts/measure_drift.py MODEL CHECKPOINT FOLDER [--batch 16] [--anchor RELEASE]

MODEL is supervlad or boq; the model flags and the release checkpoint are
the ones scripts/common.sh gives scripts/train.sh. Every image under FOLDER is encoded once by
both models, in batches of --batch at the training image size, the same way the drift
constraint of train.py (--ood_drift_folder) measures it: ||f(x) - f0(x)||^2 per image. The
mean is the value to pass as the constraint's allowed drift (--ood_drift_rho).
"""
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "supervlad": ("SUPERVLAD_TRAIN_FLAGS", "SUPERVLAD_PRETRAINED"),
    "boq": ("BOQ_TRAIN_FLAGS", "BOQ_PRETRAINED"),
}


def common_settings(model: str) -> Tuple[List[str], str]:
    """The model's train.py flags and release checkpoint, read from scripts/common.sh."""
    flags_name, release_name = MODELS[model]
    script = f'source scripts/common.sh; printf "%s\\0" "${{{flags_name}[@]}}" "${{{release_name}}}"'
    output = subprocess.run(["bash", "-c", script], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout
    values = output.split("\0")[:-1]
    return values[:-1], values[-1]


def build_model(args, adapter, checkpoint: str):
    bundle = adapter.build(args)
    model = bundle.model.to(args.device)
    adapter.load_weights(model, checkpoint, args)
    model.eval()
    model.requires_grad_(False)
    return model


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Descriptor drift of a checkpoint from its pretrained release.")
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("checkpoint")
    parser.add_argument("folder")
    parser.add_argument("--batch", type=int, default=16, help="images per forward pass, as --ood_drift_batch")
    parser.add_argument("--anchor", default=None, help="reference checkpoint (default: the model's release)")
    options = parser.parse_args(argv)

    from src.cli import parse_arguments
    from src.config import amp_autocast
    from src.drift_images import DriftBatches
    from src.models import get_model_adapter

    flags, release = common_settings(options.model)
    anchor_path = options.anchor or release
    args = parse_arguments([*flags, "--eval_datasets_folder=datasets", "--mixed_precision", f"--resume={options.checkpoint}"])
    adapter = get_model_adapter(args.model)
    model = build_model(args, adapter, options.checkpoint)
    anchor = build_model(args, adapter, anchor_path)

    images = DriftBatches(options.folder, args.train_resize, options.batch, seed=0, pin_memory=args.device == "cuda")
    batches = -(-len(images) // options.batch)
    drifts = []
    with torch.no_grad():
        for _ in range(batches):
            batch = images.next().to(args.device)
            with amp_autocast(args.mixed_precision, args.device):
                current = model(batch, queryflag=0).float()
                reference = anchor(batch, queryflag=0).float()
            drifts.append(((current - reference) ** 2).sum(dim=-1).cpu().numpy())
    drift = np.concatenate(drifts)[: len(images)]
    print(
        f"drift of {options.checkpoint} from {anchor_path} on {len(images)} images of {options.folder}: "
        f"mean {drift.mean():.6f}, std {drift.std():.6f}"
    )


if __name__ == "__main__":
    main()
