import os
import random
import shutil
import json
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from os.path import exists, join
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
import yaml


def load_trusted_checkpoint(path: str, map_location=None):
    return torch.load(path, map_location=map_location, weights_only=False)


def extract_model_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dictionary, got {type(checkpoint).__name__}")
    for key in ("model_state_dict", "state_dict"):
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, dict):
            return state_dict
    return checkpoint


def normalize_state_dict_keys(state_dict):
    normalized = OrderedDict(state_dict)
    for prefix in ("module.", "model."):
        if normalized and all(key.startswith(prefix) for key in normalized):
            normalized = OrderedDict((key[len(prefix) :], value) for key, value in normalized.items())
    return normalized


def load_model_state_dict(model, checkpoint, strict: bool = True) -> None:
    state_dict = normalize_state_dict_keys(extract_model_state_dict(checkpoint))
    model.load_state_dict(state_dict, strict=strict)


def load_model_weights(model, checkpoint_path: str, map_location=None, strict: bool = True) -> None:
    checkpoint = load_trusted_checkpoint(checkpoint_path, map_location=map_location)
    load_model_state_dict(model, checkpoint, strict=strict)


def checkpoint_next_epoch(checkpoint) -> int:
    if "next_epoch" in checkpoint:
        return int(checkpoint["next_epoch"])
    return int(checkpoint["epoch_num"]) + 1


@dataclass(frozen=True)
class TrainingResumeState:
    next_epoch: int
    best_score: float
    not_improved: int
    runtime_state: Mapping[str, object]


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Mapping[str, object]]) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([rng_state.cpu() for rng_state in state["torch_cuda"]])


def load_training_state(
    checkpoint_path: str | Path,
    model,
    optimizer,
    scaler=None,
    *,
    map_location=None,
    strict: bool = False,
) -> TrainingResumeState:
    checkpoint = load_trusted_checkpoint(str(checkpoint_path), map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    runtime_state = dict(checkpoint.get("runtime_state", {}))
    # Keep legacy checkpoints distinguishable from a fresh run even when they do not
    # contain the newer runtime payload. This prevents repeating initial validation.
    runtime_state["resumed_from_checkpoint"] = True
    if checkpoint.get("rng_state") is not None:
        runtime_state["rng_state"] = checkpoint["rng_state"]
    return TrainingResumeState(
        next_epoch=checkpoint_next_epoch(checkpoint),
        best_score=float(checkpoint["best_r5"]),
        not_improved=int(checkpoint["not_improved_num"]),
        runtime_state=runtime_state,
    )


def resolve_run_directory(log_dir: str, save_dir: str, run_dir, timestamp: str) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    return Path(log_dir) / save_dir / timestamp


def atomic_torch_save(state: Mapping[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, destination)


def atomic_copy(source: str | Path, destination: str | Path) -> None:
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def save_checkpoint(args, state: Mapping[str, object], is_best: bool, filename: str) -> None:
    model_path = Path(args.save_dir) / filename
    atomic_torch_save(state, model_path)
    if is_best:
        atomic_copy(model_path, Path(args.save_dir) / "best_model.pth")


def write_run_status(run_dir: str | Path, state: str, **details) -> Path:
    destination = Path(run_dir) / "run_status.json"
    payload = {
        "state": state,
        "updated_at": datetime.now().astimezone().isoformat(),
        **details,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def append_resume_history(run_dir: str | Path, checkpoint: str | Path, next_epoch: int) -> Path:
    path = Path(run_dir) / "resume_history.jsonl"
    record = {
        "resumed_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint),
        "next_epoch": int(next_epoch),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def _to_yaml_serializable(value):
    if isinstance(value, dict):
        return {key: _to_yaml_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_yaml_serializable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_training_config(args, filename: str = "training_config.yaml") -> str:
    config = {key: _to_yaml_serializable(value) for key, value in vars(args).items()}
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    config_path = join(args.save_dir, filename)
    with open(config_path, "w") as config_file:
        yaml.safe_dump(config, config_file, default_flow_style=False, sort_keys=True)
    return config_path


def maybe_copy_resume_checkpoint(args, model=None) -> None:
    if args.resume is None and not args.download_pretrained:
        return

    initial_checkpoint_path = join(args.save_dir, "initial_model.pth")
    if not exists(initial_checkpoint_path):
        Path(initial_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        if args.resume is not None:
            import shutil

            shutil.copyfile(args.resume, initial_checkpoint_path)
        elif model is not None:
            unwrapped_model = model.module if hasattr(model, "module") else model
            torch.save({"model_state_dict": unwrapped_model.state_dict()}, initial_checkpoint_path)


def apply_lr_schedule(optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def should_drop_lr_on_plateau(not_improved: int, plateau_patience) -> bool:
    if plateau_patience is None:
        return False
    return not_improved > 0 and not_improved % plateau_patience == 0


def maybe_remove_old_checkpoint(args, checkpoint_epoch: int) -> None:
    if args.keep_every <= 1:
        return
    if checkpoint_epoch % args.keep_every == 0:
        return
    checkpoint_name = join(args.save_dir, f"checkpoint_epoch_{checkpoint_epoch:04d}.pth")
    if exists(checkpoint_name):
        os.remove(checkpoint_name)


def copy_budget_checkpoints(args, source_filename: str, budget_keys) -> None:
    """Copy ``source_filename`` to ``best_model_budget<b>.pth`` for each improved budget (spec D2)."""
    for key in budget_keys:
        atomic_copy(Path(args.save_dir) / source_filename, Path(args.save_dir) / f"best_model_budget{key}.pth")
