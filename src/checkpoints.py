import os
from collections import OrderedDict
from os.path import exists, join
from pathlib import Path

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
