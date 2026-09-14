"""Persistent run-group metadata for restartable Phase 2 pilot grids."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Mapping, Optional

import yaml


MANIFEST_NAME = "pilot_manifest.yaml"
TERMINAL_STATES = frozenset({"completed", "early_stopped", "collapse_aborted"})


@dataclass(frozen=True)
class ConfigRunState:
    state: str
    checkpoint: Optional[Path] = None


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _configuration_rows(values: Mapping[str, object]):
    return [
        {
            "label": f"tau{tau}_k{k}_pool{pool}",
            "tau": str(tau),
            "k": str(k),
            "pool": str(pool),
        }
        for tau, k, pool in product(values["taus"], values["ks"], values["pools"])
    ]


def _normalized_manifest(values: Mapping[str, object]) -> dict:
    normalized = {
        "schema_version": 1,
        "stage": "phase2_pilot",
        "taus": [str(value) for value in values["taus"]],
        "ks": [str(value) for value in values["ks"]],
        "pools": [str(value) for value in values["pools"]],
        "seed": int(values["seed"]),
        "ramp_epochs": int(values["ramp_epochs"]),
        "abort_knn": str(values["abort_knn"]),
        "base_checkpoint": str(values["base_checkpoint"]),
        "from_scratch": bool(values.get("from_scratch", False)),
    }
    normalized["configurations"] = _configuration_rows(normalized)
    return normalized


def _atomic_yaml(path: Path, content: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(content), sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def load_pilot_manifest(run_root: Path) -> dict:
    manifest_path = Path(run_root) / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"Not a Phase 2 pilot run root: missing {manifest_path}")
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle)
    if not isinstance(manifest, dict) or manifest.get("stage") != "phase2_pilot":
        raise ValueError(f"Invalid Phase 2 pilot manifest: {manifest_path}")
    return manifest


def create_pilot_run_group(
    base_dir: Path,
    values: Mapping[str, object],
    *,
    run_root: Optional[Path] = None,
) -> Path:
    expected = _normalized_manifest(values)
    if run_root is not None:
        resolved_root = Path(run_root)
        existing = load_pilot_manifest(resolved_root)
        comparable = {key: existing.get(key) for key in expected}
        if comparable != expected:
            raise ValueError("Existing pilot manifest does not match the requested pilot configuration")
        return resolved_root

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    resolved_root = Path(base_dir) / timestamp
    resolved_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        **expected,
        "created_at": datetime.now().astimezone().isoformat(),
        "git_revision": _git_revision(),
    }
    _atomic_yaml(resolved_root / MANIFEST_NAME, manifest)
    for config in manifest["configurations"]:
        (resolved_root / config["label"]).mkdir()
    return resolved_root


def classify_config_run(config_dir: Path) -> ConfigRunState:
    config_dir = Path(config_dir)
    status_path = config_dir / "run_status.json"
    if status_path.is_file():
        try:
            state = json.loads(status_path.read_text(encoding="utf-8")).get("state")
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Invalid run status file: {status_path}") from exc
        if state in TERMINAL_STATES:
            return ConfigRunState(str(state))

    for checkpoint_name in ("last_model.pth", "initial_validation_model.pth"):
        checkpoint = config_dir / checkpoint_name
        if checkpoint.is_file():
            return ConfigRunState("resumable", checkpoint)

    unexpected = [entry for entry in config_dir.iterdir() if entry.name != "run_status.json"]
    if unexpected:
        return ConfigRunState("invalid")
    return ConfigRunState("pending")


def _manifest_values(manifest: Mapping[str, object]) -> str:
    fields = (
        " ".join(manifest["taus"]),
        " ".join(manifest["ks"]),
        " ".join(manifest["pools"]),
        str(manifest["seed"]),
        str(manifest["ramp_epochs"]),
        str(manifest["abort_knn"]),
        str(manifest["base_checkpoint"]),
        "1" if manifest.get("from_scratch", False) else "0",
    )
    return "\t".join(fields)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create")
    create.add_argument("--base-dir", required=True)
    create.add_argument("--taus", nargs="+", required=True)
    create.add_argument("--ks", nargs="+", required=True)
    create.add_argument("--pools", nargs="+", required=True)
    create.add_argument("--seed", type=int, required=True)
    create.add_argument("--ramp-epochs", type=int, required=True)
    create.add_argument("--abort-knn", required=True)
    create.add_argument("--base-checkpoint", required=True)
    create.add_argument("--from-scratch", action="store_true")

    values = subparsers.add_parser("values")
    values.add_argument("--run-root", required=True)

    classify = subparsers.add_parser("classify")
    classify.add_argument("--config-dir", required=True)

    args = parser.parse_args()
    if args.command == "create":
        root = create_pilot_run_group(
            Path(args.base_dir),
            {
                "taus": args.taus,
                "ks": args.ks,
                "pools": args.pools,
                "seed": args.seed,
                "ramp_epochs": args.ramp_epochs,
                "abort_knn": args.abort_knn,
                "base_checkpoint": args.base_checkpoint,
                "from_scratch": args.from_scratch,
            },
        )
        print(root)
    elif args.command == "values":
        print(_manifest_values(load_pilot_manifest(Path(args.run_root))))
    else:
        state = classify_config_run(Path(args.config_dir))
        print(f"{state.state}\t{state.checkpoint or ''}")


if __name__ == "__main__":
    main()
