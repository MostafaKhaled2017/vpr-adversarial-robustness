#!/usr/bin/env python3
"""Run a budgeted Rank-PGD strength sweep with optional parallel jobs."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))
DEFAULT_OUTPUT_ROOT = Path("test/rank_eval/sweeps")
ALLOWED_DATASETS = {"msls", "sped"}
REQUIRED_RECALLS = (1, 5, 10, 100)


@dataclass(frozen=True)
class SweepConfig:
    eval_datasets_folder: str = "datasets"
    datasets: tuple[str, ...] = ("sped",)
    models: tuple[str, ...] = (
        "checkpoints/SuperVLAD.pth",
        "checkpoints/perceptual_adv_checkpoint.pth",
    )
    model_tags: tuple[str, ...] = ("base", "checkpoint")
    foundation_model_path: str = "checkpoints/dinov2_vitb14_pretrain.pth"
    backbone: str = "dino"
    supervlad_clusters: int = 4
    freeze_te: int = 8
    crossimage_encoder: bool = True
    infer_batch_size: int = 16
    test_method: str = "hard_resize"
    rank_attack: str = "rank_pgd_linf"
    epsilons: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.20)
    seed: int = 0
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    sweep_id: str | None = None
    max_experiments_per_dataset: int = 20
    parallel_runs: int = 1
    execution_mode: str = "in_process"
    max_dataset_samples: int | None = None
    compute_diagnostics: bool = False
    python_bin: str = sys.executable
    smoke: bool = False
    max_queries: int | None = None
    resume_sweep_dir: str | None = None
    force_rerun_completed: bool = False


@dataclass(frozen=True)
class SweepCondition:
    condition_id: str
    group: str
    epsilon: float
    rank_steps: int
    rank_step_size: float | None
    step_size_mode: str
    step_size_multiplier: float | None
    rank_restarts: int
    adv_negatives: int

    @property
    def resolved_step_size(self) -> float:
        if self.rank_step_size is not None:
            return float(self.rank_step_size)
        return 2.0 * float(self.epsilon) / float(self.rank_steps)

    @property
    def compute_budget(self) -> int:
        return int(self.rank_steps * self.rank_restarts * self.adv_negatives)


@dataclass
class JobResult:
    dataset: str
    condition_id: str
    command: list[str]
    returncode: int
    stdout_path: str
    stderr_path: str
    run_dir: str
    output_json: str | None
    output_csv: str | None
    started_at: str
    finished_at: str
    error: str | None = None


@dataclass(frozen=True)
class SweepJob:
    dataset: str
    condition: SweepCondition

    @property
    def job_id(self) -> str:
        return f"{self.dataset}_{self.condition.condition_id}"

    @property
    def condition_id(self) -> str:
        return self.condition.condition_id


@dataclass(frozen=True)
class JobState:
    job: SweepJob
    status: str
    run_dir: str
    output_json: str | None = None
    reason: str | None = None


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MSLS/SPED Rank-PGD strength sweep with optional parallel subprocesses."
    )
    parser.add_argument("--config", default=None, help="Optional JSON config file. CLI values override config values.")
    parser.add_argument("--eval_datasets_folder", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--model_tags", nargs="+", default=None)
    parser.add_argument("--foundation_model_path", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--supervlad_clusters", type=int, default=None)
    parser.add_argument("--freeze_te", type=int, default=None)
    parser.add_argument("--crossimage_encoder", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--infer_batch_size", type=int, default=None)
    parser.add_argument("--test_method", default=None)
    parser.add_argument("--rank_attack", default=None)
    parser.add_argument("--epsilons", nargs="+", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--sweep_id", default=None)
    parser.add_argument("--max_experiments_per_dataset", type=int, default=None)
    parser.add_argument("--parallel_runs", type=int, default=None)
    parser.add_argument("--execution_mode", choices=("in_process", "subprocess"), default=None)
    parser.add_argument("--max_dataset_samples", type=int, default=None)
    parser.add_argument(
        "--compute_diagnostics",
        action="store_true",
        default=None,
        help="Compute per-query diagnostic CSV files for every sweep condition.",
    )
    parser.add_argument("--python_bin", default=None)
    parser.add_argument(
        "--resume_sweep_dir",
        default=None,
        help="Existing sweep directory to resume, for example test/rank_eval/sweeps/<sweep_id>.",
    )
    parser.add_argument(
        "--force_rerun_completed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Rerun jobs even when valid result JSON files already exist.",
    )
    parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run smoke-mode commands with a query cap.",
    )
    parser.add_argument("--max_queries", type=int, default=None, help="Smoke-mode query cap. Defaults to 5 in smoke mode.")
    parser.add_argument("--dry_run", action="store_true", help="Print the planned commands without launching jobs.")
    return parser.parse_args(argv)


def _sequence_value(value: Any, cast_type: type) -> tuple[Any, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"Expected a list value, received {value!r}.")
    return tuple(cast_type(item) for item in value)


def load_config_file(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Sweep config must be a JSON object.")

    valid_keys = set(SweepConfig.__dataclass_fields__.keys())
    unknown = sorted(set(loaded) - valid_keys)
    if unknown:
        raise ValueError(f"Unknown config key(s): {', '.join(unknown)}")
    return loaded


def build_config(args: argparse.Namespace) -> SweepConfig:
    config_values: dict[str, Any] = asdict(SweepConfig())
    config_values.update(load_config_file(args.config))

    cli_overrides = {
        "eval_datasets_folder": args.eval_datasets_folder,
        "datasets": args.datasets,
        "models": args.models,
        "model_tags": args.model_tags,
        "foundation_model_path": args.foundation_model_path,
        "backbone": args.backbone,
        "supervlad_clusters": args.supervlad_clusters,
        "freeze_te": args.freeze_te,
        "crossimage_encoder": args.crossimage_encoder,
        "infer_batch_size": args.infer_batch_size,
        "test_method": args.test_method,
        "rank_attack": args.rank_attack,
        "epsilons": args.epsilons,
        "seed": args.seed,
        "output_root": args.output_root,
        "sweep_id": args.sweep_id,
        "max_experiments_per_dataset": args.max_experiments_per_dataset,
        "parallel_runs": args.parallel_runs,
        "execution_mode": args.execution_mode,
        "max_dataset_samples": args.max_dataset_samples,
        "compute_diagnostics": args.compute_diagnostics,
        "python_bin": args.python_bin,
        "smoke": args.smoke,
        "max_queries": args.max_queries,
        "resume_sweep_dir": args.resume_sweep_dir,
        "force_rerun_completed": args.force_rerun_completed,
    }
    config_values.update({key: value for key, value in cli_overrides.items() if value is not None})

    config = SweepConfig(
        eval_datasets_folder=str(config_values["eval_datasets_folder"]),
        datasets=_sequence_value(config_values["datasets"], str),
        models=_sequence_value(config_values["models"], str),
        model_tags=_sequence_value(config_values["model_tags"], str),
        foundation_model_path=str(config_values["foundation_model_path"]),
        backbone=str(config_values["backbone"]),
        supervlad_clusters=int(config_values["supervlad_clusters"]),
        freeze_te=int(config_values["freeze_te"]),
        crossimage_encoder=bool(config_values["crossimage_encoder"]),
        infer_batch_size=int(config_values["infer_batch_size"]),
        test_method=str(config_values["test_method"]),
        rank_attack=str(config_values["rank_attack"]),
        epsilons=_sequence_value(config_values["epsilons"], float),
        seed=int(config_values["seed"]),
        output_root=str(config_values["output_root"]),
        sweep_id=config_values["sweep_id"],
        max_experiments_per_dataset=int(config_values["max_experiments_per_dataset"]),
        parallel_runs=int(config_values["parallel_runs"]),
        execution_mode=str(config_values["execution_mode"]),
        max_dataset_samples=(
            None
            if config_values["max_dataset_samples"] is None
            else int(config_values["max_dataset_samples"])
        ),
        compute_diagnostics=bool(config_values["compute_diagnostics"]),
        python_bin=str(config_values["python_bin"]),
        smoke=bool(config_values["smoke"]),
        max_queries=(
            None
            if config_values["max_queries"] is None
            else int(config_values["max_queries"])
        ),
        resume_sweep_dir=(
            None
            if config_values["resume_sweep_dir"] is None
            else str(config_values["resume_sweep_dir"])
        ),
        force_rerun_completed=bool(config_values["force_rerun_completed"]),
    )
    if config.smoke and config.max_queries is None:
        config = replace(config, max_queries=5)
    validate_config(config)
    return config


def validate_config(config: SweepConfig) -> None:
    if not config.datasets:
        raise ValueError("At least one dataset is required.")
    invalid_datasets = sorted(set(config.datasets) - ALLOWED_DATASETS)
    if invalid_datasets:
        raise ValueError(f"This sweep supports only msls and sped, not: {', '.join(invalid_datasets)}")
    if len(config.models) != len(config.model_tags):
        raise ValueError("models and model_tags must have the same length.")
    if not config.epsilons:
        raise ValueError("At least one epsilon is required.")
    if any(epsilon < 0 for epsilon in config.epsilons):
        raise ValueError("epsilons must be non-negative.")
    if config.max_experiments_per_dataset < 1 or config.max_experiments_per_dataset > 20:
        raise ValueError("max_experiments_per_dataset must be between 1 and 20.")
    if config.parallel_runs < 1:
        raise ValueError("parallel_runs must be at least 1.")
    if config.execution_mode not in {"in_process", "subprocess"}:
        raise ValueError("execution_mode must be either 'in_process' or 'subprocess'.")
    if config.execution_mode == "in_process" and config.parallel_runs > len(config.datasets):
        raise ValueError("parallel_runs cannot exceed the number of selected datasets in in_process mode.")
    if config.max_dataset_samples is not None and config.max_dataset_samples < 1:
        raise ValueError("max_dataset_samples must be at least 1 when provided.")
    if config.max_queries is not None and not config.smoke:
        raise ValueError("--max_queries is allowed only with --smoke so final runs are not accidentally capped.")
    if config.max_queries is not None and config.max_queries < 1:
        raise ValueError("max_queries must be at least 1.")


def validate_condition_dependent_config(config: SweepConfig, conditions: Sequence[SweepCondition]) -> None:
    if config.max_dataset_samples is None:
        return
    max_negatives = max(condition.adv_negatives for condition in conditions)
    if config.max_dataset_samples < max_negatives + 1:
        raise ValueError(
            "max_dataset_samples must be at least the largest configured adv_negatives value plus one positive "
            f"({max_negatives + 1} for this sweep)."
        )


def default_conditions(epsilons: Sequence[float]) -> list[SweepCondition]:
    conditions: list[SweepCondition] = []

    def add(
        group: str,
        epsilon: float,
        steps: int,
        step_size: float | None,
        step_size_mode: str,
        step_size_multiplier: float | None,
        restarts: int,
        negatives: int,
    ) -> None:
        condition_number = len(conditions) + 1
        conditions.append(
            SweepCondition(
                condition_id=f"condition_{condition_number:02d}",
                group=group,
                epsilon=float(epsilon),
                rank_steps=int(steps),
                rank_step_size=step_size,
                step_size_mode=step_size_mode,
                step_size_multiplier=step_size_multiplier,
                rank_restarts=int(restarts),
                adv_negatives=int(negatives),
            )
        )

    for epsilon in epsilons:
        add("epsilon_baseline", epsilon, 20, None, "default_2eps_over_steps", None, 1, 5)

    for steps in (5, 10, 40):
        add("step_count_probe", 0.05, steps, None, "default_2eps_over_steps", None, 1, 5)

    for multiplier in (1.0, 4.0):
        epsilon = 0.05
        steps = 20
        add(
            "step_size_probe",
            epsilon,
            steps,
            multiplier * epsilon / steps,
            "multiplier_eps_over_steps",
            multiplier,
            1,
            5,
        )

    for restarts in (3, 5):
        add("restart_probe", 0.05, 20, None, "default_2eps_over_steps", None, restarts, 5)

    for negatives in (1, 10, 20):
        add("negative_count_probe", 0.05, 20, None, "default_2eps_over_steps", None, 1, negatives)

    for epsilon in epsilons:
        add("strong_candidate_sweep", epsilon, 40, None, "default_2eps_over_steps", None, 3, 10)

    return conditions


def selected_conditions(config: SweepConfig) -> list[SweepCondition]:
    conditions = default_conditions(config.epsilons)
    return conditions[: config.max_experiments_per_dataset]


def expand_jobs(config: SweepConfig, conditions: Sequence[SweepCondition]) -> list[SweepJob]:
    return [
        SweepJob(dataset=dataset, condition=condition)
        for dataset in config.datasets
        for condition in conditions
    ]


def portable_config(config: SweepConfig, conditions: Sequence[SweepCondition]) -> dict[str, Any]:
    return {
        "datasets": list(config.datasets),
        "model_tags": list(config.model_tags),
        "rank_attack": config.rank_attack,
        "seed": config.seed,
        "smoke": config.smoke,
        "max_queries": config.max_queries,
        "max_dataset_samples": config.max_dataset_samples,
        "max_experiments_per_dataset": config.max_experiments_per_dataset,
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "epsilon": condition.epsilon,
                "rank_steps": condition.rank_steps,
                "rank_step_size": condition.rank_step_size,
                "rank_restarts": condition.rank_restarts,
                "adv_negatives": condition.adv_negatives,
            }
            for condition in conditions
        ],
    }


def portable_signature(config: SweepConfig, conditions: Sequence[SweepCondition]) -> str:
    payload = json.dumps(portable_config(config, conditions), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _portable_config_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    if isinstance(manifest.get("portable_config"), dict):
        return manifest["portable_config"]
    if not isinstance(manifest.get("config"), dict) or not isinstance(manifest.get("conditions"), list):
        return None

    config = manifest["config"]
    return {
        "datasets": list(config.get("datasets", manifest.get("datasets", []))),
        "model_tags": list(config.get("model_tags", [])),
        "rank_attack": config.get("rank_attack", "rank_pgd_linf"),
        "seed": config.get("seed", 0),
        "smoke": config.get("smoke", False),
        "max_queries": config.get("max_queries"),
        "max_dataset_samples": config.get("max_dataset_samples"),
        "max_experiments_per_dataset": config.get(
            "max_experiments_per_dataset",
            manifest.get("experiment_count_per_dataset"),
        ),
        "conditions": [
            {
                "condition_id": condition.get("condition_id"),
                "epsilon": condition.get("epsilon"),
                "rank_steps": condition.get("rank_steps"),
                "rank_step_size": condition.get("rank_step_size"),
                "rank_restarts": condition.get("rank_restarts"),
                "adv_negatives": condition.get("adv_negatives"),
            }
            for condition in manifest["conditions"]
        ],
    }


def load_existing_manifest(sweep_dir: Path) -> dict[str, Any] | None:
    manifest_path = sweep_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Existing manifest is not a JSON object: {manifest_path}")
    return loaded


def validate_resume_manifest(
    existing_manifest: Mapping[str, Any] | None,
    config: SweepConfig,
    conditions: Sequence[SweepCondition],
) -> None:
    if existing_manifest is None:
        return
    existing_portable = _portable_config_from_manifest(existing_manifest)
    if existing_portable is None:
        return
    current_portable = portable_config(config, conditions)
    if existing_portable != current_portable:
        raise ValueError(
            "The existing sweep manifest does not match the current portable sweep grid. "
            "Changing parallel_runs, infer_batch_size, python_bin, model paths, or output paths is allowed; "
            "changing datasets, model_tags, attack, seed, smoke/query-cap mode, or condition settings requires a new sweep."
        )


def make_sweep_id(config: SweepConfig) -> str:
    if config.sweep_id:
        return config.sweep_id
    suffix = "smoke" if config.smoke else "full"
    return f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{suffix}"


def job_run_dir(sweep_dir: Path, job: SweepJob) -> Path:
    return sweep_dir / "runs" / job.dataset / job.condition_id


def legacy_condition_run_dir(sweep_dir: Path, job: SweepJob) -> Path:
    return sweep_dir / "runs" / job.condition_id


def command_for_job(config: SweepConfig, job: SweepJob, job_dir: Path) -> list[str]:
    condition = job.condition
    command = [
        config.python_bin,
        str(REPO_ROOT / "src" / "rank_eval.py"),
        f"--eval_datasets_folder={config.eval_datasets_folder}",
        "--datasets",
        job.dataset,
        "--model_type=supervlad",
        "--model_paths",
        *config.models,
        "--model_tags",
        *config.model_tags,
        f"--foundation_model_path={config.foundation_model_path}",
        f"--backbone={config.backbone}",
        f"--supervlad_clusters={config.supervlad_clusters}",
        f"--freeze_te={config.freeze_te}",
        f"--infer_batch_size={config.infer_batch_size}",
        f"--test_method={config.test_method}",
        f"--rank_attack={config.rank_attack}",
        f"--rank_steps={condition.rank_steps}",
        f"--rank_restarts={condition.rank_restarts}",
        f"--adv_negatives={condition.adv_negatives}",
        "--epsilons",
        f"{condition.epsilon:g}",
        f"--seed={config.seed}",
        f"--output_json={job_dir / 'rank_eval_results.json'}",
        f"--output_csv={job_dir / 'rank_eval_results.csv'}",
    ]
    if config.crossimage_encoder:
        command.append("--crossimage_encoder")
    if condition.rank_step_size is not None:
        command.append(f"--rank_step_size={condition.rank_step_size:g}")
    if config.max_queries is not None:
        command.append(f"--max_queries={config.max_queries}")
    if config.max_dataset_samples is not None:
        command.append(f"--max_dataset_samples={config.max_dataset_samples}")
    if config.compute_diagnostics:
        command.append("--compute_diagnostics")
    return command


def command_for_condition(config: SweepConfig, condition: SweepCondition, condition_dir: Path) -> list[str]:
    dataset = config.datasets[0] if config.datasets else "msls"
    return command_for_job(config, SweepJob(dataset=dataset, condition=condition), condition_dir)


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    paths = [str(REPO_ROOT), str(REPO_ROOT / "third_party" / "SuperVLAD")]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def find_single_file(root: Path, name: str) -> str | None:
    matches = sorted(root.rglob(name))
    return str(matches[-1]) if matches else None


def _has_recalls(metrics: Mapping[str, Any]) -> bool:
    recalls = metrics.get("recalls")
    if not isinstance(recalls, dict):
        return False
    return all(f"R@{recall}" in recalls for recall in REQUIRED_RECALLS)


def _filename_token(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)


def validate_result_report(report: Mapping[str, Any], job: SweepJob, config: SweepConfig) -> tuple[bool, str | None]:
    attack = report.get("attack", {})
    if isinstance(attack, dict) and attack.get("mode") not in {None, config.rank_attack}:
        return False, "attack mode does not match"

    results = report.get("results")
    if not isinstance(results, dict):
        return False, "missing results object"
    dataset_results = results.get(job.dataset)
    if not isinstance(dataset_results, dict):
        return False, f"missing dataset results for {job.dataset}"

    condition_name = f"{config.rank_attack}_eps_{job.condition.epsilon:g}"
    for model_tag in config.model_tags:
        model_results = dataset_results.get(model_tag)
        if not isinstance(model_results, dict):
            return False, f"missing model results for {model_tag}"
        clean_metrics = model_results.get("clean_attacked_subset")
        attacked_metrics = model_results.get(condition_name)
        if not isinstance(clean_metrics, dict) or not _has_recalls(clean_metrics):
            return False, f"missing clean recalls for {model_tag}"
        if not isinstance(attacked_metrics, dict) or not _has_recalls(attacked_metrics):
            return False, f"missing attacked recalls for {model_tag}"
        if not isinstance(attacked_metrics.get("attack_success"), dict):
            return False, f"missing attack_success for {model_tag}"
        if not isinstance(attacked_metrics.get("rank_displacement"), dict):
            return False, f"missing rank_displacement for {model_tag}"
        metadata = attacked_metrics.get("attack_metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("perturbation_norm"), dict):
            return False, f"missing perturbation metadata for {model_tag}"

    if config.compute_diagnostics:
        diagnostics_value = report.get("diagnostics_output_dir")
        if not isinstance(diagnostics_value, str) or not diagnostics_value:
            return False, "missing diagnostics output directory"
        diagnostics_dir = Path(diagnostics_value)
        epsilon_label = f"{job.condition.epsilon:g}"
        for model_tag in config.model_tags:
            filename = (
                f"{_filename_token(job.dataset)}_{_filename_token(model_tag)}_"
                f"{_filename_token(config.rank_attack)}_eps_{epsilon_label}.csv"
            )
            diagnostics_path = diagnostics_dir / filename
            if not diagnostics_path.is_file() or diagnostics_path.stat().st_size == 0:
                return False, f"missing diagnostic CSV for {model_tag}"
    return True, None


def validate_result_json(path: Path, job: SweepJob, config: SweepConfig) -> tuple[bool, str | None]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return False, str(exc)
    if not isinstance(report, dict):
        return False, "result JSON is not an object"
    return validate_result_report(report, job, config)


def find_valid_result_json(search_dir: Path, job: SweepJob, config: SweepConfig) -> tuple[Path | None, str | None]:
    if not search_dir.exists():
        return None, "run directory does not exist"

    invalid_reasons: list[str] = []
    for path in sorted(search_dir.rglob("rank_eval_results.json"), reverse=True):
        valid, reason = validate_result_json(path, job, config)
        if valid:
            return path, None
        invalid_reasons.append(f"{path}: {reason}")
    if invalid_reasons:
        return None, "; ".join(invalid_reasons)
    return None, "rank_eval_results.json was not found"


def inspect_job_state(sweep_dir: Path, job: SweepJob, config: SweepConfig) -> JobState:
    current_dir = job_run_dir(sweep_dir, job)
    valid_path, invalid_reason = find_valid_result_json(current_dir, job, config)
    if valid_path is not None:
        return JobState(job=job, status="completed", run_dir=str(current_dir), output_json=str(valid_path))

    legacy_dir = legacy_condition_run_dir(sweep_dir, job)
    legacy_valid_path, legacy_reason = find_valid_result_json(legacy_dir, job, config)
    if legacy_valid_path is not None:
        return JobState(
            job=job,
            status="completed_legacy",
            run_dir=str(legacy_dir),
            output_json=str(legacy_valid_path),
        )

    reason = invalid_reason
    if legacy_dir.exists() and legacy_reason:
        reason = f"{reason}; legacy layout: {legacy_reason}" if reason else legacy_reason
    status = "invalid" if current_dir.exists() or legacy_dir.exists() else "pending"
    return JobState(job=job, status=status, run_dir=str(current_dir), reason=reason)


def inspect_jobs(sweep_dir: Path, jobs: Sequence[SweepJob], config: SweepConfig) -> list[JobState]:
    return [inspect_job_state(sweep_dir, job, config) for job in jobs]


def pending_jobs_from_states(states: Sequence[JobState], force_rerun_completed: bool = False) -> list[SweepJob]:
    pending = []
    for state in states:
        if force_rerun_completed or state.status not in {"completed", "completed_legacy"}:
            pending.append(state.job)
    return pending


def run_job(config: SweepConfig, sweep_dir: Path, job: SweepJob) -> JobResult:
    job_dir = job_run_dir(sweep_dir, job)
    attempt_dir = job_dir / "attempts" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    command = command_for_job(config, job, job_dir)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    started_at = datetime.now().isoformat()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=_subprocess_env(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    finished_at = datetime.now().isoformat()
    output_json, validation_error = find_valid_result_json(job_dir, job, config)
    output_csv = find_single_file(job_dir, "rank_eval_results.csv")
    return JobResult(
        dataset=job.dataset,
        condition_id=job.condition_id,
        command=command,
        returncode=int(completed.returncode),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        run_dir=str(job_dir),
        output_json=str(output_json) if output_json is not None else None,
        output_csv=output_csv,
        started_at=started_at,
        finished_at=finished_at,
        error=(
            validation_error
            if completed.returncode == 0 and output_json is None
            else None
            if completed.returncode == 0
            else f"src/rank_eval.py exited with code {completed.returncode}"
        ),
    )


def run_jobs(
    config: SweepConfig,
    sweep_dir: Path,
    jobs: Sequence[SweepJob],
    on_update=None,
) -> list[JobResult]:
    results: list[JobResult] = []
    job_iter = iter(jobs)
    failed = False
    with ThreadPoolExecutor(max_workers=config.parallel_runs) as executor:
        pending = {}
        for _ in range(min(config.parallel_runs, len(jobs))):
            job = next(job_iter, None)
            if job is None:
                break
            pending[executor.submit(run_job, config, sweep_dir, job)] = job

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive subprocess orchestration
                    job_dir = job_run_dir(sweep_dir, job)
                    result = JobResult(
                        dataset=job.dataset,
                        condition_id=job.condition_id,
                        command=command_for_job(config, job, job_dir),
                        returncode=1,
                        stdout_path=str(job_dir / "attempts"),
                        stderr_path=str(job_dir / "attempts"),
                        run_dir=str(job_dir),
                        output_json=None,
                        output_csv=None,
                        started_at="",
                        finished_at=datetime.now().isoformat(),
                        error=str(exc),
                    )
                results.append(result)
                if on_update is not None:
                    on_update(results)
                if result.returncode != 0:
                    failed = True

            while not failed and len(pending) < config.parallel_runs:
                job = next(job_iter, None)
                if job is None:
                    break
                pending[executor.submit(run_job, config, sweep_dir, job)] = job

    return sorted(results, key=lambda item: (item.dataset, item.condition_id))


def rank_eval_args_for_job(config: SweepConfig, job: SweepJob, job_dir: Path):
    from src import rank_eval

    argv = command_for_job(config, job, job_dir)[2:]
    args = rank_eval.build_parser().parse_args(argv)
    return rank_eval.finalize_arguments(args)


def configure_rank_eval_output_dirs(args, job_dir: Path) -> None:
    args.trace_output_dir_path = job_dir / "traces"
    args.diagnostics_output_dir_path = job_dir / "diagnostics" if args.compute_diagnostics else None
    args.attack_image_output_dir_path = job_dir / "attack_images"
    args.save_dir = str(job_dir)


def write_condition_rank_eval_report(
    config: SweepConfig,
    job: SweepJob,
    args,
    dataset_results: Mapping[str, Mapping[str, object]],
    runtimes: Mapping[str, object],
    query_counts: Mapping[str, object],
    audit_results: Mapping[str, object],
    attack_image_manifest: Sequence[Mapping[str, object]],
    output_json: Path,
    output_csv: Path,
    started_at: datetime,
) -> None:
    from src import rank_eval

    results = {job.dataset: dataset_results}
    rows = rank_eval.flatten_rows(results, args.recall_values)
    attack_image_manifest_path = args.attack_image_output_dir_path / "attack_image_manifest.csv"
    report: dict[str, Any] = {
        "timestamp": started_at.isoformat(),
        "command": " ".join(shlex.quote(argument) for argument in command_for_job(config, job, output_json.parent)),
        "argv": command_for_job(config, job, output_json.parent),
        "model_type": args.model_type,
        "checkpoints": dict(zip(args.model_tags, args.model_paths)),
        "model_configuration": {
            "input_size": list(args.resize),
            "descriptor_dimensions": {
                model_tag: int(args.features_dim * args.supervlad_clusters)
                for model_tag in args.model_tags
            },
        },
        "datasets": [job.dataset],
        "arguments": rank_eval.serialize_args(args),
        "attack": {
            "mode": args.rank_attack,
            "epsilons": [float(job.condition.epsilon)],
            "scope": "queries_only",
            "epsilon_space": "normalized_image_tensor",
            "attack_reference_model": rank_eval.attack_reference_tag(args) if args.shared_attacks else None,
            "attack_generation": rank_eval.attack_generation_mode(args),
            "shared_attacks_across_models": bool(args.shared_attacks),
            "target_selection": {
                "adv_negatives": int(args.adv_negatives),
                "adv_margin": float(args.adv_margin),
                "max_queries": args.max_queries,
                "max_dataset_samples": config.max_dataset_samples,
                "sampled_gallery_mode": config.max_dataset_samples is not None,
                "audit_sample_database_size": args.audit_sample_database_size,
                "audit_sample_mode": args.audit_sample_database_size is not None,
            },
            "query_counts": {job.dataset: query_counts},
        },
        "results": results,
        "runtime_seconds": {job.dataset: runtimes},
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "audit_output_json": None,
        "trace_output_dir": str(args.trace_output_dir_path),
        "diagnostics_output_dir": (
            str(args.diagnostics_output_dir_path) if args.diagnostics_output_dir_path is not None else None
        ),
        "attack_image_output_dir": str(args.attack_image_output_dir_path),
        "attack_image_manifest_csv": str(attack_image_manifest_path) if attack_image_manifest else None,
        "attack_image_manifest": list(attack_image_manifest),
        "duration_seconds": (datetime.now() - started_at).total_seconds(),
    }
    if args.audit_attack_implementation:
        report["implementation_audit"] = {
            "purpose": "Rank attack implementation audit",
            "epsilon_convention": "normalized_image_tensor",
            "epsilon_raw_pixel_equivalents": [
                rank_eval.normalized_epsilon_to_raw_pixels(float(job.condition.epsilon))
            ],
            "datasets": {job.dataset: audit_results},
            "notes": [
                "Audit mode records per-condition summaries from the attacked valid query subset.",
                "Sampled-gallery mode is not benchmark-comparable when max_dataset_samples is set.",
            ],
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    rank_eval.write_csv(output_csv, rows, args.recall_values)
    rank_eval.write_attack_image_manifest(attack_image_manifest_path, attack_image_manifest)


def run_condition_in_context(
    config: SweepConfig,
    sweep_dir: Path,
    job: SweepJob,
    context: MutableMapping[str, object],
) -> JobResult:
    from src import rank_eval

    job_dir = job_run_dir(sweep_dir, job)
    attempt_dir = job_dir / "attempts" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    output_json = job_dir / "rank_eval_results.json"
    output_csv = job_dir / "rank_eval_results.csv"
    command = command_for_job(config, job, job_dir)
    started_at = datetime.now()
    error: str | None = None
    returncode = 0

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
            try:
                args = rank_eval_args_for_job(config, job, job_dir)
                configure_rank_eval_output_dirs(args, job_dir)
                dataset_results, runtimes, query_counts, audit_results, image_manifest = (
                    rank_eval.evaluate_condition_from_context(
                        args,
                        context,
                        job.dataset,
                        job.condition.epsilon,
                    )
                )
                write_condition_rank_eval_report(
                    config,
                    job,
                    args,
                    dataset_results,
                    runtimes,
                    query_counts,
                    audit_results,
                    image_manifest,
                    output_json,
                    output_csv,
                    started_at,
                )
            except Exception as exc:  # pragma: no cover - defensive orchestration
                returncode = 1
                error = str(exc)
                print(f"In-process condition failed: {exc}", file=stderr_handle)

    finished_at = datetime.now().isoformat()
    valid_json, validation_error = find_valid_result_json(job_dir, job, config)
    if returncode == 0 and valid_json is None:
        returncode = 1
        error = validation_error
    return JobResult(
        dataset=job.dataset,
        condition_id=job.condition_id,
        command=command,
        returncode=returncode,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        run_dir=str(job_dir),
        output_json=str(valid_json) if valid_json is not None else None,
        output_csv=str(output_csv) if output_csv.exists() else None,
        started_at=started_at.isoformat(),
        finished_at=finished_at,
        error=error,
    )


def run_dataset_pass(config: SweepConfig, sweep_dir: Path, dataset: str, jobs: Sequence[SweepJob]) -> list[JobResult]:
    import commons
    from src import rank_eval

    dataset_jobs = [job for job in jobs if job.dataset == dataset]
    if not dataset_jobs:
        return []

    dataset_dir = sweep_dir / "runs" / dataset
    attempt_dir = dataset_dir / "dataset_attempts" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    started_at = datetime.now().isoformat()

    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
            with contextlib.redirect_stdout(stdout_handle), contextlib.redirect_stderr(stderr_handle):
                commons.make_deterministic(config.seed)
                base_args = rank_eval_args_for_job(config, dataset_jobs[0], job_run_dir(sweep_dir, dataset_jobs[0]))
                base_args.max_adv_negatives = max(job.condition.adv_negatives for job in dataset_jobs)
                configure_rank_eval_output_dirs(base_args, dataset_dir)
                models = rank_eval.load_models(base_args)
                context = rank_eval.prepare_dataset_context(
                    base_args,
                    dataset,
                    models,
                    max_adv_negatives=base_args.max_adv_negatives,
                )
    except Exception as exc:  # pragma: no cover - defensive orchestration
        failed_job = dataset_jobs[0]
        return [
            JobResult(
                dataset=failed_job.dataset,
                condition_id=failed_job.condition_id,
                command=command_for_job(config, failed_job, job_run_dir(sweep_dir, failed_job)),
                returncode=1,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                run_dir=str(job_run_dir(sweep_dir, failed_job)),
                output_json=None,
                output_csv=None,
                started_at=started_at,
                finished_at=datetime.now().isoformat(),
                error=str(exc),
            )
        ]

    results: list[JobResult] = []
    for job in dataset_jobs:
        result = run_condition_in_context(config, sweep_dir, job, context)
        results.append(result)
        if result.returncode != 0:
            break
    return results


def run_in_process_jobs(
    config: SweepConfig,
    sweep_dir: Path,
    jobs: Sequence[SweepJob],
    on_update=None,
) -> list[JobResult]:
    grouped: dict[str, list[SweepJob]] = {}
    for job in jobs:
        grouped.setdefault(job.dataset, []).append(job)

    results: list[JobResult] = []
    if config.parallel_runs == 1:
        for dataset in config.datasets:
            dataset_results = run_dataset_pass(config, sweep_dir, dataset, grouped.get(dataset, []))
            results.extend(dataset_results)
            if on_update is not None:
                on_update(results)
            if any(result.returncode != 0 for result in dataset_results):
                break
        return sorted(results, key=lambda item: (item.dataset, item.condition_id))

    failed = False
    with ProcessPoolExecutor(max_workers=config.parallel_runs) as executor:
        pending = {
            executor.submit(run_dataset_pass, config, sweep_dir, dataset, grouped.get(dataset, [])): dataset
            for dataset in config.datasets
            if grouped.get(dataset)
        }
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                pending.pop(future)
                dataset_results = future.result()
                results.extend(dataset_results)
                if on_update is not None:
                    on_update(results)
                if any(result.returncode != 0 for result in dataset_results):
                    failed = True
            if failed:
                break
    return sorted(results, key=lambda item: (item.dataset, item.condition_id))


def run_sweep_jobs(
    config: SweepConfig,
    sweep_dir: Path,
    jobs: Sequence[SweepJob],
    on_update=None,
) -> list[JobResult]:
    if config.execution_mode == "subprocess":
        return run_jobs(config, sweep_dir, jobs, on_update=on_update)
    return run_in_process_jobs(config, sweep_dir, jobs, on_update=on_update)


def recall_value(metrics: Mapping[str, Any], recall: int) -> float | None:
    recalls = metrics.get("recalls", {})
    value = recalls.get(f"R@{recall}")
    return None if value is None else float(value)


def rows_from_report(
    report: Mapping[str, Any],
    condition: SweepCondition,
    expected_dataset: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    results = report.get("results", {})
    if not isinstance(results, dict):
        return rows

    for dataset, model_results in results.items():
        if expected_dataset is not None and dataset != expected_dataset:
            continue
        for model, conditions in model_results.items():
            clean_metrics = conditions.get("clean_attacked_subset", {})
            attack_condition = f"{report.get('attack', {}).get('mode', 'rank_pgd_linf')}_eps_{condition.epsilon:g}"
            attacked_metrics = conditions.get(attack_condition)
            if attacked_metrics is None:
                continue

            row: dict[str, Any] = {
                "condition_id": condition.condition_id,
                "group": condition.group,
                "dataset": dataset,
                "model": model,
                "epsilon": condition.epsilon,
                "rank_steps": condition.rank_steps,
                "rank_step_size": "" if condition.rank_step_size is None else condition.rank_step_size,
                "resolved_step_size": condition.resolved_step_size,
                "step_size_mode": condition.step_size_mode,
                "step_size_multiplier": "" if condition.step_size_multiplier is None else condition.step_size_multiplier,
                "rank_restarts": condition.rank_restarts,
                "adv_negatives": condition.adv_negatives,
                "compute_budget": condition.compute_budget,
                "attacked_queries": attacked_metrics.get("attacked_queries", ""),
                "runtime_per_query_seconds": attacked_metrics.get("runtime_per_query_seconds", ""),
                "clean_correct_attack_success_rate": "",
                "all_valid_attack_success_rate": "",
                "mean_rank_displacement": "",
                "p95_rank_displacement": "",
                "mean_perturbation_norm": "",
            }
            for recall in REQUIRED_RECALLS:
                row[f"clean_R@{recall}"] = recall_value(clean_metrics, recall)
                row[f"attacked_R@{recall}"] = recall_value(attacked_metrics, recall)

            attack_success = attacked_metrics.get("attack_success", {})
            if attack_success:
                row["clean_correct_attack_success_rate"] = attack_success.get("clean_correct", {}).get("rate", "")
                row["all_valid_attack_success_rate"] = attack_success.get("all_valid", {}).get("rate", "")

            displacement = attacked_metrics.get("rank_displacement", {})
            if displacement:
                row["mean_rank_displacement"] = displacement.get("mean", "")
                row["p95_rank_displacement"] = displacement.get("p95", "")

            perturbation = attacked_metrics.get("attack_metadata", {}).get("perturbation_norm", {})
            if perturbation:
                row["mean_perturbation_norm"] = perturbation.get("mean", "")

            rows.append(row)
    return rows


def collate_results(job_results: Sequence[JobResult], jobs: Sequence[SweepJob]) -> list[dict[str, Any]]:
    job_by_key = {(job.dataset, job.condition_id): job for job in jobs}
    rows: list[dict[str, Any]] = []
    for job in job_results:
        if job.returncode != 0 or job.output_json is None:
            continue
        job_spec = job_by_key[(job.dataset, job.condition_id)]
        with Path(job.output_json).open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        rows.extend(rows_from_report(report, job_spec.condition, expected_dataset=job.dataset))
    return rows


def collate_completed_states(states: Sequence[JobState]) -> list[JobResult]:
    results = []
    for state in states:
        if state.status not in {"completed", "completed_legacy"} or state.output_json is None:
            continue
        results.append(
            JobResult(
                dataset=state.job.dataset,
                condition_id=state.job.condition_id,
                command=[],
                returncode=0,
                stdout_path="",
                stderr_path="",
                run_dir=state.run_dir,
                output_json=state.output_json,
                output_csv=None,
                started_at="",
                finished_at="",
                error=None,
            )
        )
    return results


def write_summary_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "condition_id",
        "group",
        "dataset",
        "model",
        "epsilon",
        "rank_steps",
        "rank_step_size",
        "resolved_step_size",
        "step_size_mode",
        "step_size_multiplier",
        "rank_restarts",
        "adv_negatives",
        "compute_budget",
        "attacked_queries",
        "clean_R@1",
        "clean_R@5",
        "clean_R@10",
        "clean_R@100",
        "attacked_R@1",
        "attacked_R@5",
        "attacked_R@10",
        "attacked_R@100",
        "clean_correct_attack_success_rate",
        "all_valid_attack_success_rate",
        "mean_rank_displacement",
        "p95_rank_displacement",
        "runtime_per_query_seconds",
        "mean_perturbation_norm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_default(value: Any, default: float) -> float:
    if value in ("", None):
        return default
    return float(value)


def select_strongest_setting(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("model") != "base":
            continue
        grouped.setdefault(str(row["condition_id"]), []).append(row)

    candidates = []
    for condition_id, condition_rows in grouped.items():
        attacked_r1 = [_float_or_default(row.get("attacked_R@1"), float("inf")) for row in condition_rows]
        success = [_float_or_default(row.get("clean_correct_attack_success_rate"), 0.0) for row in condition_rows]
        runtime = [_float_or_default(row.get("runtime_per_query_seconds"), float("inf")) for row in condition_rows]
        budget = [_float_or_default(row.get("compute_budget"), float("inf")) for row in condition_rows]
        candidates.append(
            {
                "condition_id": condition_id,
                "datasets": sorted(str(row.get("dataset")) for row in condition_rows),
                "mean_attacked_R@1": sum(attacked_r1) / len(attacked_r1),
                "mean_clean_correct_attack_success_rate": sum(success) / len(success),
                "mean_runtime_per_query_seconds": sum(runtime) / len(runtime),
                "compute_budget": min(budget),
                "epsilon": condition_rows[0].get("epsilon"),
                "rank_steps": condition_rows[0].get("rank_steps"),
                "resolved_step_size": condition_rows[0].get("resolved_step_size"),
                "rank_restarts": condition_rows[0].get("rank_restarts"),
                "adv_negatives": condition_rows[0].get("adv_negatives"),
                "selection_key": [
                    sum(attacked_r1) / len(attacked_r1),
                    -(sum(success) / len(success)),
                    sum(runtime) / len(runtime),
                    min(budget),
                ],
            }
        )
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: tuple(item["selection_key"]))[0]


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def job_state_counts(states: Sequence[JobState]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in states:
        counts[state.status] = counts.get(state.status, 0) + 1
    return counts


def write_manifest(
    path: Path,
    config: SweepConfig,
    conditions: Sequence[SweepCondition],
    jobs: Sequence[SweepJob],
    job_states: Sequence[JobState],
    job_results: Sequence[JobResult],
    summary_csv: Path | None,
    selected_setting: Mapping[str, Any] | None,
    dry_run: bool,
    resumed: bool = False,
    existing_manifest: Mapping[str, Any] | None = None,
) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "dry_run": dry_run,
        "resumed": resumed,
        "smoke": config.smoke,
        "execution_mode": config.execution_mode,
        "effective_parallel_dataset_workers": (
            min(config.parallel_runs, len(config.datasets)) if config.execution_mode == "in_process" else None
        ),
        "dataset_mode": "sampled_gallery" if config.max_dataset_samples is not None else "full_dataset",
        "max_dataset_samples": config.max_dataset_samples,
        "config": asdict(config),
        "portable_config": portable_config(config, conditions),
        "portable_signature": portable_signature(config, conditions),
        "condition_count": len(conditions),
        "job_count": len(jobs),
        "experiment_count_per_dataset": len(conditions),
        "datasets": list(config.datasets),
        "conditions": [asdict(condition) | {"resolved_step_size": condition.resolved_step_size} for condition in conditions],
        "planned_jobs": [
            {
                "dataset": job.dataset,
                "condition_id": job.condition_id,
                "job_id": job.job_id,
                "run_dir": str(job_run_dir(path.parent, job)),
            }
            for job in jobs
        ],
        "job_status_counts": job_state_counts(job_states),
        "job_states": [
            {
                "dataset": state.job.dataset,
                "condition_id": state.job.condition_id,
                "job_id": state.job.job_id,
                "status": state.status,
                "run_dir": state.run_dir,
                "output_json": state.output_json,
                "reason": state.reason,
            }
            for state in job_states
        ],
        "jobs": [asdict(result) for result in job_results],
        "summary_csv": str(summary_csv) if summary_csv is not None else None,
        "selected_setting": selected_setting,
        "failed_jobs": [asdict(result) for result in job_results if result.returncode != 0],
        "plotting_script": "src/visualizations.py",
        "previous_manifest_created_at": existing_manifest.get("created_at") if existing_manifest else None,
        "previous_config": existing_manifest.get("config") if existing_manifest else None,
    }
    write_json_atomic(path, manifest)


def print_dry_run(
    config: SweepConfig,
    sweep_dir: Path,
    conditions: Sequence[SweepCondition],
    jobs: Sequence[SweepJob] | None = None,
    job_states: Sequence[JobState] | None = None,
) -> None:
    jobs = list(jobs) if jobs is not None else expand_jobs(config, conditions)
    job_states = list(job_states) if job_states is not None else []
    pending_jobs = (
        pending_jobs_from_states(job_states, config.force_rerun_completed)
        if job_states
        else jobs
    )
    print(f"Sweep directory: {sweep_dir}")
    print(f"Datasets: {' '.join(config.datasets)}")
    print(f"Execution mode: {config.execution_mode}")
    print(f"Conditions per dataset: {len(conditions)}")
    print(f"Expected condition results: {len(jobs)}")
    if config.execution_mode == "in_process":
        pending_dataset_count = len(set(job.dataset for job in pending_jobs))
        print(f"Dataset passes: {pending_dataset_count}")
    else:
        print(f"src/rank_eval.py subprocess job count: {len(pending_jobs)}")
    if config.max_dataset_samples is None:
        print("Dataset mode: full_dataset")
    else:
        print(f"Dataset mode: sampled_gallery (max_dataset_samples={config.max_dataset_samples})")
    if job_states:
        counts = job_state_counts(job_states)
        print(f"Completed jobs: {counts.get('completed', 0) + counts.get('completed_legacy', 0)}")
        print(f"Pending jobs: {len(pending_jobs)}")
        print(f"Invalid/incomplete jobs: {counts.get('invalid', 0)}")
    print(f"Parallel runs: {config.parallel_runs}")
    print(f"Smoke mode: {config.smoke}")
    print("Planned pending commands:")
    for job in pending_jobs:
        job_dir = job_run_dir(sweep_dir, job)
        command = command_for_job(config, job, job_dir)
        print(" ".join(command))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    config = build_config(args)
    conditions = selected_conditions(config)
    validate_condition_dependent_config(config, conditions)
    jobs = expand_jobs(config, conditions)
    resumed = config.resume_sweep_dir is not None
    sweep_dir = (
        Path(config.resume_sweep_dir).expanduser()
        if config.resume_sweep_dir is not None
        else Path(config.output_root).expanduser() / make_sweep_id(config)
    )
    if config.resume_sweep_dir is not None and not sweep_dir.exists():
        raise FileNotFoundError(f"Resume sweep directory does not exist: {sweep_dir}")
    manifest_path = sweep_dir / "manifest.json"
    existing_manifest = load_existing_manifest(sweep_dir)
    validate_resume_manifest(existing_manifest, config, conditions)
    job_states = inspect_jobs(sweep_dir, jobs, config)
    pending_jobs = pending_jobs_from_states(job_states, config.force_rerun_completed)

    if args.dry_run:
        print_dry_run(config, sweep_dir, conditions, jobs, job_states)
        return 0

    sweep_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        manifest_path,
        config,
        conditions,
        jobs,
        job_states,
        [],
        None,
        None,
        dry_run=False,
        resumed=resumed,
        existing_manifest=existing_manifest,
    )

    def persist_progress(results_so_far: Sequence[JobResult]) -> None:
        latest_states = inspect_jobs(sweep_dir, jobs, config)
        write_manifest(
            manifest_path,
            config,
            conditions,
            jobs,
            latest_states,
            results_so_far,
            None,
            None,
            dry_run=False,
            resumed=resumed,
            existing_manifest=existing_manifest,
        )

    job_results = run_sweep_jobs(config, sweep_dir, pending_jobs, on_update=persist_progress)
    final_states = inspect_jobs(sweep_dir, jobs, config)
    completed_results = collate_completed_states(final_states)
    summary_csv = sweep_dir / "rank_pgd_strength_sweep_summary.csv"
    rows = collate_results(completed_results, jobs)
    write_summary_csv(summary_csv, rows)
    selected_setting = select_strongest_setting(rows)
    write_manifest(
        manifest_path,
        config,
        conditions,
        jobs,
        final_states,
        job_results,
        summary_csv,
        selected_setting,
        dry_run=False,
        resumed=resumed,
        existing_manifest=existing_manifest,
    )

    failures = [result for result in job_results if result.returncode != 0]
    if failures:
        print(f"{len(failures)} job(s) failed. See {manifest_path}.", file=sys.stderr)
        return 1

    print(f"Saved sweep summary to {summary_csv}")
    print(f"Saved sweep manifest to {manifest_path}")
    skipped = len([state for state in job_states if state.status in {"completed", "completed_legacy"}])
    if skipped and not config.force_rerun_completed:
        print(f"Skipped {skipped} already completed job(s).")
    if selected_setting is not None:
        print(f"Selected strongest setting: {selected_setting['condition_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
