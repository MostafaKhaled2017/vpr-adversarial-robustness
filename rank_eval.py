import copy
import csv
import json
import logging
import shlex
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

SUPERVLAD_ROOT = Path(__file__).resolve().parent / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

import parser as parser_module
from perceptual_adv_training.config import validate_cuda_runtime
from perceptual_adv_training.rank_attacks import RankAPGDLinfAttack, RankAttackConfig, RankPGDAttack
from perceptual_adv_training.retrieval_metrics import (
    attack_success_metrics,
    compute_recalls_from_features,
    nearest_positive_ranks,
    rank_displacement_summary,
)
from perceptual_adv_training.targets import RetrievalAttackBatch, build_attack_targets


SUPPORTED_TEST_METHODS = {"hard_resize", "central_crop", "single_query"}
SUPPORTED_RANK_ATTACKS = {"rank_pgd_linf", "rank_pgd_l2", "rank_apgd_linf"}
REQUIRED_RECALL_VALUES = (1, 5, 10, 100)
DEFAULT_MODEL_TAGS = ("base", "checkpoint")


def remove_parser_argument(parser, *option_strings: str) -> None:
    for action in list(parser._actions):
        if any(option_string in action.option_strings for option_string in option_strings):
            parser._actions.remove(action)
            for option_string in action.option_strings:
                parser._option_string_actions.pop(option_string, None)
            for group in parser._action_groups:
                if action in group._group_actions:
                    group._group_actions.remove(action)
            return


def build_parser():
    parser = parser_module.build_parser()
    parser.description = "Native rank attack evaluation for one or more SuperVLAD checkpoints"
    remove_parser_argument(parser, "--resume")
    remove_parser_argument(parser, "--eval_dataset_name")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset names under --eval_datasets_folder to evaluate.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="One or more checkpoint paths to evaluate. The first model is used to generate shared attacks.",
    )
    parser.add_argument(
        "--model_tags",
        type=str,
        nargs="+",
        default=None,
        help="Labels for --models. Defaults to 'base' for one model and 'base checkpoint' for two models.",
    )
    parser.add_argument(
        "--rank_attack",
        type=str,
        default="rank_pgd_linf",
        choices=sorted(SUPPORTED_RANK_ATTACKS),
        help="Retrieval-native rank attack to evaluate.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        required=True,
        help="Perturbation bounds in normalized image tensor space.",
    )
    parser.add_argument("--rank_steps", type=int, default=20, help="Attack optimization steps.")
    parser.add_argument("--rank_restarts", type=int, default=1, help="Attack restarts per query batch.")
    parser.add_argument(
        "--rank_step_size",
        type=float,
        default=None,
        help="Attack step size. Defaults to 2 * epsilon / rank_steps.",
    )
    parser.add_argument("--adv_negatives", type=int, default=5, help="Hard negatives per rank attack target.")
    parser.add_argument("--adv_margin", type=float, default=0.1, help="Margin for the retrieval rank objective.")
    parser.add_argument("--max_queries", type=int, default=None, help="Optional cap on attacked valid queries.")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional JSON report filename/path. The file is written inside a timestamped run directory.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV summary filename/path. The file is written inside the same timestamped run directory.",
    )
    return parser


def parse_arguments():
    args = build_parser().parse_args()
    args = parser_module.validate_arguments(args)
    args.model_tags = resolve_model_tags(args.models, args.model_tags)
    args.recall_values = list(dict.fromkeys([*args.recall_values, *REQUIRED_RECALL_VALUES]))
    validate_arguments(args)
    return args


def resolve_model_tags(models: Sequence[str], model_tags: Sequence[str] | None) -> list[str]:
    if model_tags is not None:
        resolved_tags = list(model_tags)
    elif len(models) <= len(DEFAULT_MODEL_TAGS):
        resolved_tags = list(DEFAULT_MODEL_TAGS[: len(models)])
    else:
        raise ValueError("--model_tags is required when evaluating more than two models.")

    if len(resolved_tags) != len(models):
        raise ValueError("--model_tags must contain exactly one tag per model.")
    if len(set(resolved_tags)) != len(resolved_tags):
        raise ValueError("--model_tags values must be unique.")
    return resolved_tags


def validate_arguments(args) -> None:
    if args.pca_dim is not None:
        raise NotImplementedError("rank_eval.py does not support PCA because attacks need differentiable descriptors.")
    if args.test_method not in SUPPORTED_TEST_METHODS:
        raise ValueError(
            f"rank_eval.py supports only {sorted(SUPPORTED_TEST_METHODS)} for --test_method, "
            f"but received {args.test_method!r}."
        )
    if any(epsilon < 0 for epsilon in args.epsilons):
        raise ValueError("--epsilons must be non-negative.")
    if args.rank_steps < 1:
        raise ValueError("--rank_steps must be at least 1.")
    if args.rank_restarts < 1:
        raise ValueError("--rank_restarts must be at least 1.")
    if args.rank_step_size is not None and args.rank_step_size <= 0:
        raise ValueError("--rank_step_size must be positive when provided.")
    if args.adv_negatives < 1:
        raise ValueError("--adv_negatives must be at least 1.")
    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError("--max_queries must be at least 1 when provided.")

    validate_cuda_runtime(args)
    for model_path in args.models:
        require_file(model_path, "--models")
    if args.foundation_model_path is not None:
        require_file(args.foundation_model_path, "--foundation_model_path")
    validate_dataset_layouts(args.eval_datasets_folder, args.datasets)


def require_file(path: str, argument_name: str) -> None:
    resolved_path = Path(path).expanduser()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"{argument_name} does not exist: {resolved_path}")


def validate_dataset_layouts(datasets_root: str, dataset_names: Sequence[str]) -> None:
    root = Path(datasets_root).expanduser()
    for dataset_name in dataset_names:
        test_root = root / dataset_name / "images" / "test"
        database_root = test_root / "database"
        queries_root = test_root / "queries"
        if not database_root.is_dir() or not queries_root.is_dir():
            raise FileNotFoundError(
                f"Dataset {dataset_name!r} must contain images/test/database and images/test/queries under {root}."
            )


def build_output_paths(args, timestamp: str | None = None) -> Tuple[Path, Path, Path]:
    timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_json is None:
        run_dir = Path("test") / "rank_eval" / timestamp
        output_json = run_dir / "rank_eval_results.json"
    else:
        requested_json = Path(args.output_json).expanduser()
        run_dir = requested_json.parent / timestamp
        output_json = run_dir / requested_json.name

    if args.output_csv is None:
        output_csv = output_json.with_suffix(".csv")
    else:
        output_csv = run_dir / Path(args.output_csv).expanduser().name
    return output_json, output_csv, run_dir


def serialize_args(args) -> Dict[str, object]:
    serialized = {}
    for key, value in vars(args).items():
        serialized[key] = list(value) if isinstance(value, tuple) else value
    return serialized


def load_model(args, checkpoint_path: str) -> Tuple[nn.Module, object]:
    import util
    from model import network

    model_args = copy.deepcopy(args)
    model_args.resume = checkpoint_path
    model = network.SuperVLADModel(
        model_args,
        pretrained_foundation=bool(model_args.foundation_model_path),
        foundation_model_path=model_args.foundation_model_path,
    )
    model = model.to(model_args.device)
    model_args.features_dim *= model_args.supervlad_clusters
    util.resume_model(model_args, model)
    model = torch.nn.DataParallel(model)
    model.eval()
    return model, model_args


def load_models(args) -> Dict[str, Tuple[nn.Module, object]]:
    models = {}
    for model_tag, model_path in zip(args.model_tags, args.models):
        logging.info("Loading %s checkpoint from %s.", model_tag, model_path)
        models[model_tag] = load_model(args, model_path)
    return models


def attack_reference_tag(args) -> str:
    return args.model_tags[0]


def query_batch_size(args) -> int:
    return 1 if args.test_method == "single_query" else args.infer_batch_size


def extract_database_features(args, eval_ds, model: nn.Module) -> np.ndarray:
    eval_ds.test_method = "hard_resize"
    database_subset = Subset(eval_ds, range(eval_ds.database_num))
    dataloader = DataLoader(
        database_subset,
        batch_size=args.infer_batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    features = np.empty((eval_ds.database_num, args.features_dim), dtype=np.float32)
    with torch.inference_mode():
        for inputs, indices in tqdm(dataloader, ncols=100, desc="Database"):
            descriptors = model(inputs.to(args.device), queryflag=0).cpu().numpy()
            features[indices.numpy(), :] = descriptors
    return features


def extract_clean_query_features(args, eval_ds, model: nn.Module) -> np.ndarray:
    eval_ds.test_method = args.test_method
    query_indices = range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)
    query_subset = Subset(eval_ds, query_indices)
    dataloader = DataLoader(
        query_subset,
        batch_size=query_batch_size(args),
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    features = np.empty((eval_ds.queries_num, args.features_dim), dtype=np.float32)
    with torch.inference_mode():
        for inputs, indices in tqdm(dataloader, ncols=100, desc="Clean queries"):
            descriptors = model(inputs.to(args.device), queryflag=0).cpu().numpy()
            local_indices = indices.numpy() - eval_ds.database_num
            features[local_indices, :] = descriptors
    return features


def extract_clean_features(args, eval_ds, model: nn.Module) -> Dict[str, np.ndarray]:
    return {
        "database": extract_database_features(args, eval_ds, model),
        "queries": extract_clean_query_features(args, eval_ds, model),
    }


def make_attack_batch(
    args,
    eval_ds,
    targets: Sequence[Mapping[str, object]],
    database_features: torch.Tensor,
    clean_query_features: np.ndarray,
) -> Tuple[torch.Tensor, RetrievalAttackBatch]:
    query_tensors = []
    query_indices = []
    positive_descriptors = []
    negative_descriptors = []

    for target in targets:
        query_index = int(target["query_index"])
        query_tensor, _ = eval_ds[eval_ds.database_num + query_index]
        query_tensors.append(query_tensor)
        query_indices.append(query_index)

        positive_descriptors.append(database_features[int(target["positive_index"])])
        negative_indexes = torch.as_tensor(target["negative_indexes"], dtype=torch.long, device=args.device)
        negative_descriptors.append(database_features[negative_indexes])

    attack_targets = RetrievalAttackBatch(
        query_indices=torch.as_tensor(query_indices, dtype=torch.long, device=args.device),
        clean_query_descriptors=torch.from_numpy(clean_query_features[query_indices]).to(args.device),
        positive_descriptors=torch.stack(positive_descriptors, dim=0),
        negative_descriptors=torch.stack(negative_descriptors, dim=0),
    )
    return torch.stack(query_tensors, dim=0).to(args.device), attack_targets


def build_rank_attack(model: nn.Module, args, epsilon: float) -> nn.Module:
    norm = "l2" if args.rank_attack == "rank_pgd_l2" else "linf"
    config = RankAttackConfig(
        epsilon=float(epsilon),
        steps=args.rank_steps,
        restarts=args.rank_restarts,
        step_size=args.rank_step_size,
        norm=norm,
        margin=args.adv_margin,
        device=args.device,
    )
    if args.rank_attack == "rank_apgd_linf":
        return RankAPGDLinfAttack(model, config)
    return RankPGDAttack(model, config)


def generate_shared_attacked_query_features(
    args,
    eval_ds,
    attack: nn.Module,
    models: Mapping[str, Tuple[nn.Module, object]],
    database_features_tensor: torch.Tensor,
    clean_query_features: np.ndarray,
    targets: Sequence[Mapping[str, object]],
    desc: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor]]:
    eval_ds.test_method = args.test_method
    attacked_features = {
        model_tag: np.empty((len(targets), model_args.features_dim), dtype=np.float32)
        for model_tag, (_, model_args) in models.items()
    }
    metadata_parts: MutableMapping[str, list[torch.Tensor]] = {}
    batch_size = query_batch_size(args)

    for offset in tqdm(range(0, len(targets), batch_size), ncols=100, desc=desc):
        batch_targets = targets[offset : offset + batch_size]
        query_inputs, attack_targets = make_attack_batch(
            args,
            eval_ds,
            batch_targets,
            database_features_tensor,
            clean_query_features,
        )
        attack_result = attack(query_inputs, attack_targets)

        with torch.inference_mode():
            for model_tag, (model, _) in models.items():
                descriptors = model(attack_result.adversarial, queryflag=0).cpu().numpy()
                attacked_features[model_tag][offset : offset + len(batch_targets), :] = descriptors

        for key, value in attack_result.metadata.items():
            metadata_parts.setdefault(key, []).append(value.cpu())

    return attacked_features, {key: torch.cat(values, dim=0) for key, values in metadata_parts.items()}


def summarize_metadata(metadata: Mapping[str, torch.Tensor]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for key, value in metadata.items():
        array = value.detach().cpu().numpy()
        if key == "restart_index":
            unique, counts = np.unique(array.astype(np.int64), return_counts=True)
            summary[key] = {str(int(index)): int(count) for index, count in zip(unique, counts)}
        else:
            summary[key] = {
                "mean": float(np.mean(array)) if array.size else 0.0,
                "max": float(np.max(array)) if array.size else 0.0,
                "min": float(np.min(array)) if array.size else 0.0,
            }
    return summary


def add_clean_results(
    results: MutableMapping[str, Dict[str, object]],
    clean_features: Mapping[str, np.ndarray],
    positives,
    valid_query_indices: np.ndarray,
    valid_positives,
    recall_values: Sequence[int],
) -> Dict[str, np.ndarray]:
    clean_ranks_by_model = {}
    for model_tag, features in clean_features.items():
        clean_all = compute_recalls_from_features(
            features["database"],
            features["queries"],
            positives,
            recall_values,
        )
        valid_clean_features = features["queries"][valid_query_indices]
        clean_subset = compute_recalls_from_features(
            features["database"],
            valid_clean_features,
            valid_positives,
            recall_values,
        )
        clean_ranks_by_model[model_tag] = nearest_positive_ranks(
            features["database"],
            valid_clean_features,
            valid_positives,
        )
        results[model_tag] = {
            "clean_all_queries": {
                **clean_all,
                "condition": "clean_all_queries",
            },
            "clean_attacked_subset": {
                **clean_subset,
                "condition": "clean_attacked_subset",
                "attacked_queries": int(len(valid_query_indices)),
            },
        }
        logging.info("%s clean all-query recalls: %s", model_tag, clean_all["recalls_str"])
        logging.info("%s clean attacked-subset recalls: %s", model_tag, clean_subset["recalls_str"])
    return clean_ranks_by_model


def evaluate_dataset(args, dataset_name: str, models: Mapping[str, Tuple[nn.Module, object]]):
    import datasets_ws

    eval_ds = datasets_ws.BaseDataset(args, args.eval_datasets_folder, dataset_name, "test")
    logging.info("Test set: %s", eval_ds)

    feature_times = {}
    clean_features = {}
    for model_tag, (model, model_args) in models.items():
        logging.info("Extracting clean descriptors for %s on %s.", model_tag, dataset_name)
        feature_start = perf_counter()
        clean_features[model_tag] = extract_clean_features(model_args, eval_ds, model)
        feature_times[model_tag] = perf_counter() - feature_start

    reference_tag = attack_reference_tag(args)
    reference_features = clean_features[reference_tag]
    positives = eval_ds.get_positives()

    target_start = perf_counter()
    targets, valid_query_indices = build_attack_targets(
        args,
        eval_ds,
        reference_features["database"],
        reference_features["queries"],
        limit_queries=args.max_queries,
    )
    target_seconds = perf_counter() - target_start
    valid_positives = [positives[index] for index in valid_query_indices]

    query_counts = {
        "total_queries": int(eval_ds.queries_num),
        "attacked_queries": int(len(valid_query_indices)),
        "skipped_queries_without_positives": int(eval_ds.queries_num - len(valid_query_indices)),
    }
    results: Dict[str, Dict[str, object]] = {}
    clean_ranks_by_model = add_clean_results(
        results,
        clean_features,
        positives,
        valid_query_indices,
        valid_positives,
        args.recall_values,
    )

    reference_model = models[reference_tag][0]
    database_features_tensor = torch.from_numpy(reference_features["database"]).to(args.device)
    attack_times: Dict[str, float] = {}
    attack_metadata_by_condition = {}
    for epsilon in args.epsilons:
        condition_name = f"{args.rank_attack}_eps_{epsilon:g}"
        attack = build_rank_attack(reference_model, args, epsilon)
        attack_start = perf_counter()
        attacked_features_by_model, attack_metadata = generate_shared_attacked_query_features(
            args,
            eval_ds,
            attack,
            models,
            database_features_tensor,
            reference_features["queries"],
            targets,
            desc=f"{dataset_name}:{condition_name}",
        )
        elapsed = perf_counter() - attack_start
        attack_times[condition_name] = elapsed
        attack_metadata_by_condition[condition_name] = summarize_metadata(attack_metadata)

        for model_tag, attacked_features in attacked_features_by_model.items():
            model_database_features = clean_features[model_tag]["database"]
            attacked_recalls = compute_recalls_from_features(
                model_database_features,
                attacked_features,
                valid_positives,
                args.recall_values,
            )
            attacked_ranks = nearest_positive_ranks(model_database_features, attacked_features, valid_positives)
            displacement = rank_displacement_summary(clean_ranks_by_model[model_tag], attacked_ranks)
            success = attack_success_metrics(clean_ranks_by_model[model_tag], attacked_ranks)

            results[model_tag][condition_name] = {
                **attacked_recalls,
                "condition": condition_name,
                "attack": args.rank_attack,
                "attack_reference_model": reference_tag,
                "epsilon": float(epsilon),
                "rank_steps": int(args.rank_steps),
                "rank_restarts": int(args.rank_restarts),
                "rank_step_size": args.rank_step_size,
                "adv_margin": float(args.adv_margin),
                "adv_negatives": int(args.adv_negatives),
                "attacked_queries": query_counts["attacked_queries"],
                "rank_displacement": displacement,
                "attack_success": success,
                "runtime_seconds": elapsed,
                "runtime_per_query_seconds": float(elapsed / max(1, len(valid_query_indices))),
                "attack_metadata": attack_metadata_by_condition[condition_name],
            }
            logging.info("%s/%s recalls: %s", model_tag, condition_name, attacked_recalls["recalls_str"])

    runtimes = {
        "feature_seconds": feature_times,
        "target_seconds": target_seconds,
        "attack_seconds": attack_times,
    }
    return results, runtimes, query_counts


def flatten_rows(results: Mapping[str, Mapping[str, Mapping[str, object]]], recall_values: Sequence[int]) -> list[Dict[str, object]]:
    rows = []
    for dataset_name, models in results.items():
        for model_tag, conditions in models.items():
            for condition_name, metrics in conditions.items():
                row = {
                    "dataset": dataset_name,
                    "model": model_tag,
                    "condition": condition_name,
                    "attack": metrics.get("attack", ""),
                    "attack_reference_model": metrics.get("attack_reference_model", ""),
                    "epsilon": metrics.get("epsilon", ""),
                    "recalls_str": metrics["recalls_str"],
                    "attacked_queries": metrics.get("attacked_queries", ""),
                    "runtime_per_query_seconds": metrics.get("runtime_per_query_seconds", ""),
                    "clean_correct_attack_success_rate": "",
                    "all_valid_attack_success_rate": "",
                    "mean_rank_displacement": "",
                }
                if "attack_success" in metrics:
                    row["clean_correct_attack_success_rate"] = metrics["attack_success"]["clean_correct"]["rate"]
                    row["all_valid_attack_success_rate"] = metrics["attack_success"]["all_valid"]["rate"]
                if "rank_displacement" in metrics:
                    row["mean_rank_displacement"] = metrics["rank_displacement"]["mean"]
                for recall_value in recall_values:
                    row[f"R@{recall_value}"] = float(metrics["recalls"][f"R@{recall_value}"])
                rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], recall_values: Sequence[int]) -> None:
    fieldnames = [
        "dataset",
        "model",
        "condition",
        "attack",
        "attack_reference_model",
        "epsilon",
        *[f"R@{value}" for value in recall_values],
        "attacked_queries",
        "runtime_per_query_seconds",
        "clean_correct_attack_success_rate",
        "all_valid_attack_success_rate",
        "mean_rank_displacement",
        "recalls_str",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()
    output_json, output_csv, run_dir = build_output_paths(args)
    args.save_dir = str(run_dir)

    import commons

    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed)

    started_at = datetime.now()
    logging.info("Arguments: %s", args)
    logging.info("The outputs are being saved in %s", run_dir)

    models = load_models(args)
    results = {}
    runtimes = {}
    query_counts = {}
    for dataset_name in args.datasets:
        logging.info("Evaluating %s.", dataset_name)
        dataset_results, dataset_runtimes, dataset_query_counts = evaluate_dataset(args, dataset_name, models)
        results[dataset_name] = dataset_results
        runtimes[dataset_name] = dataset_runtimes
        query_counts[dataset_name] = dataset_query_counts

    rows = flatten_rows(results, args.recall_values)
    report = {
        "timestamp": started_at.isoformat(),
        "command": " ".join(shlex.quote(argument) for argument in sys.argv),
        "argv": sys.argv,
        "checkpoints": dict(zip(args.model_tags, args.models)),
        "datasets": list(args.datasets),
        "arguments": serialize_args(args),
        "attack": {
            "mode": args.rank_attack,
            "epsilons": [float(epsilon) for epsilon in args.epsilons],
            "scope": "queries_only",
            "epsilon_space": "normalized_image_tensor",
            "attack_reference_model": attack_reference_tag(args),
            "shared_attacks_across_models": True,
            "target_selection": {
                "adv_negatives": int(args.adv_negatives),
                "adv_margin": float(args.adv_margin),
                "max_queries": args.max_queries,
            },
            "query_counts": query_counts,
        },
        "results": results,
        "runtime_seconds": runtimes,
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "duration_seconds": (datetime.now() - started_at).total_seconds(),
    }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    write_csv(output_csv, rows, args.recall_values)

    logging.info("Saved JSON rank evaluation report to %s", output_json)
    logging.info("Saved CSV rank evaluation summary to %s", output_csv)
    logging.info("Finished in %s", str(datetime.now() - started_at)[:-7])


if __name__ == "__main__":
    main()
