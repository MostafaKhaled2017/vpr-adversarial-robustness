import copy
import csv
import hashlib
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
from PIL import Image

SUPERVLAD_ROOT = Path(__file__).resolve().parent / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

import parser as parser_module
from src.config import denormalize_imagenet, normalized_epsilon_to_raw_pixels, validate_cuda_runtime
from src.faiss_utils import validate_faiss_runtime
from src.models import add_model_arguments, get_model_adapter, model_names
from src.rank_attacks import RankAPGDLinfAttack, RankAttackConfig, RankPGDAttack
from src.retrieval_metrics import (
    attack_success_metrics,
    compute_recalls_from_features,
    nearest_positive_ranks,
    prepare_distance_database,
    rank_displacement_summary,
    squared_l2_distance_chunk,
)
from src.targets import RetrievalAttackBatch, build_attack_targets


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
    parser.allow_abbrev = False
    parser.description = "Native rank attack evaluation for one or more VPR checkpoints"
    remove_parser_argument(parser, "--resume")
    remove_parser_argument(parser, "--eval_dataset_name")
    parser.set_defaults(resize=None)
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=model_names(),
        help="Descriptor model architecture shared by every path in --model_paths.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="One or more dataset names under --eval_datasets_folder to evaluate.",
    )
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        required=True,
        help="One or more local checkpoint paths. Attacks are generated on each model separately unless --shared_attacks is set.",
    )
    parser.add_argument(
        "--model_tags",
        type=str,
        nargs="+",
        default=None,
        help="Labels for --model_paths. Defaults to 'base' for one model and 'base checkpoint' for two models.",
    )
    parser.add_argument(
        "--shared_attacks",
        action="store_true",
        help=(
            "Generate attacks once on the first model in --model_paths and evaluate every model on the same "
            "attacked queries (transfer protocol). By default attacks are generated separately on each model."
        ),
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
        "--max_dataset_samples",
        type=int,
        default=None,
        help=(
            "Optional sampled-gallery cap. When set, at most this many query images and this many database images "
            "are selected deterministically from --seed; metrics are not full-benchmark comparable."
        ),
    )
    parser.add_argument(
        "--audit_sample_database_size",
        type=int,
        default=None,
        help=(
            "Audit-only gallery size. When set with --audit_attack_implementation, only selected query positives "
            "plus sampled non-positive database images are extracted; metrics are not benchmark-comparable."
        ),
    )
    parser.add_argument(
        "--audit_attack_implementation",
        action="store_true",
        help="Write implementation-audit diagnostics for epsilon units, clamping, descriptor norms, gradients, and loss sign.",
    )
    parser.add_argument(
        "--audit_output_json",
        type=str,
        default=None,
        help="Optional audit JSON filename/path. The file is written inside the timestamped run directory.",
    )
    parser.add_argument(
        "--trace_query_indices",
        type=int,
        nargs="+",
        default=None,
        help="Original query indices to trace at every attack step.",
    )
    parser.add_argument(
        "--trace_output_dir",
        type=str,
        default=None,
        help="Directory for per-query trace CSV files. Defaults to <run_dir>/traces.",
    )
    parser.add_argument(
        "--compute_diagnostics",
        action="store_true",
        help="Compute and write per-query diagnostic CSV files. Disabled by default.",
    )
    parser.add_argument(
        "--diagnostics_output_dir",
        type=str,
        default=None,
        help="Directory for per-query diagnostic CSV files when --compute_diagnostics is set.",
    )
    parser.add_argument(
        "--save_attack_images",
        action="store_true",
        help="Save a small set of clean, attacked, perturbation, and heatmap PNGs for report figures.",
    )
    parser.add_argument(
        "--save_attack_image_count",
        type=int,
        default=5,
        help="Maximum number of attacked queries to save image artifacts for when --save_attack_images is set.",
    )
    parser.add_argument(
        "--attack_image_output_dir",
        type=str,
        default=None,
        help="Directory for saved attack image PNGs. Defaults to <run_dir>/attack_images.",
    )
    parser.add_argument(
        "--attack_image_amplification",
        type=float,
        default=20.0,
        help="Multiplier used for amplified signed perturbation image artifacts.",
    )
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
    add_model_arguments(parser)
    return parser


def parse_arguments():
    return finalize_arguments(build_parser().parse_args())


def finalize_arguments(args):
    args = parser_module.validate_arguments(args)
    args.model_tags = resolve_model_tags(args.model_paths, args.model_tags)
    args.recall_values = list(dict.fromkeys([*args.recall_values, *REQUIRED_RECALL_VALUES]))
    adapter = get_model_adapter(args.model_type)
    if adapter.configure_evaluation is None:
        raise ValueError(f"Model {args.model_type!r} does not define rank-evaluation preprocessing.")
    adapter.configure_evaluation(args)
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
    if args.max_dataset_samples is not None:
        if args.max_dataset_samples < 1:
            raise ValueError("--max_dataset_samples must be at least 1 when provided.")
        if args.max_dataset_samples < args.adv_negatives + 1:
            raise ValueError("--max_dataset_samples must be at least --adv_negatives + 1.")
    if args.audit_output_json is not None and not args.audit_attack_implementation:
        raise ValueError("--audit_output_json requires --audit_attack_implementation.")
    if args.trace_query_indices is not None and any(index < 0 for index in args.trace_query_indices):
        raise ValueError("--trace_query_indices must contain non-negative original query indices.")
    if args.diagnostics_output_dir is not None and not args.compute_diagnostics:
        raise ValueError("--diagnostics_output_dir requires --compute_diagnostics.")
    if args.save_attack_image_count < 1:
        raise ValueError("--save_attack_image_count must be at least 1.")
    if args.attack_image_amplification <= 0:
        raise ValueError("--attack_image_amplification must be positive.")
    if args.audit_sample_database_size is not None:
        if not args.audit_attack_implementation:
            raise ValueError("--audit_sample_database_size requires --audit_attack_implementation.")
        if args.max_queries is None:
            raise ValueError("--audit_sample_database_size requires --max_queries to define the sampled query count.")
        if args.audit_sample_database_size < args.adv_negatives + 1:
            raise ValueError("--audit_sample_database_size must be at least --adv_negatives + 1.")

    validate_cuda_runtime(args)
    validate_faiss_runtime(args.device)
    for model_path in args.model_paths:
        require_file(model_path, "--model_paths")
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


def build_audit_output_path(args, run_dir: Path) -> Path | None:
    if not args.audit_attack_implementation:
        return None
    if args.audit_output_json is None:
        return run_dir / "rank_attack_audit.json"
    return run_dir / Path(args.audit_output_json).expanduser().name


def build_trace_output_dir(args, run_dir: Path) -> Path:
    if args.trace_output_dir is None:
        return run_dir / "traces"
    return Path(args.trace_output_dir).expanduser()


def build_diagnostics_output_dir(args, run_dir: Path) -> Path:
    if args.diagnostics_output_dir is None:
        return run_dir / "diagnostics"
    return Path(args.diagnostics_output_dir).expanduser()


def build_attack_image_output_dir(args, run_dir: Path) -> Path:
    if args.attack_image_output_dir is None:
        return run_dir / "attack_images"
    return Path(args.attack_image_output_dir).expanduser()


def serialize_args(args) -> Dict[str, object]:
    serialized = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        elif isinstance(value, tuple):
            serialized[key] = list(value)
        else:
            serialized[key] = value
    return serialized


def load_model(args, checkpoint_path: str) -> Tuple[nn.Module, object]:
    model_args = copy.deepcopy(args)
    model_args.resume = checkpoint_path
    adapter = get_model_adapter(model_args.model_type)
    build_model = adapter.build_evaluation or adapter.build
    model_bundle = build_model(model_args)
    model = model_bundle.model
    model = model.to(model_args.device)
    model_args.features_dim = int(model_bundle.descriptor_dim)
    load_weights = adapter.load_evaluation_weights or adapter.load_weights
    load_weights(model, checkpoint_path, model_args)
    model = torch.nn.DataParallel(model)
    model.eval()
    return model, model_args


def load_models(args) -> Dict[str, Tuple[nn.Module, object]]:
    models = {}
    for model_tag, model_path in zip(args.model_tags, args.model_paths):
        logging.info("Loading %s checkpoint from %s.", model_tag, model_path)
        models[model_tag] = load_model(args, model_path)
    return models


def build_evaluation_dataset(args, dataset_name: str):
    adapter = get_model_adapter(args.model_type)
    if adapter.build_evaluation_dataset is None:
        raise ValueError(f"Model {args.model_type!r} does not define an evaluation dataset.")
    return adapter.build_evaluation_dataset(args, dataset_name)


def attack_reference_tag(args) -> str:
    return args.model_tags[0]


def attack_groups(args, models: Mapping[str, Tuple[nn.Module, object]]):
    """Yield (reference_tag, models_to_evaluate) pairs for attack generation.

    Shared mode generates one attack set on the first model and evaluates every model
    on it. Per-model mode (the default) generates attacks on each model separately.
    """
    if args.shared_attacks:
        yield attack_reference_tag(args), dict(models)
        return
    for model_tag, model_bundle in models.items():
        yield model_tag, {model_tag: model_bundle}


def query_batch_size(args) -> int:
    return 1 if args.test_method == "single_query" else args.infer_batch_size


def extract_features_for_models(
    args,
    eval_ds,
    models: Mapping[str, Tuple[nn.Module, object]],
    dataset_indices: Sequence[int],
    desc: str,
    test_method: str,
    batch_size: int,
) -> tuple[Dict[str, np.ndarray], Dict[str, float], float]:
    eval_ds.test_method = test_method
    ordered_indices = [int(index) for index in dataset_indices]
    index_to_position = {index: position for position, index in enumerate(ordered_indices)}
    subset = Subset(eval_ds, ordered_indices)
    dataloader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )

    features = {
        model_tag: np.empty((len(ordered_indices), model_args.features_dim), dtype=np.float32)
        for model_tag, (_, model_args) in models.items()
    }
    model_seconds = {model_tag: 0.0 for model_tag in models}
    input_start = perf_counter()
    iterator = iter(dataloader)
    shared_input_seconds = perf_counter() - input_start
    with torch.inference_mode():
        with tqdm(total=len(dataloader), ncols=100, desc=desc) as progress:
            while True:
                input_start = perf_counter()
                try:
                    inputs, indices = next(iterator)
                except StopIteration:
                    break
                device_inputs = inputs.to(args.device)
                positions = [index_to_position[int(index)] for index in indices.numpy()]
                shared_input_seconds += perf_counter() - input_start

                for model_tag, (model, _) in models.items():
                    model_start = perf_counter()
                    descriptors = model(device_inputs, queryflag=0).cpu().numpy()
                    features[model_tag][positions, :] = descriptors
                    model_seconds[model_tag] += perf_counter() - model_start
                progress.update(1)
    return features, model_seconds, shared_input_seconds


def extract_database_features(args, eval_ds, model: nn.Module) -> np.ndarray:
    features, _, _ = extract_features_for_models(
        args,
        eval_ds,
        {"model": (model, args)},
        range(eval_ds.database_num),
        desc="Database",
        test_method="hard_resize",
        batch_size=args.infer_batch_size,
    )
    return features["model"]


def extract_clean_query_features(args, eval_ds, model: nn.Module) -> np.ndarray:
    query_indices = range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)
    features, _, _ = extract_features_for_models(
        args,
        eval_ds,
        {"model": (model, args)},
        query_indices,
        desc="Clean queries",
        test_method=args.test_method,
        batch_size=query_batch_size(args),
    )
    return features["model"]


def extract_indexed_features(
    args,
    eval_ds,
    model: nn.Module,
    dataset_indices: Sequence[int],
    desc: str,
    test_method: str,
) -> np.ndarray:
    features, _, _ = extract_features_for_models(
        args,
        eval_ds,
        {"model": (model, args)},
        dataset_indices,
        desc=desc,
        test_method=test_method,
        batch_size=args.infer_batch_size,
    )
    return features["model"]


def extract_clean_features(args, eval_ds, model: nn.Module) -> Dict[str, np.ndarray]:
    return {
        "database": extract_database_features(args, eval_ds, model),
        "queries": extract_clean_query_features(args, eval_ds, model),
    }


def select_valid_query_indices(positives_per_query: Sequence[Sequence[int]], limit_queries: int | None) -> np.ndarray:
    valid_query_indices = np.flatnonzero(
        np.fromiter((len(positive_candidates) > 0 for positive_candidates in positives_per_query), dtype=bool)
    )
    if len(valid_query_indices) == 0:
        raise RuntimeError("No queries with positives were found, cannot run attack evaluation.")
    if limit_queries is not None:
        valid_query_indices = valid_query_indices[:limit_queries]
    return valid_query_indices.astype(np.int64, copy=False)


def stable_sample_seed(seed: int, dataset_name: str, split_name: str) -> int:
    payload = f"{int(seed)}:{dataset_name}:{split_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False)


def attack_generation_seed(seed: int, dataset_name: str, model_tag: str, condition_name: str) -> int:
    payload = f"{int(seed)}:{dataset_name}:{model_tag}:{condition_name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="little", signed=False) % (2**63)


def reseed_attack_rng(args, dataset_name: str, model_tag: str, condition_name: str) -> None:
    """Give each per-model attack run a stable RNG state so metrics do not depend on model order.

    Shared mode keeps the legacy RNG stream untouched; seed == -1 keeps the non-deterministic convention.
    """
    if args.shared_attacks or args.seed == -1:
        return
    derived_seed = attack_generation_seed(args.seed, dataset_name, model_tag, condition_name)
    torch.manual_seed(derived_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(derived_seed)


def deterministic_subset(values: Sequence[int], requested_size: int, seed: int) -> np.ndarray:
    array = np.asarray([int(value) for value in values], dtype=np.int64)
    if requested_size >= len(array):
        return np.sort(array)
    rng = np.random.default_rng(seed)
    selected = rng.choice(array, size=int(requested_size), replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def select_sampled_query_indices(
    positives_per_query: Sequence[Sequence[int]],
    dataset_name: str,
    seed: int,
    requested_size: int,
    limit_queries: int | None,
) -> np.ndarray:
    valid_query_indices = select_valid_query_indices(positives_per_query, None)
    sample_size = min(int(requested_size), len(valid_query_indices))
    sampled = deterministic_subset(
        valid_query_indices,
        sample_size,
        stable_sample_seed(seed, dataset_name, "queries"),
    )
    if limit_queries is not None:
        sampled = sampled[:limit_queries]
    if len(sampled) == 0:
        raise RuntimeError("The sampled query set is empty.")
    return sampled.astype(np.int64, copy=False)


def select_sampled_database_indices(
    positives_per_query: Sequence[Sequence[int]],
    sampled_query_indices: np.ndarray,
    database_size: int,
    dataset_name: str,
    seed: int,
    requested_size: int,
) -> np.ndarray:
    required: list[int] = []
    required_set: set[int] = set()
    for query_index in sampled_query_indices:
        positive_candidates = sorted(int(index) for index in positives_per_query[int(query_index)])
        if not positive_candidates:
            continue
        positive_index = positive_candidates[0]
        if positive_index not in required_set:
            required.append(positive_index)
            required_set.add(positive_index)

    if len(required) > requested_size:
        raise RuntimeError(
            "The sampled database cap is too small to retain one positive for each sampled query. "
            f"Required {len(required)} positives but max_dataset_samples={requested_size}."
        )

    remaining_slots = int(requested_size) - len(required)
    candidates = [index for index in range(int(database_size)) if index not in required_set]
    filler = deterministic_subset(
        candidates,
        min(remaining_slots, len(candidates)),
        stable_sample_seed(seed, dataset_name, "database"),
    )
    selected = np.asarray([*required, *[int(index) for index in filler]], dtype=np.int64)
    if selected.size == 0:
        raise RuntimeError("The sampled database set is empty.")
    return np.sort(selected)


def filter_sampled_queries_with_targets(
    positives_per_query: Sequence[Sequence[int]],
    sampled_query_indices: np.ndarray,
    sampled_database_indices: np.ndarray,
    max_adv_negatives: int,
) -> tuple[np.ndarray, list[np.ndarray], int]:
    database_index_to_row = {int(index): row for row, index in enumerate(sampled_database_indices)}
    filtered_indices: list[int] = []
    filtered_positives: list[np.ndarray] = []
    dropped = 0
    database_rows = set(range(len(sampled_database_indices)))

    for query_index in sampled_query_indices:
        positive_rows = [
            database_index_to_row[int(positive_index)]
            for positive_index in positives_per_query[int(query_index)]
            if int(positive_index) in database_index_to_row
        ]
        positive_set = set(int(row) for row in positive_rows)
        negative_count = len(database_rows - positive_set)
        if not positive_rows or negative_count < int(max_adv_negatives):
            dropped += 1
            continue
        filtered_indices.append(int(query_index))
        filtered_positives.append(np.asarray(sorted(positive_set), dtype=np.int64))

    if not filtered_indices:
        raise RuntimeError("No sampled queries retained positives and enough negatives for attack evaluation.")
    return np.asarray(filtered_indices, dtype=np.int64), filtered_positives, dropped


def select_audit_database_indices(
    positives_per_query: Sequence[Sequence[int]],
    valid_query_indices: np.ndarray,
    database_size: int,
    requested_size: int,
) -> np.ndarray:
    selected: list[int] = []
    selected_set: set[int] = set()

    for query_index in valid_query_indices:
        for positive_index in positives_per_query[int(query_index)]:
            positive = int(positive_index)
            if positive not in selected_set:
                selected.append(positive)
                selected_set.add(positive)

    for database_index in range(database_size):
        if len(selected) >= requested_size:
            break
        if database_index not in selected_set:
            selected.append(database_index)
            selected_set.add(database_index)

    if len(selected) == 0:
        raise RuntimeError("The audit database sample is empty.")
    return np.asarray(selected, dtype=np.int64)


def map_sampled_positives(
    positives_per_query: Sequence[Sequence[int]],
    valid_query_indices: np.ndarray,
    sampled_database_indices: np.ndarray,
) -> list[np.ndarray]:
    database_index_to_row = {int(index): row for row, index in enumerate(sampled_database_indices)}
    mapped_positives = []
    for query_index in valid_query_indices:
        positive_rows = [
            database_index_to_row[int(positive_index)]
            for positive_index in positives_per_query[int(query_index)]
            if int(positive_index) in database_index_to_row
        ]
        if not positive_rows:
            raise RuntimeError(f"Audit sample omitted positives for query index {int(query_index)}.")
        mapped_positives.append(np.asarray(positive_rows, dtype=np.int64))
    return mapped_positives


def build_sampled_attack_targets(
    args,
    sampled_database_indices: np.ndarray,
    sampled_database_features: np.ndarray,
    sampled_query_features: np.ndarray,
    valid_query_indices: np.ndarray,
    sampled_positives: Sequence[np.ndarray],
) -> list[Dict[str, object]]:
    targets = []
    all_database_rows = np.arange(len(sampled_database_indices), dtype=np.int64)

    for query_row, query_index in enumerate(valid_query_indices):
        query_feature = sampled_query_features[query_row]
        positive_candidates = np.asarray(sampled_positives[query_row], dtype=np.int64)
        positive_distances = np.sum(
            (sampled_database_features[positive_candidates] - query_feature[None, :]) ** 2,
            axis=1,
        )
        positive_index = int(positive_candidates[np.argmin(positive_distances)])

        positive_mask = np.zeros(len(sampled_database_indices), dtype=bool)
        positive_mask[positive_candidates] = True
        negative_candidates = all_database_rows[~positive_mask]
        if len(negative_candidates) < args.adv_negatives:
            raise RuntimeError(
                "Audit sample does not contain enough non-positive database images for "
                f"--adv_negatives={args.adv_negatives}."
            )
        negative_distances = np.sum(
            (sampled_database_features[negative_candidates] - query_feature[None, :]) ** 2,
            axis=1,
        )
        negative_order = np.argsort(negative_distances)

        targets.append(
            {
                "query_index": int(query_index),
                "query_feature_index": int(query_row),
                "positive_index": positive_index,
                "negative_indexes": negative_candidates[negative_order[: args.adv_negatives]].astype(np.int64),
            }
        )
    return targets


def make_attack_batch(
    args,
    eval_ds,
    targets: Sequence[Mapping[str, object]],
    database_features: torch.Tensor,
    clean_query_features: np.ndarray,
) -> Tuple[torch.Tensor, RetrievalAttackBatch]:
    query_tensors = []
    query_indices = []
    query_feature_indices = []
    positive_descriptors = []
    negative_descriptors = []

    for target in targets:
        query_index = int(target["query_index"])
        query_feature_index = int(target.get("query_feature_index", query_index))
        query_tensor, _ = eval_ds[eval_ds.database_num + query_index]
        query_tensors.append(query_tensor)
        query_indices.append(query_index)
        query_feature_indices.append(query_feature_index)

        positive_descriptors.append(database_features[int(target["positive_index"])])
        negative_indexes = torch.as_tensor(target["negative_indexes"], dtype=torch.long, device=args.device)
        negative_descriptors.append(database_features[negative_indexes])

    attack_targets = RetrievalAttackBatch(
        query_indices=torch.as_tensor(query_indices, dtype=torch.long, device=args.device),
        clean_query_descriptors=torch.from_numpy(clean_query_features[query_feature_indices]).to(args.device),
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
        audit=bool(args.audit_attack_implementation),
        trace_query_indices=args.trace_query_indices,
    )
    if args.rank_attack == "rank_apgd_linf":
        return RankAPGDLinfAttack(model, config)
    return RankPGDAttack(model, config)


def clear_cuda_cache(args) -> None:
    if getattr(args, "device", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarize_descriptor_norms(descriptors: np.ndarray) -> Dict[str, float]:
    norms = np.linalg.norm(descriptors.astype(np.float32, copy=False), axis=1)
    if norms.size == 0:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": int(norms.size),
        "mean": float(np.mean(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
    }


def summarize_metadata_tensor(metadata: Mapping[str, torch.Tensor], key: str) -> Dict[str, float]:
    value = metadata.get(key)
    if value is None:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    array = value.detach().cpu().numpy()
    if array.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def build_condition_audit(
    args,
    epsilon: float,
    clean_features: Mapping[str, Mapping[str, np.ndarray]],
    attacked_features_by_model: Mapping[str, np.ndarray],
    valid_query_indices: np.ndarray,
    attack_metadata: Mapping[str, torch.Tensor],
    clean_query_feature_indices: np.ndarray | None = None,
) -> Dict[str, object]:
    initial_loss = attack_metadata["initial_loss"].detach().cpu().numpy()
    best_loss = attack_metadata["best_loss"].detach().cpu().numpy()
    positive_before = attack_metadata["positive_distance_before"].detach().cpu().numpy()
    positive_after = attack_metadata["positive_distance_after"].detach().cpu().numpy()
    negative_before = attack_metadata["hard_negative_distance_before"].detach().cpu().numpy()
    negative_after = attack_metadata["hard_negative_distance_after"].detach().cpu().numpy()
    score_before = positive_before - negative_before
    score_after = positive_after - negative_after
    best_minus_initial = best_loss - initial_loss

    query_feature_indices = clean_query_feature_indices if clean_query_feature_indices is not None else valid_query_indices
    clean_descriptor_norms = {
        model_tag: summarize_descriptor_norms(features["queries"][query_feature_indices])
        for model_tag, features in clean_features.items()
    }
    attacked_descriptor_norms = {
        model_tag: summarize_descriptor_norms(attacked_features)
        for model_tag, attacked_features in attacked_features_by_model.items()
    }

    return {
        "epsilon": normalized_epsilon_to_raw_pixels(float(epsilon)),
        "attack": args.rank_attack,
        "rank_steps": int(args.rank_steps),
        "rank_restarts": int(args.rank_restarts),
        "rank_step_size": args.rank_step_size,
        "descriptor_norms": {
            "clean_attacked_subset": clean_descriptor_norms,
            "attacked": attacked_descriptor_norms,
        },
        "gradient_flow": {
            "gradient_norm": summarize_metadata_tensor(attack_metadata, "gradient_norm"),
            "gradient_norm_max": summarize_metadata_tensor(attack_metadata, "gradient_norm_max"),
            "has_nonzero_gradients": bool(np.max(attack_metadata["gradient_norm"].detach().cpu().numpy()) > 0.0),
        },
        "clamping": {
            "denormalized_min": summarize_metadata_tensor(attack_metadata, "denormalized_min"),
            "denormalized_max": summarize_metadata_tensor(attack_metadata, "denormalized_max"),
            "within_raw_0_1_bounds": bool(
                torch.all(attack_metadata["denormalized_min"] >= -1e-6).item()
                and torch.all(attack_metadata["denormalized_max"] <= 1.0 + 1e-6).item()
            ),
        },
        "loss_sign": {
            "mean_positive_distance_delta": float(np.mean(positive_after - positive_before)),
            "mean_hard_negative_distance_delta": float(np.mean(negative_after - negative_before)),
            "mean_rank_score_delta": float(np.mean(score_after - score_before)),
            "mean_best_loss_delta": float(np.mean(best_minus_initial)),
        },
        "best_selection": {
            "all_best_loss_at_least_initial_loss": bool(np.all(best_minus_initial >= -1e-6)),
            "best_minus_initial_loss": {
                "mean": float(np.mean(best_minus_initial)),
                "min": float(np.min(best_minus_initial)),
                "max": float(np.max(best_minus_initial)),
            },
        },
    }


def _epsilon_label(epsilon: float) -> str:
    return f"{float(epsilon):g}"


def _filename_token(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in value)


def attack_image_output_paths(
    output_dir: Path,
    dataset_name: str,
    attack_name: str,
    epsilon: float,
    query_index: int,
    amplification: float,
) -> Dict[str, Path]:
    image_dir = output_dir / _filename_token(dataset_name) / _filename_token(attack_name) / f"eps_{_epsilon_label(epsilon)}"
    prefix = image_dir / str(int(query_index))
    amplification_label = f"x{_epsilon_label(amplification)}"
    return {
        "clean": prefix.with_name(f"{prefix.name}_clean.png"),
        "attacked": prefix.with_name(f"{prefix.name}_attacked.png"),
        "perturbation": prefix.with_name(f"{prefix.name}_perturbation_{amplification_label}.png"),
        "abs_heatmap": prefix.with_name(f"{prefix.name}_abs_heatmap.png"),
    }


def select_attack_image_query_indices(
    targets: Sequence[Mapping[str, object]],
    trace_query_indices: Sequence[int] | None,
    image_count: int,
) -> set[int]:
    target_order = [int(target["query_index"]) for target in targets]
    selected: list[int] = []

    if trace_query_indices is not None:
        available = set(target_order)
        for query_index in trace_query_indices:
            query_index = int(query_index)
            if query_index in available and query_index not in selected:
                selected.append(query_index)
            if len(selected) >= image_count:
                return set(selected)

    for query_index in target_order:
        if query_index not in selected:
            selected.append(query_index)
        if len(selected) >= image_count:
            break
    return set(selected)


def _tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return np.rint(array * 255.0).astype(np.uint8)


def _save_rgb_tensor_png(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_tensor_image_to_uint8(image), mode="RGB").save(path)


def save_attack_image_artifacts(
    output_dir: Path,
    dataset_name: str,
    attack_name: str,
    epsilon: float,
    query_index: int,
    clean_input: torch.Tensor,
    adversarial_input: torch.Tensor,
    amplification: float,
) -> Dict[str, str]:
    clean_raw = denormalize_imagenet(clean_input.detach().unsqueeze(0)).squeeze(0).cpu().clamp(0.0, 1.0)
    attacked_raw = denormalize_imagenet(adversarial_input.detach().unsqueeze(0)).squeeze(0).cpu().clamp(0.0, 1.0)
    delta = attacked_raw - clean_raw
    amplified = (0.5 + delta * float(amplification)).clamp(0.0, 1.0)

    abs_delta = delta.abs().max(dim=0).values
    max_delta = float(abs_delta.max().item())
    normalized_heat = abs_delta / max(max_delta, 1e-12)
    heatmap = torch.stack(
        [
            normalized_heat,
            normalized_heat * 0.6,
            torch.zeros_like(normalized_heat),
        ],
        dim=0,
    )

    paths = attack_image_output_paths(
        output_dir,
        dataset_name,
        attack_name,
        epsilon,
        query_index,
        amplification,
    )
    _save_rgb_tensor_png(clean_raw, paths["clean"])
    _save_rgb_tensor_png(attacked_raw, paths["attacked"])
    _save_rgb_tensor_png(amplified, paths["perturbation"])
    _save_rgb_tensor_png(heatmap, paths["abs_heatmap"])
    return {key: str(path) for key, path in paths.items()}


def _distance_to_rank(database_features: np.ndarray, query_feature: np.ndarray, positive_indexes: Sequence[int]) -> int:
    positives_array = np.asarray(positive_indexes, dtype=np.int64)
    if positives_array.size == 0:
        return -1
    distances = np.linalg.norm(database_features - query_feature[None, :], axis=1)
    order = np.argsort(distances)
    inverse_ranks = np.empty_like(order)
    inverse_ranks[order] = np.arange(1, len(order) + 1, dtype=np.int64)
    return int(inverse_ranks[positives_array].min())


def write_trace_csvs(
    trace_output_dir: Path,
    dataset_name: str,
    attack_name: str,
    epsilon: float,
    trace_rows: Sequence[Mapping[str, object]],
    database_features: np.ndarray,
    positives_by_query: Mapping[int, Sequence[int]],
) -> list[str]:
    if not trace_rows:
        return []

    fieldnames = [
        "query_index",
        "epsilon",
        "restart",
        "step",
        "loss",
        "positive_distance",
        "hard_negative_distance",
        "nearest_positive_rank",
        "perturbation_linf_normalized",
        "perturbation_linf_raw",
        "best_so_far",
    ]
    by_query: dict[int, list[Mapping[str, object]]] = {}
    for row in trace_rows:
        by_query.setdefault(int(row["query_index"]), []).append(row)

    written_paths = []
    output_dir = trace_output_dir / dataset_name / attack_name
    output_dir.mkdir(parents=True, exist_ok=True)
    for query_index, rows in by_query.items():
        path = output_dir / f"{query_index}_eps_{_epsilon_label(epsilon)}.csv"
        positives = positives_by_query.get(query_index, [])
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in sorted(rows, key=lambda item: (int(item["restart"]), int(item["step"]))):
                descriptor = row["descriptor"]
                if isinstance(descriptor, torch.Tensor):
                    descriptor_array = descriptor.detach().cpu().numpy()
                else:
                    descriptor_array = np.asarray(descriptor, dtype=np.float32)
                writer.writerow(
                    {
                        "query_index": query_index,
                        "epsilon": float(epsilon),
                        "restart": int(row["restart"]),
                        "step": int(row["step"]),
                        "loss": float(row["loss"]),
                        "positive_distance": float(row["positive_distance"]),
                        "hard_negative_distance": float(row["hard_negative_distance"]),
                        "nearest_positive_rank": _distance_to_rank(database_features, descriptor_array, positives),
                        "perturbation_linf_normalized": float(row["perturbation_linf_normalized"]),
                        "perturbation_linf_raw": float(row["perturbation_linf_raw"]),
                        "best_so_far": bool(row["best_so_far"]),
                    }
                )
        written_paths.append(str(path))
    return written_paths


def compute_query_diagnostic_rows(
    database_features: np.ndarray,
    clean_query_features: np.ndarray,
    attacked_query_features: np.ndarray,
    positives_per_query: Sequence[Sequence[int]],
    valid_query_indices: np.ndarray,
    targets: Sequence[Mapping[str, object]],
    perturbation_norms: np.ndarray,
    clean_query_feature_indices: np.ndarray | None = None,
    clean_ranks: np.ndarray | None = None,
    attacked_ranks: np.ndarray | None = None,
    chunk_size: int = 128,
) -> list[Dict[str, object]]:
    rows = []
    query_feature_indices = clean_query_feature_indices if clean_query_feature_indices is not None else valid_query_indices
    selected_clean_queries = clean_query_features[np.asarray(query_feature_indices, dtype=np.int64)]
    if clean_ranks is None:
        clean_ranks = nearest_positive_ranks(database_features, selected_clean_queries, positives_per_query)
    if attacked_ranks is None:
        attacked_ranks = nearest_positive_ranks(database_features, attacked_query_features, positives_per_query)

    database, database_norms = prepare_distance_database(database_features)
    for start in range(0, len(valid_query_indices), chunk_size):
        end = min(start + chunk_size, len(valid_query_indices))
        clean_batch = selected_clean_queries[start:end]
        attacked_batch = attacked_query_features[start:end]
        clean_distance_rows = squared_l2_distance_chunk(database, database_norms, clean_batch)
        attacked_distance_rows = squared_l2_distance_chunk(database, database_norms, attacked_batch)
        np.sqrt(clean_distance_rows, out=clean_distance_rows)
        np.sqrt(attacked_distance_rows, out=attacked_distance_rows)

        for local_index, row_index in enumerate(range(start, end)):
            query_index = int(valid_query_indices[row_index])
            target = targets[row_index]
            positives = np.asarray(positives_per_query[row_index], dtype=np.int64)
            positive_index = int(target["positive_index"])
            negative_indexes = np.asarray(target["negative_indexes"], dtype=np.int64)
            clean_query = clean_batch[local_index]
            attacked_query = attacked_batch[local_index]
            clean_distances = clean_distance_rows[local_index]
            attacked_distances = attacked_distance_rows[local_index]

            positive_mask = np.zeros(database_features.shape[0], dtype=bool)
            positive_mask[positives] = True
            negative_candidates = np.flatnonzero(~positive_mask)

            clean_positive_distance = float(clean_distances[positive_index])
            attacked_positive_distance = float(attacked_distances[positive_index])
            clean_nearest_negative_distance = float(np.min(clean_distances[negative_candidates]))
            attacked_nearest_negative_distance = float(np.min(attacked_distances[negative_candidates]))
            clean_rank = int(clean_ranks[row_index])
            attacked_rank = int(attacked_ranks[row_index])
            rho_q = float(np.linalg.norm(attacked_query - clean_query, ord=2))
            cwr_positive_upper = clean_positive_distance + rho_q
            cwr_negative_lower = np.maximum(clean_distances[negative_candidates] - rho_q, 0.0)

            rows.append(
                {
                    "query_index": query_index,
                    "clean_nearest_positive_rank": clean_rank,
                    "attacked_nearest_positive_rank": attacked_rank,
                    "attack_success": bool(clean_rank == 1 and attacked_rank > 1),
                    "clean_positive_distance": clean_positive_distance,
                    "clean_nearest_negative_distance": clean_nearest_negative_distance,
                    "clean_margin": clean_nearest_negative_distance - clean_positive_distance,
                    "attacked_positive_distance": attacked_positive_distance,
                    "attacked_hard_negative_distance": attacked_nearest_negative_distance,
                    "attacked_nearest_negative_distance": attacked_nearest_negative_distance,
                    "attacked_margin": attacked_nearest_negative_distance - attacked_positive_distance,
                    "rank_displacement": int(attacked_rank - clean_rank) if clean_rank > 0 and attacked_rank > 0 else "",
                    "perturbation_norm": float(perturbation_norms[row_index]),
                    "rho_q": rho_q,
                    "cwr_estimate": int(1 + np.count_nonzero(cwr_negative_lower <= cwr_positive_upper)),
                    "cwr_note": "descriptor-space sensitivity estimate, not a formal certificate",
                    "selected_positive_index": positive_index,
                    "selected_hard_negative_indexes": " ".join(str(int(index)) for index in negative_indexes),
                }
            )
    return rows


def write_diagnostics_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "query_index",
        "clean_nearest_positive_rank",
        "attacked_nearest_positive_rank",
        "attack_success",
        "clean_positive_distance",
        "clean_nearest_negative_distance",
        "clean_margin",
        "attacked_positive_distance",
        "attacked_hard_negative_distance",
        "attacked_nearest_negative_distance",
        "attacked_margin",
        "rank_displacement",
        "perturbation_norm",
        "rho_q",
        "cwr_estimate",
        "cwr_note",
        "selected_positive_index",
        "selected_hard_negative_indexes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_shared_attacked_query_features(
    args,
    eval_ds,
    attack: nn.Module,
    models: Mapping[str, Tuple[nn.Module, object]],
    database_features_tensor: torch.Tensor,
    clean_query_features: np.ndarray,
    targets: Sequence[Mapping[str, object]],
    dataset_name: str,
    epsilon: float,
    desc: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, torch.Tensor], list[Dict[str, object]], list[Dict[str, object]]]:
    eval_ds.test_method = args.test_method
    attacked_features = {
        model_tag: np.empty((len(targets), model_args.features_dim), dtype=np.float32)
        for model_tag, (_, model_args) in models.items()
    }
    metadata_parts: MutableMapping[str, list[torch.Tensor]] = {}
    trace_rows: list[Dict[str, object]] = []
    image_rows: list[Dict[str, object]] = []
    batch_size = query_batch_size(args)
    image_query_indices = (
        select_attack_image_query_indices(targets, args.trace_query_indices, args.save_attack_image_count)
        if args.save_attack_images
        else set()
    )

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
        if attack_result.traces:
            trace_rows.extend(attack_result.traces)
        if image_query_indices:
            for batch_index, target in enumerate(batch_targets):
                query_index = int(target["query_index"])
                if query_index not in image_query_indices:
                    continue
                paths = save_attack_image_artifacts(
                    args.attack_image_output_dir_path,
                    dataset_name,
                    args.rank_attack,
                    epsilon,
                    query_index,
                    query_inputs[batch_index],
                    attack_result.adversarial[batch_index],
                    args.attack_image_amplification,
                )
                image_rows.append(
                    {
                        "dataset": dataset_name,
                        "attack": args.rank_attack,
                        "epsilon": float(epsilon),
                        "query_index": query_index,
                        **paths,
                    }
                )

    return (
        attacked_features,
        {key: torch.cat(values, dim=0) for key, values in metadata_parts.items()},
        trace_rows,
        image_rows,
    )


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


def add_sampled_clean_results(
    results: MutableMapping[str, Dict[str, object]],
    clean_features: Mapping[str, np.ndarray],
    sampled_positives,
    recall_values: Sequence[int],
) -> Dict[str, np.ndarray]:
    clean_ranks_by_model = {}
    for model_tag, features in clean_features.items():
        clean_sample = compute_recalls_from_features(
            features["database"],
            features["queries"],
            sampled_positives,
            recall_values,
        )
        clean_ranks_by_model[model_tag] = nearest_positive_ranks(
            features["database"],
            features["queries"],
            sampled_positives,
        )
        results[model_tag] = {
            "clean_audit_sample": {
                **clean_sample,
                "condition": "clean_audit_sample",
                "attacked_queries": int(len(sampled_positives)),
                "audit_sample": True,
            },
        }
        logging.info("%s clean audit-sample recalls: %s", model_tag, clean_sample["recalls_str"])
    return clean_ranks_by_model


def add_sampled_gallery_clean_results(
    results: MutableMapping[str, Dict[str, object]],
    clean_features: Mapping[str, np.ndarray],
    sampled_positives,
    recall_values: Sequence[int],
) -> Dict[str, np.ndarray]:
    clean_ranks_by_model = {}
    for model_tag, features in clean_features.items():
        clean_sample = compute_recalls_from_features(
            features["database"],
            features["queries"],
            sampled_positives,
            recall_values,
        )
        clean_ranks_by_model[model_tag] = nearest_positive_ranks(
            features["database"],
            features["queries"],
            sampled_positives,
        )
        clean_block = {
            **clean_sample,
            "condition": "clean_attacked_subset",
            "attacked_queries": int(len(sampled_positives)),
            "sampled_gallery": True,
            "benchmark_comparable": False,
        }
        results[model_tag] = {
            "clean_all_queries": clean_block,
            "clean_attacked_subset": clean_block,
        }
        logging.info("%s clean sampled-gallery recalls: %s", model_tag, clean_sample["recalls_str"])
    return clean_ranks_by_model


def evaluate_audit_sample_dataset(args, dataset_name: str, models: Mapping[str, Tuple[nn.Module, object]]):
    eval_ds = build_evaluation_dataset(args, dataset_name)
    logging.info("Test set: %s", eval_ds)
    logging.info(
        "Using audit sample mode with max_queries=%s and audit_sample_database_size=%s. "
        "Results are not benchmark-comparable.",
        args.max_queries,
        args.audit_sample_database_size,
    )

    positives = eval_ds.get_positives()
    valid_query_indices = select_valid_query_indices(positives, args.max_queries)
    sampled_database_indices = select_audit_database_indices(
        positives,
        valid_query_indices,
        eval_ds.database_num,
        args.audit_sample_database_size,
    )
    sampled_query_dataset_indices = [eval_ds.database_num + int(query_index) for query_index in valid_query_indices]
    sampled_positives = map_sampled_positives(positives, valid_query_indices, sampled_database_indices)

    logging.info(
        "Extracting shared audit-sample descriptors on %s (%d database, %d queries, %d models).",
        dataset_name,
        len(sampled_database_indices),
        len(valid_query_indices),
        len(models),
    )
    database_features, database_times, database_input_seconds = extract_features_for_models(
        args,
        eval_ds,
        models,
        sampled_database_indices.tolist(),
        desc=f"{dataset_name}:audit database",
        test_method="hard_resize",
        batch_size=args.infer_batch_size,
    )
    query_features, query_times, query_input_seconds = extract_features_for_models(
        args,
        eval_ds,
        models,
        sampled_query_dataset_indices,
        desc=f"{dataset_name}:audit queries",
        test_method=args.test_method,
        batch_size=query_batch_size(args),
    )
    clean_features = {
        model_tag: {"database": database_features[model_tag], "queries": query_features[model_tag]}
        for model_tag in models
    }
    feature_times = {
        model_tag: database_times[model_tag] + query_times[model_tag]
        for model_tag in models
    }
    feature_shared_input_seconds = database_input_seconds + query_input_seconds

    reference_tag = attack_reference_tag(args)
    reference_features = clean_features[reference_tag]

    target_start = perf_counter()
    targets = build_sampled_attack_targets(
        args,
        sampled_database_indices,
        reference_features["database"],
        reference_features["queries"],
        valid_query_indices,
        sampled_positives,
    )
    target_seconds = perf_counter() - target_start

    query_counts = {
        "total_queries": int(eval_ds.queries_num),
        "attacked_queries": int(len(valid_query_indices)),
        "skipped_queries_without_positives": int(eval_ds.queries_num - len(select_valid_query_indices(positives, None))),
        "audit_sample": True,
        "sampled_database_images": int(len(sampled_database_indices)),
        "sampled_query_indices": [int(index) for index in valid_query_indices],
        "sampled_database_indices": [int(index) for index in sampled_database_indices],
    }
    results: Dict[str, Dict[str, object]] = {}
    clean_ranks_by_model = add_sampled_clean_results(
        results,
        clean_features,
        sampled_positives,
        args.recall_values,
    )

    reference_model = models[reference_tag][0]
    database_features_tensor = torch.from_numpy(reference_features["database"]).to(args.device)
    attack_times: Dict[str, float] = {}
    attack_metadata_by_condition = {}
    audit_by_condition: Dict[str, object] = {}
    image_manifest: list[Dict[str, object]] = []
    query_feature_indices = np.arange(len(valid_query_indices), dtype=np.int64)
    for epsilon in args.epsilons:
        condition_name = f"{args.rank_attack}_eps_{epsilon:g}"
        attack = build_rank_attack(reference_model, args, epsilon)
        attack_start = perf_counter()
        attacked_features_by_model, attack_metadata, trace_rows, image_rows = generate_shared_attacked_query_features(
            args,
            eval_ds,
            attack,
            models,
            database_features_tensor,
            reference_features["queries"],
            targets,
            dataset_name,
            epsilon,
            desc=f"{dataset_name}:{condition_name}:audit sample",
        )
        image_manifest.extend(image_rows)
        elapsed = perf_counter() - attack_start
        attack_times[condition_name] = elapsed
        attack_metadata_by_condition[condition_name] = summarize_metadata(attack_metadata)
        write_trace_csvs(
            args.trace_output_dir_path,
            dataset_name,
            args.rank_attack,
            epsilon,
            trace_rows,
            reference_features["database"],
            {int(index): sampled_positives[row] for row, index in enumerate(valid_query_indices)},
        )
        audit_by_condition[condition_name] = build_condition_audit(
            args,
            epsilon,
            clean_features,
            attacked_features_by_model,
            valid_query_indices,
            attack_metadata,
            clean_query_feature_indices=query_feature_indices,
        )

        for model_tag, attacked_features in attacked_features_by_model.items():
            model_database_features = clean_features[model_tag]["database"]
            attacked_ranks = nearest_positive_ranks(model_database_features, attacked_features, sampled_positives)
            if args.compute_diagnostics:
                diagnostic_rows = compute_query_diagnostic_rows(
                    model_database_features,
                    clean_features[model_tag]["queries"],
                    attacked_features,
                    sampled_positives,
                    valid_query_indices,
                    targets,
                    attack_metadata["perturbation_norm"].detach().cpu().numpy(),
                    clean_query_feature_indices=query_feature_indices,
                    clean_ranks=clean_ranks_by_model[model_tag],
                    attacked_ranks=attacked_ranks,
                )
                diagnostics_path = (
                    args.diagnostics_output_dir_path
                    / (
                        f"{_filename_token(dataset_name)}_{_filename_token(model_tag)}_"
                        f"{_filename_token(args.rank_attack)}_eps_{_epsilon_label(epsilon)}.csv"
                    )
                )
                write_diagnostics_csv(diagnostics_path, diagnostic_rows)
            attacked_recalls = compute_recalls_from_features(
                model_database_features,
                attacked_features,
                sampled_positives,
                args.recall_values,
            )
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
                "audit_sample": True,
                "sampled_database_images": query_counts["sampled_database_images"],
                "rank_displacement": displacement,
                "attack_success": success,
                "runtime_seconds": elapsed,
                "runtime_per_query_seconds": float(elapsed / max(1, len(valid_query_indices))),
                "attack_metadata": attack_metadata_by_condition[condition_name],
            }
            logging.info("%s/%s audit-sample recalls: %s", model_tag, condition_name, attacked_recalls["recalls_str"])

    runtimes = {
        "feature_seconds": feature_times,
        "feature_shared_input_seconds": feature_shared_input_seconds,
        "target_seconds": target_seconds,
        "attack_seconds": attack_times,
    }
    return results, runtimes, query_counts, audit_by_condition, image_manifest


def prepare_dataset_context(
    args,
    dataset_name: str,
    models: Mapping[str, Tuple[nn.Module, object]],
    max_adv_negatives: int | None = None,
) -> Dict[str, object]:
    eval_ds = build_evaluation_dataset(args, dataset_name)
    logging.info("Test set: %s", eval_ds)
    max_adv_negatives = int(max_adv_negatives if max_adv_negatives is not None else args.adv_negatives)
    positives = eval_ds.get_positives()

    if args.max_dataset_samples is None:
        logging.info("Extracting shared clean descriptors for %d models on %s.", len(models), dataset_name)
        database_features, database_times, database_input_seconds = extract_features_for_models(
            args,
            eval_ds,
            models,
            range(eval_ds.database_num),
            desc=f"{dataset_name}:database",
            test_method="hard_resize",
            batch_size=args.infer_batch_size,
        )
        query_dataset_indices = range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)
        query_features, query_times, query_input_seconds = extract_features_for_models(
            args,
            eval_ds,
            models,
            query_dataset_indices,
            desc=f"{dataset_name}:clean queries",
            test_method=args.test_method,
            batch_size=query_batch_size(args),
        )
        clean_features = {
            model_tag: {"database": database_features[model_tag], "queries": query_features[model_tag]}
            for model_tag in models
        }
        feature_times = {
            model_tag: database_times[model_tag] + query_times[model_tag]
            for model_tag in models
        }
        feature_shared_input_seconds = database_input_seconds + query_input_seconds

        valid_query_indices = select_valid_query_indices(positives, args.max_queries)
        valid_positives = [positives[index] for index in valid_query_indices]
        query_counts = {
            "total_queries": int(eval_ds.queries_num),
            "attacked_queries": int(len(valid_query_indices)),
            "skipped_queries_without_positives": int(eval_ds.queries_num - len(valid_query_indices)),
            "sampled_gallery": False,
            "benchmark_comparable": True,
            "max_dataset_samples": None,
        }
        clean_results: Dict[str, Dict[str, object]] = {}
        clean_ranks_by_model = add_clean_results(
            clean_results,
            clean_features,
            positives,
            valid_query_indices,
            valid_positives,
            args.recall_values,
        )
        return {
            "dataset_name": dataset_name,
            "eval_ds": eval_ds,
            "models": models,
            "clean_features": clean_features,
            "feature_times": feature_times,
            "feature_shared_input_seconds": feature_shared_input_seconds,
            "positives": positives,
            "valid_query_indices": valid_query_indices,
            "valid_positives": valid_positives,
            "clean_results": clean_results,
            "clean_ranks_by_model": clean_ranks_by_model,
            "query_counts": query_counts,
            "target_cache": {},
            "sampled_gallery": False,
        }

    logging.info(
        "Using sampled-gallery mode on %s with max_dataset_samples=%s. Results are not benchmark-comparable.",
        dataset_name,
        args.max_dataset_samples,
    )
    sampled_query_indices = select_sampled_query_indices(
        positives,
        dataset_name,
        args.seed,
        args.max_dataset_samples,
        args.max_queries,
    )
    sampled_database_indices = select_sampled_database_indices(
        positives,
        sampled_query_indices,
        eval_ds.database_num,
        dataset_name,
        args.seed,
        args.max_dataset_samples,
    )
    valid_query_indices, sampled_positives, dropped_queries = filter_sampled_queries_with_targets(
        positives,
        sampled_query_indices,
        sampled_database_indices,
        max_adv_negatives,
    )
    sampled_query_dataset_indices = [eval_ds.database_num + int(query_index) for query_index in valid_query_indices]

    logging.info(
        "Extracting shared sampled-gallery descriptors on %s (%d database, %d queries, %d models).",
        dataset_name,
        len(sampled_database_indices),
        len(valid_query_indices),
        len(models),
    )
    database_features, database_times, database_input_seconds = extract_features_for_models(
        args,
        eval_ds,
        models,
        sampled_database_indices.tolist(),
        desc=f"{dataset_name}:sampled database",
        test_method="hard_resize",
        batch_size=args.infer_batch_size,
    )
    query_features, query_times, query_input_seconds = extract_features_for_models(
        args,
        eval_ds,
        models,
        sampled_query_dataset_indices,
        desc=f"{dataset_name}:sampled queries",
        test_method=args.test_method,
        batch_size=query_batch_size(args),
    )
    clean_features = {
        model_tag: {"database": database_features[model_tag], "queries": query_features[model_tag]}
        for model_tag in models
    }
    feature_times = {
        model_tag: database_times[model_tag] + query_times[model_tag]
        for model_tag in models
    }
    feature_shared_input_seconds = database_input_seconds + query_input_seconds

    query_counts = {
        "total_queries": int(eval_ds.queries_num),
        "attacked_queries": int(len(valid_query_indices)),
        "skipped_queries_without_positives": int(eval_ds.queries_num - len(select_valid_query_indices(positives, None))),
        "sampled_gallery": True,
        "benchmark_comparable": False,
        "max_dataset_samples": int(args.max_dataset_samples),
        "sampled_query_indices": [int(index) for index in valid_query_indices],
        "sampled_database_indices": [int(index) for index in sampled_database_indices],
        "sampled_database_images": int(len(sampled_database_indices)),
        "dropped_sampled_queries_without_targets": int(dropped_queries),
        "max_adv_negatives": int(max_adv_negatives),
    }
    clean_results = {}
    clean_ranks_by_model = add_sampled_gallery_clean_results(
        clean_results,
        clean_features,
        sampled_positives,
        args.recall_values,
    )
    return {
        "dataset_name": dataset_name,
        "eval_ds": eval_ds,
        "models": models,
        "clean_features": clean_features,
        "feature_times": feature_times,
        "feature_shared_input_seconds": feature_shared_input_seconds,
        "positives": positives,
        "valid_query_indices": valid_query_indices,
        "valid_positives": sampled_positives,
        "sampled_database_indices": sampled_database_indices,
        "clean_results": clean_results,
        "clean_ranks_by_model": clean_ranks_by_model,
        "query_counts": query_counts,
        "target_cache": {},
        "sampled_gallery": True,
    }


def get_context_targets(args, context: MutableMapping[str, object]) -> tuple[list[Dict[str, object]], float]:
    cache = context.setdefault("target_cache", {})
    assert isinstance(cache, dict)
    cache_key = int(args.adv_negatives)
    if cache_key in cache:
        return cache[cache_key]["targets"], 0.0

    target_start = perf_counter()
    reference_tag = attack_reference_tag(args)
    reference_features = context["clean_features"][reference_tag]
    if context.get("sampled_gallery"):
        targets = build_sampled_attack_targets(
            args,
            context["sampled_database_indices"],
            reference_features["database"],
            reference_features["queries"],
            context["valid_query_indices"],
            context["valid_positives"],
        )
    else:
        targets, returned_valid_query_indices = build_attack_targets(
            args,
            context["eval_ds"],
            reference_features["database"],
            reference_features["queries"],
            limit_queries=args.max_queries,
        )
        if not np.array_equal(returned_valid_query_indices, context["valid_query_indices"]):
            raise RuntimeError("Unexpected change in valid query indices while building attack targets.")
    target_seconds = perf_counter() - target_start
    cache[cache_key] = {"targets": targets, "target_seconds": target_seconds}
    return targets, target_seconds


def evaluate_condition_from_context(
    args,
    context: MutableMapping[str, object],
    dataset_name: str,
    epsilon: float,
) -> tuple[Dict[str, Dict[str, object]], Dict[str, object], Dict[str, object], Dict[str, object], list[Dict[str, object]]]:
    results: Dict[str, Dict[str, object]] = copy.deepcopy(context["clean_results"])
    clean_features = context["clean_features"]
    clean_ranks_by_model = context["clean_ranks_by_model"]
    valid_query_indices = context["valid_query_indices"]
    valid_positives = context["valid_positives"]
    models = context["models"]
    reference_tag = attack_reference_tag(args)
    reference_features = clean_features[reference_tag]
    reference_model = models[reference_tag][0]
    database_features_tensor = torch.from_numpy(reference_features["database"]).to(args.device)
    audit_by_condition: Dict[str, object] = {}
    image_manifest: list[Dict[str, object]] = []
    clean_query_feature_indices = (
        np.arange(len(valid_query_indices), dtype=np.int64)
        if context.get("sampled_gallery")
        else None
    )

    targets, target_seconds = get_context_targets(args, context)
    condition_name = f"{args.rank_attack}_eps_{epsilon:g}"
    clear_cuda_cache(args)
    attack = build_rank_attack(reference_model, args, epsilon)
    attack_start = perf_counter()
    attacked_features_by_model, attack_metadata, trace_rows, image_rows = generate_shared_attacked_query_features(
        args,
        context["eval_ds"],
        attack,
        models,
        database_features_tensor,
        reference_features["queries"],
        targets,
        dataset_name,
        epsilon,
        desc=f"{dataset_name}:{condition_name}",
    )
    image_manifest.extend(image_rows)
    elapsed = perf_counter() - attack_start
    attack_metadata_summary = summarize_metadata(attack_metadata)
    write_trace_csvs(
        args.trace_output_dir_path,
        dataset_name,
        args.rank_attack,
        epsilon,
        trace_rows,
        reference_features["database"],
        {int(index): valid_positives[row] for row, index in enumerate(valid_query_indices)},
    )
    if args.audit_attack_implementation:
        audit_by_condition[condition_name] = build_condition_audit(
            args,
            epsilon,
            clean_features,
            attacked_features_by_model,
            valid_query_indices,
            attack_metadata,
            clean_query_feature_indices=clean_query_feature_indices,
        )

    for model_tag, attacked_features in attacked_features_by_model.items():
        model_database_features = clean_features[model_tag]["database"]
        attacked_ranks = nearest_positive_ranks(model_database_features, attacked_features, valid_positives)
        if args.compute_diagnostics:
            diagnostic_rows = compute_query_diagnostic_rows(
                model_database_features,
                clean_features[model_tag]["queries"],
                attacked_features,
                valid_positives,
                valid_query_indices,
                targets,
                attack_metadata["perturbation_norm"].detach().cpu().numpy(),
                clean_query_feature_indices=clean_query_feature_indices,
                clean_ranks=clean_ranks_by_model[model_tag],
                attacked_ranks=attacked_ranks,
            )
            diagnostics_path = (
                args.diagnostics_output_dir_path
                / (
                    f"{_filename_token(dataset_name)}_{_filename_token(model_tag)}_"
                    f"{_filename_token(args.rank_attack)}_eps_{_epsilon_label(epsilon)}.csv"
                )
            )
            write_diagnostics_csv(diagnostics_path, diagnostic_rows)
        attacked_recalls = compute_recalls_from_features(
            model_database_features,
            attacked_features,
            valid_positives,
            args.recall_values,
        )
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
            "attacked_queries": context["query_counts"]["attacked_queries"],
            "sampled_gallery": bool(context.get("sampled_gallery")),
            "benchmark_comparable": not bool(context.get("sampled_gallery")),
            "rank_displacement": displacement,
            "attack_success": success,
            "runtime_seconds": elapsed,
            "runtime_per_query_seconds": float(elapsed / max(1, len(valid_query_indices))),
            "attack_metadata": attack_metadata_summary,
        }
        logging.info("%s/%s recalls: %s", model_tag, condition_name, attacked_recalls["recalls_str"])

    runtimes = {
        "feature_seconds": context["feature_times"],
        "feature_shared_input_seconds": context["feature_shared_input_seconds"],
        "target_seconds": target_seconds,
        "attack_seconds": {condition_name: elapsed},
    }
    return results, runtimes, context["query_counts"], audit_by_condition, image_manifest


def evaluate_dataset(args, dataset_name: str, models: Mapping[str, Tuple[nn.Module, object]]):
    if args.audit_sample_database_size is not None:
        return evaluate_audit_sample_dataset(args, dataset_name, models)

    context = prepare_dataset_context(args, dataset_name, models)
    results: Dict[str, Dict[str, object]] = copy.deepcopy(context["clean_results"])
    runtimes = {
        "feature_seconds": context["feature_times"],
        "feature_shared_input_seconds": context["feature_shared_input_seconds"],
        "target_seconds": 0.0,
        "attack_seconds": {},
    }
    query_counts = context["query_counts"]
    audit_by_condition: Dict[str, object] = {}
    image_manifest: list[Dict[str, object]] = []

    for epsilon in args.epsilons:
        condition_results, condition_runtimes, _, condition_audit, condition_images = evaluate_condition_from_context(
            args,
            context,
            dataset_name,
            epsilon,
        )
        for model_tag, model_results in condition_results.items():
            results.setdefault(model_tag, {}).update(model_results)
        runtimes["target_seconds"] += float(condition_runtimes["target_seconds"])
        runtimes["attack_seconds"].update(condition_runtimes["attack_seconds"])
        audit_by_condition.update(condition_audit)
        image_manifest.extend(condition_images)

    return results, runtimes, query_counts, audit_by_condition, image_manifest


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


def write_attack_image_manifest(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "dataset",
        "attack",
        "epsilon",
        "query_index",
        "clean",
        "attacked",
        "perturbation",
        "abs_heatmap",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_arguments()
    output_json, output_csv, run_dir = build_output_paths(args)
    audit_output_json = build_audit_output_path(args, run_dir)
    trace_output_dir = build_trace_output_dir(args, run_dir)
    diagnostics_output_dir = build_diagnostics_output_dir(args, run_dir) if args.compute_diagnostics else None
    attack_image_output_dir = build_attack_image_output_dir(args, run_dir)
    args.trace_output_dir_path = trace_output_dir
    args.diagnostics_output_dir_path = diagnostics_output_dir
    args.attack_image_output_dir_path = attack_image_output_dir
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
    audit_results = {}
    attack_image_manifest: list[Dict[str, object]] = []
    for dataset_name in args.datasets:
        logging.info("Evaluating %s.", dataset_name)
        dataset_results, dataset_runtimes, dataset_query_counts, dataset_audit, dataset_images = evaluate_dataset(
            args,
            dataset_name,
            models,
        )
        results[dataset_name] = dataset_results
        runtimes[dataset_name] = dataset_runtimes
        query_counts[dataset_name] = dataset_query_counts
        attack_image_manifest.extend(dataset_images)
        if args.audit_attack_implementation:
            audit_results[dataset_name] = dataset_audit

    rows = flatten_rows(results, args.recall_values)
    attack_image_manifest_path = attack_image_output_dir / "attack_image_manifest.csv"
    report = {
        "timestamp": started_at.isoformat(),
        "command": " ".join(shlex.quote(argument) for argument in sys.argv),
        "argv": sys.argv,
        "model_type": args.model_type,
        "checkpoints": dict(zip(args.model_tags, args.model_paths)),
        "model_configuration": {
            "input_size": list(args.resize),
            "descriptor_dimensions": {
                model_tag: int(model_args.features_dim)
                for model_tag, (_, model_args) in models.items()
            },
        },
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
                "max_dataset_samples": args.max_dataset_samples,
                "sampled_gallery_mode": args.max_dataset_samples is not None,
                "audit_sample_database_size": args.audit_sample_database_size,
                "audit_sample_mode": args.audit_sample_database_size is not None,
            },
            "query_counts": query_counts,
        },
        "results": results,
        "runtime_seconds": runtimes,
        "output_json": str(output_json),
        "output_csv": str(output_csv),
        "audit_output_json": str(audit_output_json) if audit_output_json is not None else None,
        "trace_output_dir": str(trace_output_dir),
        "diagnostics_output_dir": str(diagnostics_output_dir) if diagnostics_output_dir is not None else None,
        "attack_image_output_dir": str(attack_image_output_dir),
        "attack_image_manifest_csv": str(attack_image_manifest_path) if attack_image_manifest else None,
        "attack_image_manifest": attack_image_manifest,
        "duration_seconds": (datetime.now() - started_at).total_seconds(),
    }
    if args.audit_attack_implementation:
        report["implementation_audit"] = {
            "purpose": "Phase 1 native rank attack implementation audit",
            "epsilon_convention": "normalized_image_tensor",
            "epsilon_raw_pixel_equivalents": [
                normalized_epsilon_to_raw_pixels(float(epsilon)) for epsilon in args.epsilons
            ],
            "datasets": audit_results,
            "notes": [
                "Audit mode records per-condition summaries from the attacked valid query subset.",
                "Descriptor norm checks summarize model outputs and do not force descriptor normalization.",
                "Gradient checks are computed during attack optimization forward passes.",
                "When audit_sample_database_size is set, recall and rank metrics are audit-only and not benchmark-comparable.",
                "When max_dataset_samples is set, recall and rank metrics use a deterministic sampled gallery and are not benchmark-comparable.",
            ],
        }

    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    if audit_output_json is not None:
        with audit_output_json.open("w", encoding="utf-8") as handle:
            json.dump(report["implementation_audit"], handle, indent=2)
    write_csv(output_csv, rows, args.recall_values)
    write_attack_image_manifest(attack_image_manifest_path, attack_image_manifest)

    logging.info("Saved JSON rank evaluation report to %s", output_json)
    if audit_output_json is not None:
        logging.info("Saved JSON implementation audit to %s", audit_output_json)
    logging.info("Saved CSV rank evaluation summary to %s", output_csv)
    logging.info("Trace CSV directory: %s", trace_output_dir)
    if diagnostics_output_dir is not None:
        logging.info("Diagnostics CSV directory: %s", diagnostics_output_dir)
    else:
        logging.info("Per-query diagnostics disabled; use --compute_diagnostics to enable them.")
    if args.save_attack_images:
        logging.info("Attack image directory: %s", attack_image_output_dir)
    logging.info("Finished in %s", str(datetime.now() - started_at)[:-7])


if __name__ == "__main__":
    main()
