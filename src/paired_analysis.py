"""Paired analysis of per-query rank CSVs produced by ``src/rank_eval.py``.

Per-model CC-ASR conditions on each model's *own* clean-correct queries, so two models
are scored on different query sets. When adversarial fine-tuning shrinks the clean-correct
set, a lower per-model CC-ASR can reflect an easier denominator rather than real robustness.
Intersection CC-ASR fixes the denominator to the queries both models get right cleanly,
which is the only comparison where the two rates are directly commensurable.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import csv

INTEGER_FIELDS = ("query_id", "clean_rank", "attacked_rank")
BOOLEAN_FIELDS = ("clean_correct_at_1", "attacked_correct_at_1")
PAIR_KEY_FIELDS = ("dataset", "condition", "query_id")


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_per_query_rows(path: Path) -> list[Dict[str, object]]:
    """Read a ``per_query_ranks.csv`` file, restoring integer and boolean column types."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for field in INTEGER_FIELDS:
            row[field] = int(row[field])
        for field in BOOLEAN_FIELDS:
            row[field] = _parse_bool(row[field])
        if row.get("epsilon", "") != "":
            row["epsilon"] = float(row["epsilon"])
    return rows


def _filter_rows(
    rows: Sequence[Mapping[str, object]],
    source: str,
    model_tag: str | None,
    dataset: str | None,
    condition: str | None,
) -> list[Mapping[str, object]]:
    selected = list(rows)
    if dataset is not None:
        selected = [row for row in selected if row["dataset"] == dataset]
    if condition is not None:
        selected = [row for row in selected if row["condition"] == condition]
    if model_tag is not None:
        selected = [row for row in selected if row["model_tag"] == model_tag]

    model_tags = {row["model_tag"] for row in selected}
    if len(model_tags) > 1:
        raise ValueError(
            f"{source} contains rows for model tags {sorted(model_tags)}; pass an explicit model_tag "
            "to select one of them."
        )
    if not selected:
        raise ValueError(f"{source} contains no rows for the requested model_tag/dataset/condition.")
    return selected


def _pair_key(row: Mapping[str, object]) -> tuple:
    return tuple(row[field] for field in PAIR_KEY_FIELDS)


def _ccasr(rows: Sequence[Mapping[str, object]]) -> tuple[float, int]:
    """Attack success rate over the clean-correct@1 subset of ``rows``, as a percentage."""
    clean_correct = [row for row in rows if row["clean_correct_at_1"]]
    if not clean_correct:
        return 0.0, 0
    broken = sum(1 for row in clean_correct if not row["attacked_correct_at_1"])
    return broken / len(clean_correct) * 100.0, len(clean_correct)


def compute_paired_ccasr(
    base_csv: Path,
    trained_csv: Path,
    base_model_tag: str | None = None,
    trained_model_tag: str | None = None,
    dataset: str | None = None,
    condition: str | None = None,
) -> Dict[str, object]:
    """Report CC-ASR per model and on the queries both models answer correctly cleanly.

    ``base_csv`` and ``trained_csv`` may be the same file when one evaluation run covered
    both checkpoints; in that case the model tags must be given explicitly.
    """
    base_rows = _filter_rows(
        load_per_query_rows(base_csv), f"{base_csv}", base_model_tag, dataset, condition
    )
    trained_rows = _filter_rows(
        load_per_query_rows(trained_csv), f"{trained_csv}", trained_model_tag, dataset, condition
    )

    base_by_key = {_pair_key(row): row for row in base_rows}
    trained_by_key = {_pair_key(row): row for row in trained_rows}
    if set(base_by_key) != set(trained_by_key):
        raise ValueError(
            "Paired CC-ASR requires both CSVs to cover the same queries, but they differ on "
            f"{len(set(base_by_key) ^ set(trained_by_key))} (dataset, condition, query_id) keys."
        )

    ccasr_own_base, n_clean_correct_base = _ccasr(base_rows)
    ccasr_own_trained, n_clean_correct_trained = _ccasr(trained_rows)

    intersection_keys = [
        key
        for key in base_by_key
        if base_by_key[key]["clean_correct_at_1"] and trained_by_key[key]["clean_correct_at_1"]
    ]
    intersection_base = [base_by_key[key] for key in intersection_keys]
    intersection_trained = [trained_by_key[key] for key in intersection_keys]
    ccasr_intersection_base, _ = _ccasr(intersection_base)
    ccasr_intersection_trained, _ = _ccasr(intersection_trained)

    return {
        "ccasr_own_base": ccasr_own_base,
        "ccasr_own_trained": ccasr_own_trained,
        "ccasr_intersection_base": ccasr_intersection_base,
        "ccasr_intersection_trained": ccasr_intersection_trained,
        "n_intersection": len(intersection_keys),
        "n_clean_correct_base": n_clean_correct_base,
        "n_clean_correct_trained": n_clean_correct_trained,
        "n_paired": len(base_by_key),
        "model_tag_base": base_rows[0]["model_tag"],
        "model_tag_trained": trained_rows[0]["model_tag"],
        "base_csv": str(base_csv),
        "trained_csv": str(trained_csv),
        "dataset": dataset,
        "condition": condition,
        "rate_units": "percent",
    }


def conditions_in(rows: Sequence[Mapping[str, object]]) -> list[tuple[str, str]]:
    """Every (dataset, condition) pair present, in first-seen order."""
    seen: list[tuple[str, str]] = []
    for row in rows:
        key = (str(row["dataset"]), str(row["condition"]))
        if key not in seen:
            seen.append(key)
    return seen


def compute_paired_ccasr_by_condition(
    base_csv: Path,
    trained_csv: Path,
    base_model_tag: str | None = None,
    trained_model_tag: str | None = None,
) -> list[Dict[str, object]]:
    """Run :func:`compute_paired_ccasr` once per (dataset, condition) pair in the base CSV."""
    return [
        compute_paired_ccasr(
            base_csv,
            trained_csv,
            base_model_tag=base_model_tag,
            trained_model_tag=trained_model_tag,
            dataset=dataset,
            condition=condition,
        )
        for dataset, condition in conditions_in(load_per_query_rows(base_csv))
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute per-model and intersection CC-ASR from per-query rank CSVs."
    )
    parser.add_argument("--base_csv", type=str, required=True, help="per_query_ranks.csv holding the base model rows.")
    parser.add_argument(
        "--trained_csv",
        type=str,
        default=None,
        help="per_query_ranks.csv holding the trained model rows. Defaults to --base_csv.",
    )
    parser.add_argument("--base_model_tag", type=str, default=None, help="model_tag identifying the base model.")
    parser.add_argument("--trained_model_tag", type=str, default=None, help="model_tag identifying the trained model.")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Where to write the paired summaries. Defaults to paired_ccasr.json beside --base_csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_csv = Path(args.base_csv).expanduser()
    trained_csv = Path(args.trained_csv).expanduser() if args.trained_csv else base_csv
    output_json = (
        Path(args.output_json).expanduser() if args.output_json else base_csv.parent / "paired_ccasr.json"
    )

    summaries = compute_paired_ccasr_by_condition(
        base_csv,
        trained_csv,
        base_model_tag=args.base_model_tag,
        trained_model_tag=args.trained_model_tag,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump({"paired_ccasr": summaries}, handle, indent=2)

    for summary in summaries:
        print(
            f"{summary['dataset']}/{summary['condition']}: "
            f"own base {summary['ccasr_own_base']:.2f}% (n={summary['n_clean_correct_base']}), "
            f"own trained {summary['ccasr_own_trained']:.2f}% (n={summary['n_clean_correct_trained']}), "
            f"intersection base {summary['ccasr_intersection_base']:.2f}%, "
            f"intersection trained {summary['ccasr_intersection_trained']:.2f}% "
            f"(n={summary['n_intersection']})"
        )
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
