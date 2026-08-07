"""Phase 1 three-way comparison table: pretrained vs clean-FT vs PAT.

Task 1.2 of the full-conference plan requires the paper to attribute effects to
adversarial training rather than to fine-tuning itself. That needs all three arms side by
side, aggregated over seeds, with deltas measured against the pretrained reference. This
module reads the JSON summaries written by ``eval.py`` (``src/rank_eval.py``) and produces
that table.

Model tags carry the arm and seed, e.g. ``pretrained``, ``clean_ft_s0``, ``pat_s1``. The
per-model CC-ASR reported here is the *own-clean-correct* rate; the intersection CC-ASR the
supervisor asked for comes from ``src/paired_analysis.py`` on the per-query CSVs, because it
is defined over a query subset shared by two models and cannot be recovered from either
model's summary alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

CLEAN_CONDITION = "clean_all_queries"
ATTACK_CONDITION_PATTERN = re.compile(r"^(?P<attack>.+)_eps_(?P<epsilon>[0-9.]+)$")
MODEL_TAG_PATTERN = re.compile(r"^(?P<arm>.+?)(?:_s(?P<seed>\d+))?$")
PRETRAINED_ARM = "pretrained"

# Arms are reported in this order when present; unknown arms follow alphabetically.
ARM_ORDER = (PRETRAINED_ARM, "clean_ft", "pat")


def parse_model_tag(model_tag: str) -> tuple[str, Optional[int]]:
    """Split ``clean_ft_s1`` into ``("clean_ft", 1)``; unseeded tags return seed ``None``."""
    match = MODEL_TAG_PATTERN.match(model_tag)
    if match is None:  # pragma: no cover - the pattern matches any non-empty string
        return model_tag, None
    seed = match.group("seed")
    return match.group("arm"), None if seed is None else int(seed)


def summarize_seeds(values: Sequence[float]) -> Dict[str, float]:
    """Mean and range over seeds.

    With only two seeds a standard deviation is not meaningful, so the plan asks for
    mean +/- range. ``half_range`` is the number printed after the +/-.
    """
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("summarize_seeds requires at least one value")
    lowest = min(numbers)
    highest = max(numbers)
    return {
        "mean": sum(numbers) / len(numbers),
        "min": lowest,
        "max": highest,
        "half_range": (highest - lowest) / 2.0,
        "n": len(numbers),
    }


def _recall_at_1(condition: Mapping[str, object]) -> Optional[float]:
    recalls = condition.get("recalls")
    if not isinstance(recalls, Mapping) or "R@1" not in recalls:
        return None
    return float(recalls["R@1"])


def _own_ccasr(condition: Mapping[str, object]) -> Optional[float]:
    attack_success = condition.get("attack_success")
    if not isinstance(attack_success, Mapping):
        return None
    clean_correct = attack_success.get("clean_correct")
    if not isinstance(clean_correct, Mapping) or "rate" not in clean_correct:
        return None
    return float(clean_correct["rate"])


def _median_displacement(condition: Mapping[str, object]) -> Optional[float]:
    displacement = condition.get("rank_displacement")
    if not isinstance(displacement, Mapping) or "median" not in displacement:
        return None
    return float(displacement["median"])


def load_results(paths: Sequence[Path]) -> List[Mapping[str, object]]:
    documents = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            documents.append(json.load(handle))
    return documents


def collect_measurements(documents: Sequence[Mapping[str, object]]) -> Dict[tuple, Dict[str, List[float]]]:
    """Group per-seed measurements by ``(dataset, attack, epsilon, arm)``.

    Several JSON documents can contribute to the same cell — one eval run per seed is the
    common case — so measurements accumulate across documents rather than overwriting.
    """
    measurements: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for document in documents:
        results = document.get("results")
        if not isinstance(results, Mapping):
            continue
        for dataset_name, per_model in results.items():
            if not isinstance(per_model, Mapping):
                continue
            for model_tag, conditions in per_model.items():
                if not isinstance(conditions, Mapping):
                    continue
                arm, _seed = parse_model_tag(str(model_tag))
                clean_r1 = _recall_at_1(conditions.get(CLEAN_CONDITION, {}))

                for condition_name, condition in conditions.items():
                    match = ATTACK_CONDITION_PATTERN.match(str(condition_name))
                    if match is None or not isinstance(condition, Mapping):
                        continue
                    epsilon = float(condition.get("epsilon", match.group("epsilon")))
                    key = (dataset_name, match.group("attack"), epsilon, arm)

                    if clean_r1 is not None:
                        measurements[key]["clean_r1"].append(clean_r1)
                    attacked_r1 = _recall_at_1(condition)
                    if attacked_r1 is not None:
                        measurements[key]["attacked_r1"].append(attacked_r1)
                    ccasr = _own_ccasr(condition)
                    if ccasr is not None:
                        measurements[key]["ccasr_own"].append(ccasr)
                    displacement = _median_displacement(condition)
                    if displacement is not None:
                        measurements[key]["median_displacement"].append(displacement)

    return measurements


def _arm_sort_key(arm: str) -> tuple:
    if arm in ARM_ORDER:
        return (0, ARM_ORDER.index(arm), arm)
    return (1, 0, arm)


def build_three_way_table(paths: Sequence[Path]) -> List[Dict[str, object]]:
    """Build the pretrained / clean-FT / PAT comparison rows from eval JSON summaries."""
    measurements = collect_measurements(load_results(paths))

    pretrained_means: Dict[tuple, Dict[str, float]] = {}
    for (dataset_name, attack, epsilon, arm), metrics in measurements.items():
        if arm != PRETRAINED_ARM:
            continue
        pretrained_means[(dataset_name, attack, epsilon)] = {
            metric: summarize_seeds(values)["mean"] for metric, values in metrics.items()
        }

    rows: List[Dict[str, object]] = []
    for (dataset_name, attack, epsilon, arm), metrics in measurements.items():
        row: Dict[str, object] = {
            "dataset": dataset_name,
            "attack": attack,
            "epsilon": epsilon,
            "arm": arm,
        }
        seed_counts = set()
        for metric, values in metrics.items():
            summary = summarize_seeds(values)
            row[metric] = summary
            seed_counts.add(summary["n"])
        row["n_seeds"] = max(seed_counts) if seed_counts else 0

        reference = pretrained_means.get((dataset_name, attack, epsilon))
        for metric in ("clean_r1", "attacked_r1", "ccasr_own", "median_displacement"):
            summary = row.get(metric)
            if reference is None or metric not in reference or not isinstance(summary, Mapping):
                row[f"{metric}_delta_vs_pretrained"] = None
            else:
                row[f"{metric}_delta_vs_pretrained"] = summary["mean"] - reference[metric]
        rows.append(row)

    rows.sort(key=lambda row: (row["dataset"], row["attack"], row["epsilon"], _arm_sort_key(row["arm"])))
    return rows


def _format_cell(summary: object) -> str:
    if not isinstance(summary, Mapping):
        return "-"
    if summary["n"] < 2:
        return f"{summary['mean']:.2f}"
    return f"{summary['mean']:.2f} ± {summary['half_range']:.2f}"


def _format_delta(delta: object) -> str:
    if delta is None:
        return "-"
    return f"{float(delta):+.2f}"


TABLE_COLUMNS = (
    "dataset",
    "attack",
    "epsilon",
    "arm",
    "n_seeds",
    "clean R@1",
    "Δ clean R@1",
    "attacked R@1",
    "Δ attacked R@1",
    "CC-ASR (own)",
    "median displacement",
)


def format_table_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(TABLE_COLUMNS)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["attack"]),
                    f"{row['epsilon']:g}",
                    str(row["arm"]),
                    str(row["n_seeds"]),
                    _format_cell(row.get("clean_r1")),
                    _format_delta(row.get("clean_r1_delta_vs_pretrained")),
                    _format_cell(row.get("attacked_r1")),
                    _format_delta(row.get("attacked_r1_delta_vs_pretrained")),
                    _format_cell(row.get("ccasr_own")),
                    _format_cell(row.get("median_displacement")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


CSV_FIELDNAMES = (
    "dataset",
    "attack",
    "epsilon",
    "arm",
    "n_seeds",
    "clean_r1_mean",
    "clean_r1_half_range",
    "clean_r1_delta_vs_pretrained",
    "attacked_r1_mean",
    "attacked_r1_half_range",
    "attacked_r1_delta_vs_pretrained",
    "ccasr_own_mean",
    "ccasr_own_half_range",
    "median_displacement_mean",
)


def write_table_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        for row in rows:
            record = {
                "dataset": row["dataset"],
                "attack": row["attack"],
                "epsilon": row["epsilon"],
                "arm": row["arm"],
                "n_seeds": row["n_seeds"],
            }
            for metric in ("clean_r1", "attacked_r1", "ccasr_own", "median_displacement"):
                summary = row.get(metric)
                if isinstance(summary, Mapping):
                    record[f"{metric}_mean"] = f"{summary['mean']:.4f}"
                    if f"{metric}_half_range" in CSV_FIELDNAMES:
                        record[f"{metric}_half_range"] = f"{summary['half_range']:.4f}"
                delta = row.get(f"{metric}_delta_vs_pretrained")
                if f"{metric}_delta_vs_pretrained" in CSV_FIELDNAMES and delta is not None:
                    record[f"{metric}_delta_vs_pretrained"] = f"{float(delta):.4f}"
            writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_json",
        type=str,
        nargs="+",
        required=True,
        help="rank_eval_results.json files to aggregate (one or more, across seeds and datasets).",
    )
    parser.add_argument("--output_markdown", type=str, default=None, help="Write the table as markdown.")
    parser.add_argument("--output_csv", type=str, default=None, help="Write the table as CSV.")
    args = parser.parse_args()

    rows = build_three_way_table([Path(path) for path in args.results_json])
    markdown = format_table_markdown(rows)
    print(markdown)

    if args.output_markdown is not None:
        output_path = Path(args.output_markdown)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown + "\n", encoding="utf-8")
    if args.output_csv is not None:
        write_table_csv(Path(args.output_csv), rows)


if __name__ == "__main__":
    main()
