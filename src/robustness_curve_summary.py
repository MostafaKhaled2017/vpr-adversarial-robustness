"""Summarize rank-evaluation CSVs into R@1-vs-epsilon curves with normalized area under curve.

Processes attack results to compute robustness summary: R@1 regression per attack epsilon,
area-under-curve normalized by epsilon range, and clean accuracy drop vs reference model.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from typing import Dict, List, Mapping, Optional, Sequence

from src.config import normalized_epsilon_to_raw_pixels

CLEAN_CONDITION = "clean_all_queries"
# Attacked R@1 is measured on the valid-query subset, so the curve's eps=0 point uses the same subset.
CURVE_ANCHOR_CONDITION = "clean_attacked_subset"
EPSILON_PATTERN = re.compile(r"_eps_[^_]+")


def normalized_auc(epsilons: Sequence[float], recalls: Sequence[float], clean: float) -> float:
    """Compute trapezoid area under curve, normalized by epsilon range.

    Args:
        epsilons: Attack epsilon values (increasing order)
        recalls: R@1 recall at each epsilon
        clean: Clean R@1 (at epsilon 0)

    Returns:
        Normalized area under curve (area / max_epsilon)
    """
    xs = [0.0, *epsilons]
    ys = [clean, *recalls]
    area = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2.0 for i in range(len(xs) - 1))
    return area / xs[-1]


def summarize(rows: Sequence[Mapping[str, str]], reference_model: Optional[str] = None) -> List[Dict[str, object]]:
    """Summarize rank evaluation results by attack family and model.

    Args:
        rows: CSV rows (dicts with keys: dataset, model, condition, epsilon, R@1, targeted_success_rate)
        reference_model: Model to use as reference for clean accuracy drop (e.g., 'clean_ft')

    Returns:
        List of summary dicts with keys: dataset, model, family, clean_r1, r1_by_epsilon,
        auc, clean_drop_vs_reference, mean_targeted_success. clean_r1 and the clean drop use
        clean_all_queries; the AUC's eps=0 point uses clean_attacked_subset (falling back to
        clean_all_queries when absent).
    """
    clean: Dict[tuple, float] = {}
    anchors: Dict[tuple, float] = {}
    curves: Dict[tuple, Dict[float, float]] = defaultdict(dict)
    successes: Dict[tuple, List[float]] = defaultdict(list)

    for row in rows:
        key = (row["dataset"], row["model"])
        if row["condition"] == CLEAN_CONDITION:
            clean[key] = float(row["R@1"])
        elif row["condition"] == CURVE_ANCHOR_CONDITION:
            anchors[key] = float(row["R@1"])
        if row.get("epsilon"):
            family_key = (*key, EPSILON_PATTERN.sub("", row["condition"]))
            epsilon = float(row["epsilon"])
            if epsilon in curves[family_key]:
                raise ValueError(f"duplicate curve point (dataset, model, family, epsilon)={(*family_key, epsilon)}; "
                                 "were CSVs from different runs combined?")
            curves[family_key][epsilon] = float(row["R@1"])
            if row.get("targeted_success_rate"):
                successes[family_key].append(float(row["targeted_success_rate"]))

    summary = []
    for (dataset, model, family), curve in sorted(curves.items()):
        epsilons = sorted(curve)
        clean_r1 = clean[(dataset, model)]
        anchor_r1 = anchors.get((dataset, model), clean_r1)
        reference = clean.get((dataset, reference_model)) if reference_model else None
        summary.append(
            {
                "dataset": dataset,
                "model": model,
                "family": family,
                "clean_r1": clean_r1,
                "r1_by_epsilon": {epsilon: curve[epsilon] for epsilon in epsilons},
                "auc": normalized_auc(epsilons, [curve[epsilon] for epsilon in epsilons], anchor_r1),
                "clean_drop_vs_reference": (reference - clean_r1) if reference is not None else None,
                "mean_targeted_success": sum(successes[dataset, model, family]) / len(successes[dataset, model, family]) if successes[dataset, model, family] else None,
            }
        )
    return summary


def worst_case(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    """Worst-case R@1 per (dataset, model, epsilon) from per_query_ranks.csv rows.

    A query counts as correct only if it stays correct at rank 1 under every attack run at
    that epsilon. Every attack must cover the same queries, or the union is meaningless.
    """
    queries: Dict[tuple, Dict[str, set]] = defaultdict(dict)
    failed: Dict[tuple, set] = defaultdict(set)
    for row in rows:
        if row["attacked_rank"] == "-1":  # no positive in the gallery: excluded, as in every summary
            continue
        key = (row["dataset"], row["model_tag"], float(row["epsilon"]))
        queries[key].setdefault(EPSILON_PATTERN.sub("", row["condition"]), set()).add(row["query_id"])
        if row["attacked_correct_at_1"] != "True":
            failed[key].add(row["query_id"])

    summary = []
    for key, by_attack in sorted(queries.items()):
        query_sets = list(by_attack.values())
        if any(ids != query_sets[0] for ids in query_sets):
            raise ValueError(f"attacks at (dataset, model, epsilon)={key} cover different queries; "
                             "were they run with the same --max-queries?")
        total = len(query_sets[0])
        summary.append({
            "dataset": key[0], "model": key[1], "epsilon": key[2],
            "attacks": " ".join(sorted(by_attack)), "queries": total,
            "worst_case_r1": 100.0 * (total - len(failed[key])) / total,
        })
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", help="rank_eval_results.csv paths (per_query_ranks.csv with --worst_case)")
    parser.add_argument("--reference_model", default=None, help="Model for R@1 reference (clean accuracy drop)")
    parser.add_argument("--worst_case", action="store_true",
                        help="Read per_query_ranks.csv files and report worst-case R@1 over all attacks per epsilon")
    parser.add_argument("--output", default=None, help="Output CSV path (default: stdout)")
    args = parser.parse_args(argv)

    rows = []
    for csv_path in args.csv_paths:
        with open(csv_path, encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    if args.worst_case:
        fieldnames = ["dataset", "model", "epsilon", "attacks", "queries", "worst_case_r1"]
        records = worst_case(rows)
    else:
        summary = summarize(rows, args.reference_model)
        epsilons = sorted(set(epsilon for item in summary for epsilon in item["r1_by_epsilon"]))
        labels = {epsilon: f"R@1@<={normalized_epsilon_to_raw_pixels(epsilon)['max_raw_255']:.1f}/255" for epsilon in epsilons}
        fieldnames = ["dataset", "model", "family", "clean_r1", *labels.values(), "auc", "clean_drop_vs_reference", "mean_targeted_success"]
        records = []
        for item in summary:
            record = {name: item[name] for name in ("dataset", "model", "family", "clean_r1", "auc", "clean_drop_vs_reference", "mean_targeted_success")}
            record.update({labels[epsilon]: value for epsilon, value in item["r1_by_epsilon"].items()})
            records.append(record)

    handle = open(args.output, "w", encoding="utf-8", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    finally:
        if args.output:
            handle.close()

if __name__ == "__main__":
    main()
