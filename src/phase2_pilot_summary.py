"""Rank Phase 2 pilot configurations on a common held-out validation threat model."""

from __future__ import annotations

import argparse
import csv
import filecmp
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_MAX_CLEAN_DROP = 2.0


def _read_float(row: Mapping[str, str], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def best_validation_epoch(run_dir: Path) -> Optional[Dict[str, object]]:
    """Best trained validation epoch, excluding the pre-training epoch -1."""
    csv_path = Path(run_dir) / "validation_recalls.csv"
    if not csv_path.is_file():
        return None

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    clean_rows = [
        row
        for row in rows
        if row.get("split") == "clean" and _read_float(row, "epoch") is not None and _read_float(row, "epoch") >= 0
    ]
    if not clean_rows:
        return None

    best_clean = max(clean_rows, key=lambda row: _read_float(row, "selection_score") or float("-inf"))
    best_epoch = best_clean.get("epoch")

    attacked_r1 = None
    for row in rows:
        if row.get("split") == "attacked_mean" and row.get("epoch") == best_epoch:
            attacked_r1 = _read_float(row, "R@1")
            break

    return {
        "epoch": int(float(best_epoch)),
        "clean_r1": _read_float(best_clean, "R@1"),
        "attacked_r1": attacked_r1,
        "clean_score": _read_float(best_clean, "clean_score"),
        "robust_score": _read_float(best_clean, "robust_score"),
        "selection_score": _read_float(best_clean, "selection_score"),
    }


def selected_checkpoint_is_initial(run_dir: Path) -> bool:
    """Whether the run selected its unchanged pre-training checkpoint."""
    best_path = run_dir / "best_model.pth"
    initial_path = run_dir / "initial_validation_model.pth"
    return best_path.is_file() and initial_path.is_file() and filecmp.cmp(best_path, initial_path, shallow=False)


def read_collapse_metrics(run_dir: Path, epoch: int) -> Optional[Dict[str, float]]:
    """Collapse report recorded at a given epoch, if monitoring was enabled."""
    jsonl_path = Path(run_dir) / "validation_recalls.jsonl"
    if not jsonl_path.is_file():
        return None

    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if int(record.get("epoch", -999)) != int(epoch):
                continue
            collapse = record.get("collapse")
            return collapse if isinstance(collapse, Mapping) else None
    return None


def meets_acceptance(
    clean_r1: Optional[float],
    attacked_r1: Optional[float],
    clean_ft_clean_r1: Optional[float],
    pat_attacked_r1: Optional[float],
    max_clean_drop: float = DEFAULT_MAX_CLEAN_DROP,
) -> Optional[bool]:
    """Task 2.6's criterion, or ``None`` when the Phase 1 baselines are unknown."""
    if clean_ft_clean_r1 is None or pat_attacked_r1 is None:
        return None
    if clean_r1 is None or attacked_r1 is None:
        return False
    within_clean_budget = (clean_ft_clean_r1 - clean_r1) <= max_clean_drop
    matches_pat_robustness = attacked_r1 >= pat_attacked_r1
    return bool(within_clean_budget and matches_pat_robustness)


def rank_pilot_runs(
    runs: Sequence[Tuple[str, Path]],
    clean_ft_clean_r1: Optional[float],
    pat_attacked_r1: Optional[float],
    max_clean_drop: float = DEFAULT_MAX_CLEAN_DROP,
) -> List[Dict[str, object]]:
    """Rank pilot runs: accepted configurations first, then by robustness gain.

    Ordering accepted-before-rejected matters — the most robust configuration is often one
    that has traded away far too much clean recall, and picking it would reproduce exactly
    the trade-off the supervisor asked us to fix.
    """
    rows: List[Dict[str, object]] = []
    for label, run_dir in runs:
        best = best_validation_epoch(Path(run_dir))
        if best is None:
            continue

        initial_checkpoint = selected_checkpoint_is_initial(Path(run_dir))
        accepted = False if initial_checkpoint else meets_acceptance(
            best["clean_r1"],
            best["attacked_r1"],
            clean_ft_clean_r1,
            pat_attacked_r1,
            max_clean_drop,
        )
        row: Dict[str, object] = {
            "label": label,
            "run_dir": str(run_dir),
            "epoch": best["epoch"],
            "clean_r1": best["clean_r1"],
            "attacked_r1": best["attacked_r1"],
            "selection_score": best["selection_score"],
            "accepted": accepted,
            "eligible": not initial_checkpoint,
            "status": "excluded_initial_checkpoint" if initial_checkpoint else "eligible",
            "collapse": read_collapse_metrics(Path(run_dir), best["epoch"]),
        }
        row["clean_r1_drop_vs_clean_ft"] = (
            None if clean_ft_clean_r1 is None or best["clean_r1"] is None else clean_ft_clean_r1 - best["clean_r1"]
        )
        row["attacked_r1_gain_vs_pat"] = (
            None if pat_attacked_r1 is None or best["attacked_r1"] is None else best["attacked_r1"] - pat_attacked_r1
        )
        rows.append(row)

    def sort_key(row: Mapping[str, object]):
        gain = row.get("attacked_r1_gain_vs_pat")
        fallback = row.get("attacked_r1")
        robustness = gain if gain is not None else (fallback if fallback is not None else float("-inf"))
        if not row.get("eligible"):
            group = 2
        elif row.get("accepted"):
            group = 0
        else:
            group = 1
        return (group, -float(robustness))

    rows.sort(key=sort_key)
    return rows


def read_common_evaluation(path: Path) -> Dict[str, Dict[str, float]]:
    """Read clean and mean attacked R@1 per model from a fixed val-split rank evaluation."""
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("dataset_split") != "val":
        raise ValueError("--common_eval_json must be a rank evaluation of the validation split.")
    attack = report.get("attack", {})
    if attack.get("mode") != "rank_pgd_linf":
        raise ValueError("--common_eval_json must use rank_pgd_linf.")

    metrics: Dict[str, Dict[str, list[float]]] = {}
    for models in report.get("results", {}).values():
        for tag, conditions in models.items():
            clean = conditions.get("clean_all_queries", {}).get("recalls", {}).get("R@1")
            attacked = [
                condition.get("recalls", {}).get("R@1")
                for name, condition in conditions.items()
                if name.startswith("rank_pgd_linf_eps_")
            ]
            if clean is None or not attacked or any(value is None for value in attacked):
                continue
            record = metrics.setdefault(tag, {"clean_r1": [], "attacked_r1": []})
            record["clean_r1"].append(float(clean))
            record["attacked_r1"].extend(float(value) for value in attacked)
    return {
        tag: {"clean_r1": mean(values["clean_r1"]), "attacked_r1": mean(values["attacked_r1"])}
        for tag, values in metrics.items()
    }


def common_control_baselines(metrics: Mapping[str, Mapping[str, float]]) -> Tuple[float, float]:
    clean_ft = [values["clean_r1"] for tag, values in metrics.items() if tag.startswith("clean_ft_s")]
    pat = [values["attacked_r1"] for tag, values in metrics.items() if tag.startswith("pat_s")]
    if not clean_ft or not pat:
        raise ValueError("Common evaluation must include clean_ft_s* and pat_s* control model tags.")
    return mean(clean_ft), mean(pat)


def rank_common_evaluation(
    runs: Sequence[Tuple[str, Path]], common_metrics: Mapping[str, Mapping[str, float]], max_clean_drop: float
) -> List[Dict[str, object]]:
    """Rank eligible pilots using the common validation evaluation and its controls."""
    clean_ft_clean_r1, pat_attacked_r1 = common_control_baselines(common_metrics)
    rows: List[Dict[str, object]] = []
    for label, run_dir in runs:
        best = best_validation_epoch(run_dir)
        if best is None:
            continue
        initial_checkpoint = selected_checkpoint_is_initial(run_dir)
        values = common_metrics.get(label)
        status = "eligible"
        if initial_checkpoint:
            status = "excluded_initial_checkpoint"
        elif values is None:
            status = "missing_common_evaluation"
        eligible = status == "eligible"
        clean_r1 = None if values is None else values["clean_r1"]
        attacked_r1 = None if values is None else values["attacked_r1"]
        accepted = meets_acceptance(clean_r1, attacked_r1, clean_ft_clean_r1, pat_attacked_r1, max_clean_drop)
        if not eligible:
            accepted = False
        rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "epoch": best["epoch"],
                "clean_r1": clean_r1,
                "attacked_r1": attacked_r1,
                "selection_score": best["selection_score"],
                "accepted": accepted,
                "eligible": eligible,
                "status": status,
                "collapse": read_collapse_metrics(run_dir, best["epoch"]),
                "clean_r1_drop_vs_clean_ft": None if clean_r1 is None else clean_ft_clean_r1 - clean_r1,
                "attacked_r1_gain_vs_pat": None if attacked_r1 is None else attacked_r1 - pat_attacked_r1,
            }
        )
    rows.sort(key=lambda row: (0 if row["eligible"] and row["accepted"] else 1 if row["eligible"] else 2, -float(row["attacked_r1"] or float("-inf"))))
    return rows


def rank_phase2_only(runs: Sequence[Tuple[str, Path]], common_metrics: Mapping[str, Mapping[str, float]]) -> List[Dict[str, object]]:
    """Rank eligible pilots only by common-validation attacked R@1."""
    rows: List[Dict[str, object]] = []
    for label, run_dir in runs:
        best = best_validation_epoch(run_dir)
        if best is None:
            continue
        initial_checkpoint = selected_checkpoint_is_initial(run_dir)
        values = common_metrics.get(label)
        status = "excluded_initial_checkpoint" if initial_checkpoint else "missing_common_evaluation" if values is None else "eligible"
        rows.append(
            {
                "label": label,
                "run_dir": str(run_dir),
                "epoch": best["epoch"],
                "clean_r1": None if values is None else values["clean_r1"],
                "attacked_r1": None if values is None else values["attacked_r1"],
                "selection_score": best["selection_score"],
                "accepted": None,
                "eligible": status == "eligible",
                "status": status,
                "collapse": read_collapse_metrics(run_dir, best["epoch"]),
                "clean_r1_drop_vs_clean_ft": None,
                "attacked_r1_gain_vs_pat": None,
            }
        )
    rows.sort(key=lambda row: (0 if row["eligible"] else 1, -float(row["attacked_r1"] or float("-inf")), -float(row["clean_r1"] or float("-inf"))))
    return rows


TABLE_COLUMNS = (
    "rank",
    "label",
    "epoch",
    "clean R@1",
    "drop vs clean-FT",
    "attacked R@1",
    "gain vs PAT",
    "knn overlap",
    "status",
    "accepted",
)


def _format(value: object, spec: str = ".2f") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return format(float(value), spec)


def format_ranking_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "|" + "|".join(["---"] * len(TABLE_COLUMNS)) + "|",
    ]
    for position, row in enumerate(rows, start=1):
        collapse = row.get("collapse")
        knn = collapse.get("knn_overlap") if isinstance(collapse, Mapping) else None
        lines.append(
            "| "
            + " | ".join(
                [
                    str(position),
                    str(row["label"]),
                    str(row["epoch"]),
                    _format(row.get("clean_r1")),
                    _format(row.get("clean_r1_drop_vs_clean_ft"), "+.2f"),
                    _format(row.get("attacked_r1")),
                    _format(row.get("attacked_r1_gain_vs_pat"), "+.2f"),
                    _format(knn, ".3f"),
                    str(row.get("status", "eligible")),
                    "unknown" if row.get("accepted") is None else _format(row.get("accepted")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="A pilot run as label=path-to-run-directory. Repeatable.",
    )
    parser.add_argument(
        "--common_eval_json",
        type=Path,
        default=None,
        help="Common rank_pgd_linf validation-split report containing the pilot checkpoints.",
    )
    parser.add_argument(
        "--phase2_only",
        action="store_true",
        help="Rank eligible pilots by common-validation attacked R@1 without Phase 1 controls.",
    )
    parser.add_argument(
        "--clean_ft_clean_r1",
        type=float,
        default=None,
        help="Legacy scalar clean-FT baseline. Prefer --common_eval_json for comparable pilot selection.",
    )
    parser.add_argument(
        "--pat_attacked_r1",
        type=float,
        default=None,
        help="Legacy scalar PAT baseline. Prefer --common_eval_json for comparable pilot selection.",
    )
    parser.add_argument("--max_clean_drop", type=float, default=DEFAULT_MAX_CLEAN_DROP)
    parser.add_argument("--output_markdown", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    runs = []
    for entry in args.run:
        if "=" not in entry:
            raise SystemExit(f"--run expects LABEL=DIR, got {entry!r}")
        label, _, directory = entry.partition("=")
        runs.append((label, Path(directory)))

    if args.phase2_only and args.common_eval_json is None:
        raise SystemExit("--phase2_only requires --common_eval_json.")
    if args.common_eval_json is None:
        rows = rank_pilot_runs(runs, args.clean_ft_clean_r1, args.pat_attacked_r1, args.max_clean_drop)
    elif args.phase2_only:
        rows = rank_phase2_only(runs, read_common_evaluation(args.common_eval_json))
    else:
        rows = rank_common_evaluation(runs, read_common_evaluation(args.common_eval_json), args.max_clean_drop)
    markdown = format_ranking_markdown(rows)
    print(markdown)

    if not rows:
        print("\nNo pilot run produced validation results yet.")
    elif args.phase2_only:
        print(f"\nWinner by common-validation attacked R@1: {rows[0]['label']} ({rows[0]['run_dir']})")
    elif rows[0].get("accepted") is None:
        print("\nAcceptance is UNKNOWN: use --common_eval_json from shared validation screening.")
    elif not rows[0]["accepted"]:
        print("\nNo configuration met the Task 2.6 criterion. Stop and reassess with the supervisor")
        print("before training the full run matrix.")
    else:
        print(f"\nWinner: {rows[0]['label']} ({rows[0]['run_dir']})")

    if args.output_markdown is not None:
        path = Path(args.output_markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown + "\n", encoding="utf-8")
    if args.output_json is not None:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
