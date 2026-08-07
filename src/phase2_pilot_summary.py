"""Rank Phase 2 pilot configurations against the Phase 1 baselines (Task 2.6).

The pilot grid trains one run per hyper-parameter combination and then has to pick a
winner. The acceptance criterion is stated relative to Phase 1's arms, not in absolute
terms: clean R@1 must stay within a small drop of the *clean-FT* control, and attacked R@1
must be at least the *PAT* re-run's. Neither number exists until Phase 1 has run, which is
why Phase 1 is a gate on Phase 2.

Selection uses each run's ``validation_recalls.csv``, the same validation signal that chose
the checkpoint, so ranking costs nothing beyond the training already done. The winner then
goes through the full Phase 0 evaluation grid for the reported numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
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
    """Clean and attacked R@1 at the epoch with the highest selection score."""
    csv_path = Path(run_dir) / "validation_recalls.csv"
    if not csv_path.is_file():
        return None

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None

    clean_rows = [row for row in rows if row.get("split") == "clean"]
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

        accepted = meets_acceptance(
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
        return (0 if row.get("accepted") else 1, -float(robustness))

    rows.sort(key=sort_key)
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
        "--clean_ft_clean_r1",
        type=float,
        default=None,
        help="Phase 1 clean-FT clean R@1 baseline. Without it acceptance is reported as unknown.",
    )
    parser.add_argument(
        "--pat_attacked_r1",
        type=float,
        default=None,
        help="Phase 1 PAT attacked R@1 baseline. Without it acceptance is reported as unknown.",
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

    rows = rank_pilot_runs(runs, args.clean_ft_clean_r1, args.pat_attacked_r1, args.max_clean_drop)
    markdown = format_ranking_markdown(rows)
    print(markdown)

    if not rows:
        print("\nNo pilot run produced validation results yet.")
    elif rows[0].get("accepted") is None:
        print("\nAcceptance is UNKNOWN: pass --clean_ft_clean_r1 and --pat_attacked_r1 from Phase 1.")
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
