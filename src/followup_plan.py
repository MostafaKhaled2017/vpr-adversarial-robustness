"""Follow-up training after Step 2: seven full-length runs in a fixed order.

No stage logic: every run is fixed, and a run is done once its run_status.json records a
terminal state. Each run is one arm of scripts/mplc_v2_train.sh at Step 2's full-run settings
(validate every 3 epochs, patience 6, lr-plateau patience 3, up to 100 epochs) with
freeze_te 0 and lr 1e-5. The single-positive MPLC runs (sp_*) train on rank L-inf attacks
only, against the hardest positive. Keys, in default order:

  sp_s1          single-positive MPLC, seed 1
  sp_hinge       single-positive MPLC with the hinge loss
  sp_aw2         single-positive MPLC, lambda_anchor 2
  fare_eps2      fare at training eps 0.0342 (<= 2/255) instead of 0.0685
  plain_at_eps2  plain_at at training eps 0.0342
  sp_eps2        single-positive MPLC at training eps 0.0342
  sp_aw3         single-positive MPLC, lambda_anchor 3

Every run validates at eps 0.0342 and 0.0685 whatever its training eps, so robust scores
compare across runs. The driver (scripts/mplc_v2_followup_train.sh) also passes
MPLC_V2_CLEAN_BUDGETS (CLEAN_BUDGETS below unless FOLLOWUP_CLEAN_BUDGETS is set): its value
holds spaces, so it is not one of the VAR=value words printed here.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from . import step1_sweep as sweep
from . import step2_plan as step2
from .phase2_run_group import TERMINAL_STATES

SUMMARY_DIR = "summary"
CLEAN_BUDGETS = "1 3 5 7 10"  # default MPLC_V2_CLEAN_BUDGETS of every run
BUDGET = 5.0  # the budget-5 checkpoint of the tables
VAL_EPSILONS = ("0.0342", "0.0685")  # the launcher's --val_rank_epsilons, the same for every run
DEFAULT_TRAIN_EPS = "0.0685"  # the launcher's MPLC_V2_TRAIN_EPS default (not named in the run)
EPS2 = "0.0342"

SHARED = {"MPLC_V2_FREEZE_TE": "0", "MPLC_V2_LR": "1e-5", "MPLC_V2_NUM_EPOCHS": "100", **step2.FULL}
SINGLE_POSITIVE = {"MPLC_V2_ARMS": "mplc", "MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_MULTI_POSITIVE": "0"}

# Key -> launcher settings, in default order. MPLC_V2_ARMS is always one arm: the launcher's
# default would also train a clean twin.
RUNS: Dict[str, Dict[str, str]] = {
    "sp_s1": {**SHARED, **SINGLE_POSITIVE, "MPLC_V2_SEED": "1"},
    "sp_hinge": {**SHARED, **SINGLE_POSITIVE, "MPLC_V2_DEFENSE_LOSS": "hinge", "MPLC_V2_SEED": "0"},
    "sp_aw2": {**SHARED, **SINGLE_POSITIVE, "MPLC_V2_ALIGN_WEIGHT": "2", "MPLC_V2_SEED": "0"},
    "fare_eps2": {**SHARED, "MPLC_V2_ARMS": "fare", "MPLC_V2_TRAIN_EPS": EPS2, "MPLC_V2_ALIGN_WEIGHT": "1.0",
                  "MPLC_V2_SEED": "0"},
    "plain_at_eps2": {**SHARED, "MPLC_V2_ARMS": "plain_at", "MPLC_V2_TRAIN_EPS": EPS2, "MPLC_V2_SEED": "0"},
    "sp_eps2": {**SHARED, **SINGLE_POSITIVE, "MPLC_V2_TRAIN_EPS": EPS2, "MPLC_V2_SEED": "0"},
    "sp_aw3": {**SHARED, **SINGLE_POSITIVE, "MPLC_V2_ALIGN_WEIGHT": "3", "MPLC_V2_SEED": "0"},
}


def run_name(env: Mapping[str, str]) -> str:
    """The save_dir name scripts/mplc_v2_train.sh gives this run: step2_plan.run_name plus the
    training-eps suffix, which Step 2 never set."""
    eps = env.get("MPLC_V2_TRAIN_EPS", DEFAULT_TRAIN_EPS)
    rest = {key: value for key, value in env.items() if key != "MPLC_V2_TRAIN_EPS"}
    name = step2.run_name(rest)
    if eps == DEFAULT_TRAIN_EPS or rest["MPLC_V2_ARMS"] == "clean_ft":
        return name
    # The launcher names eps right after the arm and its lambda_anchor, before every other setting.
    head = step2.run_name({key: rest[key] for key in ("MPLC_V2_ARMS", "MPLC_V2_ALIGN_WEIGHT") if key in rest})
    head = head[: -len("_s0")]
    return f"{head}_eps{eps}{name[len(head):]}"


def run_state(run_dir: Path) -> str:
    """pending (no directory), the state in run_status.json, or started (no status yet)."""
    if not run_dir.is_dir():
        return "pending"
    status = run_dir / "run_status.json"
    if not status.is_file():
        return "started"
    return str(json.loads(status.read_text(encoding="utf-8")).get("state", "") or "started")


def is_finished(run_dir: Path) -> bool:
    return run_state(run_dir) in TERMINAL_STATES


def best_record(records: Sequence[dict]) -> Optional[dict]:
    """The unconstrained best: the trained epoch with the highest robust score (ties: higher clean)."""
    trained = [r for r in records if int(r["epoch"]) >= 0]
    return max(trained, key=lambda r: (float(r["robust_score"]), float(r["clean"]["R@1"])), default=None)


def _attacked_r1(record: Mapping, eps: str) -> Optional[float]:
    value = record.get("attacks", {}).get(f"rank_pgd_linf_eps_{eps}", {}).get("R@1")
    return None if value is None else float(value)


CHECKPOINT_FIELDS = ["epoch", "clean_r1", "clean_drop", *[f"r1_eps{eps}" for eps in VAL_EPSILONS], "robust_score"]
CHECKPOINTS = ("b5", "best")
TABLE_FIELDS = [
    "key", "run", "state", "epochs_run", "hours", "initial_clean_r1",
    *[f"{checkpoint}_{field}" for checkpoint in CHECKPOINTS for field in CHECKPOINT_FIELDS],
    "run_dir",
]


def _checkpoint_columns(prefix: str, record: Optional[Mapping], initial_clean: Optional[float]) -> Dict[str, object]:
    if record is None:
        return {f"{prefix}_{field}": "" for field in CHECKPOINT_FIELDS}
    clean = float(record["clean"]["R@1"])
    return {
        f"{prefix}_epoch": int(record["epoch"]),
        f"{prefix}_clean_r1": sweep._fmt(clean),
        f"{prefix}_clean_drop": sweep._fmt(None if initial_clean is None else initial_clean - clean),
        **{f"{prefix}_r1_eps{eps}": sweep._fmt(_attacked_r1(record, eps)) for eps in VAL_EPSILONS},
        f"{prefix}_robust_score": sweep._fmt(float(record["robust_score"])),
    }


def table_row(key: str, root: Path) -> Dict[str, object]:
    """One run's row: its state, and for a finished run its budget-5 and best checkpoints."""
    name = run_name(RUNS[key])
    run_dir = root / name
    records = sweep._read_records(run_dir)
    row: Dict[str, object] = {
        "key": key,
        "run": name,
        "state": run_state(run_dir),
        "epochs_run": max((int(r["epoch"]) for r in records if int(r["epoch"]) >= 0), default=""),
        "hours": sweep._fmt(sweep._wall_hours(run_dir), ".1f"),
        "initial_clean_r1": "",
        **_checkpoint_columns("b5", None, None),
        **_checkpoint_columns("best", None, None),
        "run_dir": str(run_dir),
    }
    result = sweep.read_screen(run_dir, BUDGET)
    if result is None:  # not finished: its checkpoints may still change
        return row
    by_epoch = {int(r["epoch"]): r for r in records}
    row["epochs_run"] = result.epochs_run
    row["initial_clean_r1"] = sweep._fmt(result.initial_clean)
    if result.epoch is None:
        row["b5_epoch"] = "initial"  # no trained epoch stayed within the budget
    else:
        row.update(_checkpoint_columns("b5", by_epoch[result.epoch], result.initial_clean))
    row.update(_checkpoint_columns("best", best_record(records), result.initial_clean))
    return row


def write_summary(rows: Sequence[Mapping[str, object]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    shown = TABLE_FIELDS[:-1]
    lines = [
        "# Follow-up training runs", "",
        f"- b5: the budget-{BUDGET:g} checkpoint, the most robust trained epoch within {BUDGET:g} clean R@1 "
        "points of the initial model (`initial`: no trained epoch stayed within the budget).",
        "- best: the most robust trained epoch, with no clean budget (ties: higher clean R@1).",
        f"- Validation R@1 under rank-PGD L-inf at eps {' and '.join(VAL_EPSILONS)} (<= 2/255 and 4/255) for every "
        "run, whatever its training eps; robust score = their mean.",
        "- Checkpoint columns are filled once a run has finished.",
        "", "| " + " | ".join(shown) + " |", "|" + "---|" * len(shown),
        *["| " + " | ".join(str(row[f]) for f in shown) + " |" for row in rows],
    ]
    (out / "runs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runs_help = "Run keys, in order (default: all, in the default order)."

    def add_runs(sub):
        sub.add_argument("--runs", nargs="+", choices=list(RUNS), default=list(RUNS), metavar="KEY", help=runs_help)

    listing = commands.add_parser("list", help="Print each run's key, name and launcher settings.")
    add_runs(listing)
    following = commands.add_parser("next", help="Print the settings of the first unfinished run, or DONE.")
    following.add_argument("--root", type=Path, required=True, help="Follow-up run directory (FOLLOWUP_ROOT).")
    following.add_argument("--all", action="store_true", help="Print every unfinished run, one per line.")
    add_runs(following)
    naming = commands.add_parser("name", help="Print a run's directory name.")
    naming.add_argument("key", choices=list(RUNS), metavar="KEY")
    summary = commands.add_parser("summary", help="Write ROOT/summary/runs.md and runs.csv.")
    summary.add_argument("--root", type=Path, required=True, help="Follow-up run directory (FOLLOWUP_ROOT).")
    add_runs(summary)
    args = parser.parse_args(argv)

    if args.command == "name":
        print(run_name(RUNS[args.key]))
        return
    keys = list(dict.fromkeys(args.runs))
    if args.command == "list":
        for key in keys:
            print(f"{key}  {run_name(RUNS[key])}")
            print(f"    {step2.env_assignments(RUNS[key])}")
        print(f'All runs: MPLC_V2_CLEAN_BUDGETS="{CLEAN_BUDGETS}" (driver: FOLLOWUP_CLEAN_BUDGETS)')
        return
    if args.command == "summary":
        write_summary([table_row(key, args.root) for key in keys], args.root / SUMMARY_DIR)
        print(f"Tables written to {args.root / SUMMARY_DIR}")
        return
    pending = [key for key in keys if not is_finished(args.root / run_name(RUNS[key]))]
    for key in pending if args.all else pending[:1]:
        print(step2.env_assignments(RUNS[key]))
    if not pending:
        print("DONE")


if __name__ == "__main__":
    main()
