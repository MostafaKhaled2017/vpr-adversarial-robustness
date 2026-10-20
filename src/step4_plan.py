"""Step 4 (runbook §4): ARAFT and its baselines on a second architecture, BoQ or MixVPR.

Each run is one arm of scripts/mplc_v2_train.sh with MPLC_V2_MODEL set, at the follow-up runs'
settings (validate every 3 epochs, patience 6, lr-plateau patience 3, up to 100 epochs, lr
1e-5; freeze_te 0 for BoQ, the whole network for MixVPR). Keys, in order, the same for both
models:

  araft          ARAFT: single-positive MPLC on rank L-inf attacks only   (default)
  plain_at       plain_at baseline                                         (default)
  fare           FARE baseline, lambda_anchor 1.0                          (default)
  twin           clean twin                                                (default)
  fare_eps2      FARE at training eps 0.0342 (<= 2/255)                    (optional)
  screen_lr1e-5  9-epoch ARAFT lr screen, validated every epoch            (optional)
  screen_lr3e-6  the same at lr 3e-6                                       (optional)

The driver (scripts/mplc_v2_step4.sh) also passes MPLC_V2_CLEAN_BUDGETS, as the follow-up does.
`models` prints the evaluation checkpoints of the trained rows, chosen as in the follow-up
evaluation (src/followup_eval_plan.py, choose): Table A the budget-5 checkpoint, Table B the
most robust one, one file when they are the same epoch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

from . import followup_eval_plan as followup_eval
from . import followup_plan as followup
from . import step2_plan as step2
from .step3_plan import PRETRAINED_PATH, Refused

MODELS = ("boq", "mixvpr")
DEFAULT_RUNS = ("araft", "plain_at", "fare", "twin")
EVAL_ROWS = ("araft", "plain_at", "fare", "fare_eps2")  # the main table's rows, after the pretrained model
TWIN = "twin"


def runs(model: str) -> Dict[str, Dict[str, str]]:
    """Key -> launcher settings, in order. MixVPR has no DINOv2 blocks: no MPLC_V2_FREEZE_TE."""
    shared = {"MPLC_V2_MODEL": model, **followup.SHARED}
    screen = {"MPLC_V2_MODEL": model, "MPLC_V2_FREEZE_TE": "0", "MPLC_V2_NUM_EPOCHS": "9"}
    if model == "mixvpr":
        del shared["MPLC_V2_FREEZE_TE"], screen["MPLC_V2_FREEZE_TE"]
    fare = {**shared, "MPLC_V2_ARMS": "fare", "MPLC_V2_ALIGN_WEIGHT": "1.0", "MPLC_V2_SEED": "0"}
    araft_screen = {**screen, **followup.SINGLE_POSITIVE, "MPLC_V2_SEED": "0"}
    return {
        "araft": {**shared, **followup.SINGLE_POSITIVE, "MPLC_V2_SEED": "0"},
        "plain_at": {**shared, "MPLC_V2_ARMS": "plain_at", "MPLC_V2_SEED": "0"},
        "fare": fare,
        "twin": {**shared, "MPLC_V2_ARMS": "clean_ft", "MPLC_V2_SEED": "0"},
        "fare_eps2": {**fare, "MPLC_V2_TRAIN_EPS": followup.EPS2},
        "screen_lr1e-5": {**araft_screen, "MPLC_V2_LR": "1e-5"},
        "screen_lr3e-6": {**araft_screen, "MPLC_V2_LR": "3e-6"},
    }


def run_name(env: Mapping[str, str]) -> str:
    """The launcher's save_dir: the follow-up name with mplc_v2_<model>_ in place of mplc_v2_supervlad_.
    Without MPLC_V2_FREEZE_TE (MixVPR) the follow-up name has no _fte part, as the launcher's."""
    model = env["MPLC_V2_MODEL"]
    name = followup.run_name({key: value for key, value in env.items() if key != "MPLC_V2_MODEL"})
    return f"mplc_v2_{model}_{name[len('mplc_v2_supervlad_'):]}"


def eval_models(model: str, root: Path, pretrained: str) -> list:
    """`GROUP TAG=PATH` lines (main, then twin) for the trained rows, then `pending KEY` for the
    default rows not trained yet. A row's tag is its key, plus _b5 / _best when Tables A and B
    use different files; a table whose checkpoint is the pretrained model adds nothing."""
    table = runs(model)
    lines, pending = [f"main pretrained={pretrained}"], []
    for key in (*EVAL_ROWS, TWIN):
        name = run_name(table[key])
        entry = followup_eval.Entry(key, key, "clean" if key == TWIN else "main", "step4", name)
        choice = followup_eval.choose(entry, root / name)
        if choice is None:
            if key in DEFAULT_RUNS or (root / name).is_dir():
                pending.append(key)
            continue
        for checkpoint in choice.checkpoints:
            if checkpoint.path == PRETRAINED_PATH:
                continue
            tag = key if checkpoint.tables == "AB" else f"{key}_{'b5' if checkpoint.tables == 'A' else 'best'}"
            lines.append(f"{'twin' if key == TWIN else 'main'} {tag}={checkpoint.path}")
    return lines + [f"pending {key}" for key in pending]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    keys = list(runs("boq"))

    def add(name, help_text, root=True, run_keys=True):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--model", choices=MODELS, required=True)
        if root:
            sub.add_argument("--root", type=Path, required=True, help="Step 4 run directory (STEP4_ROOT).")
        if run_keys:
            sub.add_argument("--runs", nargs="+", choices=keys, default=None, metavar="KEY",
                             help=f"Run keys, in order (default: {' '.join(DEFAULT_RUNS)}).")
        return sub

    add("list", "Print each run's key, name and launcher settings.", root=False)
    add("next", "Print the settings of the first unfinished run, or DONE.").add_argument(
        "--all", action="store_true", help="Print every unfinished run, one per line.")
    add("name", "Print a run's directory name.", root=False, run_keys=False).add_argument(
        "key", choices=keys, metavar="KEY")
    add("summary", "Write ROOT/summary/runs.md and runs.csv (default: every key).")
    add("models", "Print the evaluation checkpoints.", run_keys=False).add_argument(
        "--pretrained", required=True, help="The pretrained release's checkpoint.")
    args = parser.parse_args(argv)

    table = runs(args.model)
    if args.command == "name":
        print(run_name(table[args.key]))
        return
    if args.command == "models":
        try:
            print("\n".join(eval_models(args.model, args.root, args.pretrained)))
        except Refused as refused:
            print(refused, file=sys.stderr)
            sys.exit(refused.code)
        return
    selected = list(dict.fromkeys(args.runs or (keys if args.command == "summary" else DEFAULT_RUNS)))
    if args.command == "list":
        for key in selected:
            print(f"{key}  {run_name(table[key])}")
            print(f"    {step2.env_assignments(table[key])}")
        print(f'All runs: MPLC_V2_CLEAN_BUDGETS="{followup.CLEAN_BUDGETS}" (driver: STEP4_CLEAN_BUDGETS)')
        return
    if args.command == "summary":
        rows = [followup.table_row(key, args.root, runs=table, naming=run_name) for key in selected]
        followup.write_summary(rows, args.root / followup.SUMMARY_DIR, title=f"Step 4 training runs ({args.model})")
        print(f"Tables written to {args.root / followup.SUMMARY_DIR}")
        return
    pending = [key for key in selected if not followup.is_finished(args.root / run_name(table[key]))]
    for key in pending if args.all else pending[:1]:
        print(step2.env_assignments(table[key]))
    if not pending:
        print("DONE")


if __name__ == "__main__":
    main()
