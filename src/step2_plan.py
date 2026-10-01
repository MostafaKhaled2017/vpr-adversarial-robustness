"""Step 2 full training: plan every run from Step 1's result and the runs on disk.

Reads MPLC* from the finished Step 1 sweep, then lists Step 2's runs in order: 2.0 two more
MPLC* screens, so that the noise is measured at MPLC* (Step 1a measured it at a
configuration with no robustness, where every seed scored 0); 2.1 MPLC* and its clean twin,
seeds 0 and 1; 2.2 the plain_at and fare screens, then each baseline winner's full run; 2.3
the MPLC* ablations (runbook §2). Baseline winners are picked from finished screens with
Step 1's rule and the 2.0 noise, so the driver can stop and resume anywhere.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import yaml

from . import step1_sweep as sweep

CONFIG_NAME = "step2_config.yaml"
SUMMARY_DIR = "summary"

# Launcher defaults (scripts/mplc_v2_train.sh) for every setting Step 2 sets.
LAUNCHER_DEFAULTS = {
    "MPLC_V2_SEED": "0",
    "MPLC_V2_ATTACK_MIX": "all",
    "MPLC_V2_FREEZE_TE": "8",
    "MPLC_V2_LR": "1e-5",
    "MPLC_V2_ALIGN_WEIGHT": "1.0",
    "MPLC_V2_DEFENSE_LOSS": "listwise",
    "MPLC_V2_MULTI_POSITIVE": "1",
    "MPLC_V2_NUM_EPOCHS": "100",
}


def run_name(env: Mapping[str, str]) -> str:
    """The save_dir name scripts/mplc_v2_train.sh gives this run (one arm)."""
    v = {**LAUNCHER_DEFAULTS, **env}
    arm = v["MPLC_V2_ARMS"]
    name = f"mplc_v2_supervlad_{arm}"
    if arm in ("mplc", "fare") and v["MPLC_V2_ALIGN_WEIGHT"] != "1.0":
        name += f"_aw{v['MPLC_V2_ALIGN_WEIGHT']}"
    if arm == "mplc":
        if v["MPLC_V2_ATTACK_MIX"] != "all":
            name += f"_mix{v['MPLC_V2_ATTACK_MIX']}"
        if v["MPLC_V2_DEFENSE_LOSS"] != "listwise":
            name += f"_{v['MPLC_V2_DEFENSE_LOSS']}"
        if v["MPLC_V2_MULTI_POSITIVE"] != "1":
            name += "_sp"
    if v["MPLC_V2_FREEZE_TE"] != "8":
        name += f"_fte{v['MPLC_V2_FREEZE_TE']}"
    if v["MPLC_V2_LR"] != "1e-5":
        name += f"_lr{v['MPLC_V2_LR']}"
    if v["MPLC_V2_NUM_EPOCHS"] != "100":
        name += f"_ep{v['MPLC_V2_NUM_EPOCHS']}"
    return f"{name}_s{v['MPLC_V2_SEED']}"


@dataclass
class Step1:
    star: Dict[str, str]  # MPLC*'s launcher settings: attack mix, freeze_te, lr, align weight
    noise: float  # Step 1a's noise, for reference; Step 2 decides with the 2.0 noise
    epochs: int  # screen length and validation schedule, from sweep_config.yaml
    val_every: int
    budget: float
    star_screen: sweep.ScreenResult  # MPLC*'s seed-0 Step 1 screen: first score of the 2.0 noise


def load_step1(root: Path, batch_size: int, batches_per_epoch: int) -> Step1:
    """MPLC* and its screen from a finished Step 1 sweep; SystemExit with the reason otherwise."""
    path = root / sweep.CONFIG_NAME
    if not path.is_file():
        raise SystemExit(f"{path} is missing: run scripts/mplc_v2_step1.sh first")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (config["batch_size"], config["batches_per_epoch"]) != (batch_size, batches_per_epoch):
        raise SystemExit(
            f"Step 1 ran at batch size {config['batch_size']} x {config['batches_per_epoch']} batches, "
            f"lib/supervlad_common.sh now sets {batch_size} x {batches_per_epoch}"
        )
    # accept_noise only skips the noise stop: an unaccepted high-noise sweep never reached 1b-1e.
    results = sweep._results_reader(root, int(config["epochs"]), float(config["budget"]))
    state = sweep.plan_sweep(results, accept_noise=True)
    if state.final is None:
        raise SystemExit(f"Step 1 in {root} is not finished ({state.stop_reason or 'screens pending'})")
    star = {sweep.ENV_NAMES[key]: state.final[key] for key in ("mix", "fte", "lr", "aw")}
    return Step1(star, state.noise, int(config["epochs"]), int(config["val_every"]), float(config["budget"]),
                 results(state.final))


# Full-length runs (2.1, 2.2 full, 2.3) validate every 3 epochs; patience counts validations.
FULL = {"MPLC_V2_VAL_EVERY": "3", "MPLC_V2_PATIENCE": "6", "MPLC_V2_LR_PLATEAU_PATIENCE": "3"}
NOISE_SEEDS = ("1", "2")  # seed 0 is MPLC*'s Step 1 screen
MPLC_SEEDS = ("0", "1")
BASELINE_LRS = ("1e-5", "3e-6")  # preferred first
FARE_ANCHORS = ("1.0", "10")  # preferred first
FARE_EDGES = {"10": "30", "1.0": "0.1"}  # winning grid-edge λ -> the one extra screen
ABLATIONS = (
    ("no perceptual attacks", "MPLC_V2_ATTACK_MIX", "linf"),
    ("no anchor", "MPLC_V2_ALIGN_WEIGHT", "0"),
    ("hinge loss", "MPLC_V2_DEFENSE_LOSS", "hinge"),
    ("single positive", "MPLC_V2_MULTI_POSITIVE", "0"),
)


@dataclass
class Run:
    stage: str  # 2.0, 2.1, 2.2 screen, 2.2 full, 2.3
    label: str
    env: Dict[str, str]
    preferred: Optional[Dict[str, str]] = None  # baseline screens: their arm's preferred screen


@dataclass
class Plan:
    runs: List[Run]  # every run known so far, in execution order
    decisions: List[str]
    noise: Optional[float] = None  # 2.0 noise, once its screens are finished
    stop_reason: str = ""  # reported once no run is pending


def _winner(grid, preferred, results, noise):
    """Step 1's rule; None while a screen is unfinished or when every screen is a loss."""
    if any(results(env) is None for env in grid):
        return None
    return sweep.choose(grid, preferred, results, noise)


def _label(env: Mapping[str, str]) -> str:
    arm = env["MPLC_V2_ARMS"]
    aw = f" λ {env['MPLC_V2_ALIGN_WEIGHT']}" if arm == "fare" else ""
    return f"{arm}{aw} lr {env['MPLC_V2_LR']}"


def plan_step2(
    step1: Step1,
    results: Callable[[Mapping[str, str]], Optional[sweep.ScreenResult]],
    baseline_seeds: Sequence[str] = ("0",),
) -> Plan:
    """Every Step 2 run known so far, in order, and the decisions made so far."""
    plan = Plan(runs=[], decisions=[])
    screen = {"MPLC_V2_NUM_EPOCHS": str(step1.epochs), "MPLC_V2_VAL_EVERY": str(step1.val_every)}

    # 2.0 noise at MPLC*: MPLC*'s Step 1 screen is seed 0; two more seeds at the same length.
    noise_envs = [{**step1.star, **screen, "MPLC_V2_ARMS": "mplc", "MPLC_V2_SEED": s} for s in NOISE_SEEDS]
    plan.runs += [Run("2.0", f"MPLC* screen s{e['MPLC_V2_SEED']}", e) for e in noise_envs]
    scores = [step1.star_screen] + [results(e) for e in noise_envs]
    if any(r is None for r in scores):
        plan.decisions.append("2.0 noise: screens pending")
    elif any(r.is_loss for r in scores):
        plan.stop_reason = "an MPLC* noise screen (2.0) is a loss, so the noise is undefined and no baseline was picked"
        plan.decisions.append(f"2.0 noise: undefined ({plan.stop_reason})")
    else:
        plan.noise = max(r.robust for r in scores) - min(r.robust for r in scores)
        plan.decisions.append(
            f"2.0 noise = {plan.noise:.2f} (MPLC* screens, seeds 0-2: {', '.join(f'{r.robust:.2f}' for r in scores)};"
            f" Step 1a noise was {step1.noise:.2f})"
        )

    # 2.1 MPLC* and its clean twin (the twin reads only freeze_te, lr and seed).
    for seed in MPLC_SEEDS:
        plan.runs.append(Run("2.1", f"clean twin s{seed}", {**step1.star, **FULL, "MPLC_V2_ARMS": "clean_ft", "MPLC_V2_SEED": seed}))
        plan.runs.append(Run("2.1", f"MPLC* s{seed}", {**step1.star, **FULL, "MPLC_V2_ARMS": "mplc", "MPLC_V2_SEED": seed}))

    # 2.2 baselines: screens at Step 1's length, then each winner at full length. Only N* is
    # taken from MPLC*; lr and λ are the baseline's own.
    def env(arm, lr, aw="1.0"):
        return {**screen, "MPLC_V2_FREEZE_TE": step1.star["MPLC_V2_FREEZE_TE"],
                "MPLC_V2_ARMS": arm, "MPLC_V2_LR": lr, "MPLC_V2_ALIGN_WEIGHT": aw}

    grids = {
        "plain_at": [env("plain_at", lr) for lr in BASELINE_LRS],
        "fare": [env("fare", lr, aw) for aw in FARE_ANCHORS for lr in BASELINE_LRS],
    }
    full_runs = []
    for arm, grid in grids.items():
        preferred = grid[0]
        winner = None if plan.noise is None else _winner(grid, preferred, results, plan.noise)
        edge = FARE_EDGES.get(winner["MPLC_V2_ALIGN_WEIGHT"]) if arm == "fare" and winner else None
        if edge:
            grid.append(env("fare", winner["MPLC_V2_LR"], edge))
            winner = _winner(grid, preferred, results, plan.noise)
        plan.runs += [Run("2.2 screen", _label(e), e, preferred) for e in grid]
        if winner is None:
            if plan.noise is None:
                reason = "winner waits for the 2.0 noise"
            elif any(results(e) is None for e in grid):
                reason = "screens pending"
            else:
                reason = "every screen is a loss; no full run"
            plan.decisions.append(f"{arm}: {reason}")
            continue
        note = f" (grid edge: added λ {edge})" if edge else ""
        plan.decisions.append(f"{arm}: winner {_label(winner)}{note}")
        full = {**{k: v for k, v in winner.items() if k not in screen}, **FULL}
        full_runs += [Run("2.2 full", f"{_label(full)} s{seed}", {**full, "MPLC_V2_SEED": seed}) for seed in baseline_seeds]
    plan.runs += full_runs

    # 2.3 ablations: MPLC* with one component removed, unless MPLC* already removed it.
    for label, key, value in ABLATIONS:
        if step1.star.get(key, LAUNCHER_DEFAULTS[key]) == value:
            plan.decisions.append(f"ablation {label}: covered by MPLC* itself ({key}={value})")
            continue
        plan.runs.append(Run("2.3", f"ablation: {label}",
                             {**step1.star, **FULL, "MPLC_V2_ARMS": "mplc", "MPLC_V2_SEED": "0", key: value}))
    return plan


def ensure_config(root: Path, step1: Step1, step1_root: Path, freeze: bool = True) -> None:
    """Record the Step 1 result Step 2 is planned from; refuse a later call from another one."""
    path = root / CONFIG_NAME
    wanted = {"mplc_star": step1.star, "mplc_star_screen_robust": step1.star_screen.robust,
              "step1_noise": step1.noise, "screen_epochs": step1.epochs,
              "screen_val_every": step1.val_every, "budget": step1.budget}
    if path.is_file():
        stored = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        stored = {key: stored.get(key) for key in wanted}
        if stored != wanted:
            raise SystemExit(
                f"{path} was planned from another Step 1 result (stored {stored}, now {wanted}); "
                "restore Step 1 or use a new STEP2_ROOT"
            )
        return
    if not freeze:
        return
    root.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    content = dict(wanted, step1_root=str(step1_root), git_revision=revision,
                   created=datetime.now().isoformat(timespec="seconds"))
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def results_reader(root: Path, budget: float):
    return lambda env: sweep.read_screen(root / run_name(env), budget)


TABLE_FIELDS = [
    "stage", "run", "state", "epochs_run", "initial_clean_r1", "epoch", "clean_r1", "clean_drop",
    "robust_score", "delta_vs_preferred", "noise_label", "hours", "run_dir",
]


def table_rows(plan: Plan, results, root: Path) -> List[Dict[str, object]]:
    rows = []
    for run in plan.runs:
        r = results(run.env)
        reference = results(run.preferred) if run.preferred else None
        delta = None
        if r is not None and not r.is_loss and reference is not None and not reference.is_loss:
            delta = r.robust - reference.robust
        rows.append({
            "stage": run.stage,
            "run": run.label,
            "state": "pending" if r is None else r.state,
            "epochs_run": "" if r is None else r.epochs_run,
            "initial_clean_r1": sweep._fmt(r and r.initial_clean),
            "epoch": "" if r is None or r.epoch is None else r.epoch,
            "clean_r1": sweep._fmt(r and r.clean),
            "clean_drop": sweep._fmt(
                None if r is None or r.clean is None or r.initial_clean is None else r.initial_clean - r.clean
            ),
            "robust_score": sweep._fmt(r and r.robust),
            "delta_vs_preferred": sweep._fmt(delta, "+.2f"),
            "noise_label": (
                "" if run.preferred is None
                else "reference" if run.env == run.preferred
                else "loss" if r is not None and r.is_loss
                else "" if plan.noise is None
                else sweep.label(delta, plan.noise)
            ),
            "hours": sweep._fmt(r and r.hours, ".1f"),
            "run_dir": str(root / run_name(run.env)),
        })
    return rows


def write_summary(plan: Plan, rows, out: Path, step1: Step1) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    star = " ".join(f"{k}={v}" for k, v in step1.star.items())
    shown = TABLE_FIELDS[:-1]
    lines = [
        "# Step 2 runs", "",
        f"- MPLC\\*: `{star}`",
        f"- Score: robust score of the budget-{step1.budget:g} checkpoint.",
        *[f"- {decision}" for decision in plan.decisions],
        *([f"- **Stopped:** {plan.stop_reason}."] if plan.stop_reason else []),
        "", "| " + " | ".join(shown) + " |", "|" + "---|" * len(shown),
        *["| " + " | ".join(str(row[f]) for f in shown) + " |" for row in rows],
    ]
    (out / "runs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def env_assignments(env: Mapping[str, str]) -> str:
    return " ".join(f"{key}={value}" for key, value in env.items())


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("next", "summarize"))
    parser.add_argument("--root", type=Path, required=True, help="Step 2 run directory (STEP2_ROOT).")
    parser.add_argument("--step1-root", type=Path, required=True, help="Finished Step 1 sweep (STEP1_ROOT).")
    parser.add_argument("--batch-size", type=int, required=True, help="SUPERVLAD_TRAIN_BATCH_SIZE.")
    parser.add_argument("--batches-per-epoch", type=int, required=True, help="SUPERVLAD_BATCHES_PER_EPOCH.")
    parser.add_argument("--baseline-seeds", nargs="+", default=["0"], help="Seeds of the baselines' full runs.")
    parser.add_argument("--no-freeze", action="store_true", help="Do not write step2_config.yaml.")
    args = parser.parse_args(argv)

    try:
        step1 = load_step1(args.step1_root, args.batch_size, args.batches_per_epoch)
        ensure_config(args.root, step1, args.step1_root, freeze=not args.no_freeze)
    except SystemExit as stop:
        if args.command != "next":
            raise
        print(f"STOP {stop}")
        return
    results = results_reader(args.root, step1.budget)
    plan = plan_step2(step1, results, args.baseline_seeds)
    write_summary(plan, table_rows(plan, results, args.root), args.root / SUMMARY_DIR, step1)
    if args.command == "summarize":
        print(f"Tables written to {args.root / SUMMARY_DIR}")
        return
    pending = [run for run in plan.runs if results(run.env) is None]
    for run in pending:
        print(env_assignments(run.env))
    if not pending:
        print(f"STOP {plan.stop_reason}" if plan.stop_reason else "DONE")


if __name__ == "__main__":
    main()
