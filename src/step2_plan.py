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
        plan.runs.append(Run("2.1", f"clean twin s{seed}", {**step1.star, "MPLC_V2_ARMS": "clean_ft", "MPLC_V2_SEED": seed}))
        plan.runs.append(Run("2.1", f"MPLC* s{seed}", {**step1.star, "MPLC_V2_ARMS": "mplc", "MPLC_V2_SEED": seed}))

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
        full = {k: v for k, v in winner.items() if k not in screen}
        full_runs += [Run("2.2 full", f"{_label(full)} s{seed}", {**full, "MPLC_V2_SEED": seed}) for seed in baseline_seeds]
    plan.runs += full_runs

    # 2.3 ablations: MPLC* with one component removed, unless MPLC* already removed it.
    for label, key, value in ABLATIONS:
        if step1.star.get(key, LAUNCHER_DEFAULTS[key]) == value:
            plan.decisions.append(f"ablation {label}: covered by MPLC* itself ({key}={value})")
            continue
        plan.runs.append(Run("2.3", f"ablation: {label}",
                             {**step1.star, "MPLC_V2_ARMS": "mplc", "MPLC_V2_SEED": "0", key: value}))
    return plan
