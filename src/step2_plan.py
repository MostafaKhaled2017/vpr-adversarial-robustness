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
