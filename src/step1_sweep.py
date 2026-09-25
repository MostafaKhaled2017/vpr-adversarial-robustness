"""Step 1 short-run sweep: decide each stage from finished screens, write tables and figures.

Every decision is recomputed from the screens on disk, so the sweep can stop and resume at
any point and always reaches the same result (runbook §1). Stages, in order:

1a noise   default configuration, seeds 0-2; noise = max - min of the three scores
1b mix     attack mix all vs linf
1c depth   freeze_te x lr, from the 1b winner
1d anchor  lambda_anchor, from the 1c winner
1e recheck the losing mix under the final settings

A screen's score is the robust score of its budget-3 checkpoint. Each stage keeps its
preferred option unless another option beats it by more than the noise.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import yaml

from .phase2_run_group import TERMINAL_STATES

CONFIG_NAME = "sweep_config.yaml"
SUMMARY_DIR = "summary"
NOISE_LIMIT = 3.0

# Launcher defaults (scripts/mplc_v2_train.sh). Values are strings because the launcher
# names runs by string comparison.
DEFAULTS = {"seed": "0", "mix": "all", "fte": "8", "lr": "1e-5", "aw": "1.0"}
ENV_NAMES = {
    "seed": "MPLC_V2_SEED",
    "mix": "MPLC_V2_ATTACK_MIX",
    "fte": "MPLC_V2_FREEZE_TE",
    "lr": "MPLC_V2_LR",
    "aw": "MPLC_V2_ALIGN_WEIGHT",
}
NOISE_SEEDS = ("0", "1", "2")
MIXES = ("all", "linf")
DEPTHS = ("8", "4", "0")  # preference order: shallower fine-tuning first
LRS = ("1e-5", "3e-6")  # preference order: the recipe's lr first
LR_EXTENSION = "1e-6"
ANCHORS = ("0", "0.1", "1.0", "10")
ANCHOR_EXTENSION = "30"


def sweep_settings(epochs: int, budget: float, batch_size: int) -> Dict[str, object]:
    """Everything that must stay fixed across a sweep's lifetime."""
    return {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "budget": float(budget),
        "noise_seeds": list(NOISE_SEEDS),
        "mixes": list(MIXES),
        "depths": list(DEPTHS),
        "lrs": list(LRS),
        "lr_extension": LR_EXTENSION,
        "anchors": list(ANCHORS),
        "anchor_extension": ANCHOR_EXTENSION,
    }


# --------------------------------------------------------------------------- configs


def config(**overrides: str) -> Dict[str, str]:
    values = dict(DEFAULTS)
    values.update({key: str(value) for key, value in overrides.items()})
    return values


def run_name(cfg: Mapping[str, str], epochs: int) -> str:
    """The save_dir name scripts/mplc_v2_train.sh gives this configuration (mplc arm)."""
    name = "mplc_v2_supervlad_mplc"
    if cfg["aw"] != DEFAULTS["aw"]:
        name += f"_aw{cfg['aw']}"
    if cfg["mix"] != DEFAULTS["mix"]:
        name += f"_mix{cfg['mix']}"
    if cfg["fte"] != DEFAULTS["fte"]:
        name += f"_fte{cfg['fte']}"
    if cfg["lr"] != DEFAULTS["lr"]:
        name += f"_lr{cfg['lr']}"
    if int(epochs) != 100:
        name += f"_ep{int(epochs)}"
    return f"{name}_s{cfg['seed']}"


def env_assignments(cfg: Mapping[str, str], epochs: int) -> str:
    pairs = [f"{ENV_NAMES[key]}={cfg[key]}" for key in ("seed", "mix", "fte", "lr", "aw")]
    return " ".join(pairs + [f"MPLC_V2_NUM_EPOCHS={int(epochs)}"])


def star_assignments(cfg: Mapping[str, str]) -> str:
    """MPLC*'s non-default settings as launcher variables, for Step 2."""
    return " ".join(f"{ENV_NAMES[k]}={cfg[k]}" for k in ("mix", "fte", "lr", "aw") if cfg[k] != DEFAULTS[k])


# --------------------------------------------------------------------------- results


@dataclass
class ScreenResult:
    state: str
    epochs_run: int
    initial_clean: Optional[float]
    epoch: Optional[int] = None
    clean: Optional[float] = None
    robust: Optional[float] = None
    curve: List[Dict[str, float]] = field(default_factory=list)
    hours: Optional[float] = None

    @property
    def is_loss(self) -> bool:
        """No usable score: collapsed, or no epoch stayed within the clean budget."""
        return self.state == "collapse_aborted" or self.robust is None


def _read_records(run_dir: Path) -> List[dict]:
    path = run_dir / "validation_recalls.jsonl"
    if not path.is_file():
        return []
    by_epoch: Dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            by_epoch[int(record["epoch"])] = record  # a resumed epoch replaces the earlier one
    return [by_epoch[epoch] for epoch in sorted(by_epoch)]


def _wall_hours(run_dir: Path) -> Optional[float]:
    """First to last info.log timestamp, including any pause before a resume."""
    path = run_dir / "info.log"
    if not path.is_file():
        return None
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line[:4].isdigit()]
    try:
        first, last = (datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S") for line in (lines[0], lines[-1]))
    except (IndexError, ValueError):
        return None
    return (last - first).total_seconds() / 3600.0


def read_screen(run_dir: Path, budget: float) -> Optional[ScreenResult]:
    """The screen's result, or None while it is missing or unfinished."""
    status_path = run_dir / "run_status.json"
    if not status_path.is_file():
        return None
    state = json.loads(status_path.read_text(encoding="utf-8")).get("state", "")
    if state not in TERMINAL_STATES:
        return None
    records = _read_records(run_dir)
    initial = next((r for r in records if r["epoch"] == -1), None)
    trained = [r for r in records if r["epoch"] >= 0]
    result = ScreenResult(
        state=state,
        epochs_run=len(trained),
        initial_clean=None if initial is None else float(initial["clean"]["R@1"]),
        curve=[
            {"epoch": r["epoch"], "clean": float(r["clean"]["R@1"]), "robust": float(r["robust_score"])} for r in records
        ],
        hours=_wall_hours(run_dir),
    )
    eligible = [r for r in trained if float(budget) in [float(b) for b in r.get("eligible_budgets", [])]]
    if eligible:
        best = max(eligible, key=lambda r: (float(r["robust_score"]), float(r["clean"]["R@1"])))
        result.epoch = int(best["epoch"])
        result.clean = float(best["clean"]["R@1"])
        result.robust = float(best["robust_score"])
    return result


# --------------------------------------------------------------------------- decisions


def label(delta: Optional[float], noise: float) -> str:
    if delta is None:
        return "n/a"
    if delta > noise:
        return "real gain"
    if delta < -noise:
        return "real loss"
    return "within noise"


def choose(candidates: Sequence[Mapping[str, str]], preferred: Mapping[str, str], results, noise: float):
    """Keep ``preferred`` unless another candidate beats it by more than ``noise``.

    Losses never win. If the preferred option is itself a loss, the best-scoring candidate
    wins. Returns None when every candidate is a loss.
    """
    scored = [(c, results(c)) for c in candidates]
    valid = [(c, r) for c, r in scored if not r.is_loss]
    if not valid:
        return None
    reference = results(preferred)
    if reference.is_loss:
        return max(valid, key=lambda item: item[1].robust)[0]
    better = [(c, r) for c, r in valid if r.robust - reference.robust > noise]
    if not better:
        return dict(preferred)
    return max(better, key=lambda item: item[1].robust)[0]


@dataclass
class Stage:
    name: str
    varied: str
    candidates: List[Dict[str, str]]
    preferred: Dict[str, str]
    winner: Optional[Dict[str, str]] = None
    note: str = ""


@dataclass
class SweepState:
    stages: List[Stage]
    noise: Optional[float] = None
    pending: List[Dict[str, str]] = field(default_factory=list)
    stop_reason: str = ""
    final: Optional[Dict[str, str]] = None


def plan_sweep(results: Callable[[Mapping[str, str]], Optional[ScreenResult]], accept_noise: bool = False) -> SweepState:
    """Walk the stages as far as the finished screens allow.

    ``results(cfg)`` returns the finished ScreenResult or None. The returned state lists
    the decided stages, and either the pending screens of the current stage, a stop
    reason, or the final configuration.
    """
    state = SweepState(stages=[])

    def missing(candidates):
        return [c for c in candidates if results(c) is None]

    def decide(stage: Stage) -> bool:
        state.stages.append(stage)
        state.pending = missing(stage.candidates)
        if state.pending:
            return False
        stage.winner = choose(stage.candidates, stage.preferred, results, state.noise)
        if stage.winner is None:
            state.stop_reason = f"every screen of stage {stage.name} is a loss (collapsed or over the clean budget)"
            return False
        return True

    # 1a: noise.
    seeds = [config(seed=seed) for seed in NOISE_SEEDS]
    noise_stage = Stage("1a", "seed", seeds, seeds[0])
    state.stages.append(noise_stage)
    state.pending = missing(seeds)
    if state.pending:
        return state
    if any(results(c).is_loss for c in seeds):
        state.stop_reason = "a default-configuration seed is a loss, so the noise is undefined"
        return state
    scores = [results(c).robust for c in seeds]
    state.noise = max(scores) - min(scores)
    noise_stage.winner = seeds[0]
    noise_stage.note = f"noise = {state.noise:.2f} (max - min of {', '.join(f'{s:.2f}' for s in scores)})"
    if state.noise > NOISE_LIMIT and not accept_noise:
        state.stop_reason = (
            f"noise {state.noise:.2f} is above {NOISE_LIMIT:.0f} points; raise --val_queries and repeat 1a, "
            "or set STEP1_ACCEPT_NOISE=1 to continue"
        )
        return state

    # 1b: attack mix; linf preferred (fewer attacks).
    mix_candidates = [config(mix=mix) for mix in MIXES]
    if not decide(Stage("1b", "mix", mix_candidates, config(mix="linf"))):
        return state
    mix = state.stages[-1].winner["mix"]

    # 1c: freeze_te x lr; shallower and the recipe's lr preferred.
    grid = [config(mix=mix, fte=fte, lr=lr) for fte in DEPTHS for lr in LRS]
    depth_stage = Stage("1c", "fte x lr", grid, config(mix=mix, fte=DEPTHS[0], lr=LRS[0]))
    if not decide(depth_stage):
        return state
    if (depth_stage.winner["fte"], depth_stage.winner["lr"]) == (DEPTHS[-1], LRS[-1]):
        depth_stage.candidates.append(config(mix=mix, fte=DEPTHS[-1], lr=LR_EXTENSION))
        depth_stage.note = f"grid edge: added lr {LR_EXTENSION} at freeze_te {DEPTHS[-1]}"
        state.stages.pop()
        if not decide(depth_stage):
            return state
    fte, lr = depth_stage.winner["fte"], depth_stage.winner["lr"]

    # 1d: anchor weight; the default preferred.
    anchors = [config(mix=mix, fte=fte, lr=lr, aw=aw) for aw in ANCHORS]
    anchor_stage = Stage("1d", "aw", anchors, config(mix=mix, fte=fte, lr=lr))
    if not decide(anchor_stage):
        return state
    if anchor_stage.winner["aw"] == ANCHORS[-1]:
        anchor_stage.candidates.append(config(mix=mix, fte=fte, lr=lr, aw=ANCHOR_EXTENSION))
        anchor_stage.note = f"grid edge: added aw {ANCHOR_EXTENSION}"
        state.stages.pop()
        if not decide(anchor_stage):
            return state
    best = anchor_stage.winner

    # 1e: re-check the losing mix under the final settings; the current winner preferred.
    other_mix = next(m for m in MIXES if m != best["mix"])
    if not decide(Stage("1e", "mix", [dict(best), dict(best, mix=other_mix)], dict(best))):
        return state
    state.final = state.stages[-1].winner
    return state


# --------------------------------------------------------------------------- outputs

TITLES = {"mix": "attack mix", "fte x lr": "freeze_te × lr", "aw": "λ_anchor"}

TABLE_FIELDS = [
    "stage", "varied", "value", "seed", "mix", "fte", "lr", "aw", "state", "epochs_run", "initial_clean_r1",
    "epoch", "clean_r1", "clean_drop", "robust_score", "delta_vs_preferred", "noise_label", "preferred", "winner",
    "hours", "run_dir",
]


def _value(stage: Stage, cfg: Mapping[str, str]) -> str:
    if stage.varied == "fte x lr":
        return f"{cfg['fte']} x {cfg['lr']}"
    return cfg[stage.varied]


def _fmt(value, spec=".2f"):
    return "" if value is None else format(value, spec)


def table_rows(state: SweepState, results, epochs: int, root: Path) -> List[Dict[str, object]]:
    rows = []
    for stage in state.stages:
        reference = results(stage.preferred)
        for cfg in stage.candidates:
            result = results(cfg)
            delta = None
            if result is not None and not result.is_loss and reference is not None and not reference.is_loss:
                delta = result.robust - reference.robust
            is_preferred = cfg == stage.preferred
            rows.append({
                "stage": stage.name,
                "varied": stage.varied,
                "value": _value(stage, cfg),
                **{key: cfg[key] for key in ("seed", "mix", "fte", "lr", "aw")},
                "state": "pending" if result is None else result.state,
                "epochs_run": "" if result is None else result.epochs_run,
                "initial_clean_r1": _fmt(result and result.initial_clean),
                "epoch": "" if result is None or result.epoch is None else result.epoch,
                "clean_r1": _fmt(result and result.clean),
                "clean_drop": _fmt(
                    None if result is None or result.clean is None or result.initial_clean is None
                    else result.initial_clean - result.clean
                ),
                "robust_score": _fmt(result and result.robust),
                "delta_vs_preferred": _fmt(delta, "+.2f"),
                "noise_label": (
                    "reference" if is_preferred and stage.name != "1a"
                    else "loss" if result is not None and result.is_loss
                    else "" if stage.name == "1a" or state.noise is None
                    else label(delta, state.noise)
                ),
                "preferred": "yes" if is_preferred else "",
                "winner": "yes" if stage.winner is not None and cfg == stage.winner and stage.name != "1a" else "",
                "hours": _fmt(result and result.hours, ".1f"),
                "run_dir": str(root / run_name(cfg, epochs)),
            })
    return rows


def write_tables(state: SweepState, rows, out: Path, epochs: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "screens.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    shown = [f for f in TABLE_FIELDS if f not in ("varied", "run_dir")]
    lines = ["# Step 1 screens", "", f"Noise (1a, max − min): {_fmt(state.noise)}", ""]
    lines += ["| " + " | ".join(shown) + " |", "|" + "---|" * len(shown)]
    lines += ["| " + " | ".join(str(row[f]) for f in shown) + " |" for row in rows]
    (out / "screens.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    decisions = ["# Step 1 decisions", ""]
    decisions.append(f"- Screen length: {epochs} epochs; score: robust score of the budget-3 checkpoint.")
    decisions.append(f"- Noise: {_fmt(state.noise)} (a difference counts only if larger).")
    for stage in state.stages:
        preferred = _value(stage, stage.preferred)
        winner = "undecided" if stage.winner is None else _value(stage, stage.winner)
        text = f"- **{stage.name}** ({stage.varied}): preferred {preferred}; winner **{winner}**."
        if stage.note:
            text += f" {stage.note}."
        decisions.append(text)
    if state.final is not None:
        decisions.append(f"- **MPLC\\***: `{star_assignments(state.final) or '(all defaults)'}`")
    if state.stop_reason:
        decisions.append(f"- **Stopped:** {state.stop_reason}.")
    (out / "decisions.md").write_text("\n".join(decisions) + "\n", encoding="utf-8")

    if state.final is not None:
        (out / "mplc_star.env").write_text(star_assignments(state.final) + "\n", encoding="utf-8")


def write_figures(state: SweepState, results, out: Path, budget: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7, "pdf.fonttype": 42})

    def save(fig, name):
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out / f"{name}.{ext}", dpi=300)
        plt.close(fig)

    decided = [s for s in state.stages if s.name != "1a" and any(results(c) for c in s.candidates)]
    if decided:
        fig, axes = plt.subplots(2, len(decided), figsize=(2.3 * len(decided), 3.4), squeeze=False)
        for column, stage in enumerate(decided):
            labels = [_value(stage, c) for c in stage.candidates]
            found = [results(c) for c in stage.candidates]
            robust = [0.0 if r is None or r.is_loss else r.robust for r in found]
            clean = [0.0 if r is None or r.clean is None else r.clean for r in found]
            colors = ["tab:red" if c == stage.winner else "tab:gray" for c in stage.candidates]
            positions = range(len(labels))
            top, bottom = axes[0][column], axes[1][column]
            top.bar(positions, robust, color=colors, zorder=2)
            reference = results(stage.preferred)
            if reference is not None and not reference.is_loss and state.noise is not None:
                top.axhspan(
                    reference.robust - state.noise, reference.robust + state.noise, color="tab:blue", alpha=0.15, zorder=0
                )
            for x, r in zip(positions, found):
                if r is not None and r.is_loss:
                    top.text(x, 0, "loss", ha="center", va="bottom", fontsize=6)
            top.set_title(f"{stage.name}: {TITLES.get(stage.varied, stage.varied)}")
            top.set_ylabel("robust R@1 (val)" if column == 0 else "")
            bottom.bar(positions, clean, color=colors)
            bottom.set_ylabel("clean R@1 (val)" if column == 0 else "")
            # Zoom both panels on the scored values: differences of a few points matter here.
            for axis, values, pad in ((top, robust, 5.0), (bottom, clean, 1.5)):
                finite = [v for v in values if v > 0]
                if finite:
                    axis.set_ylim(min(finite) - pad, max(finite) + pad)
            for axis in (top, bottom):
                axis.set_xticks(list(positions))
                axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        save(fig, "fig_sensitivity")

    points, seen = [], set()
    for stage in state.stages:
        for c in stage.candidates:
            r = results(c)
            key = tuple(sorted(c.items()))
            if r is not None and not r.is_loss and key not in seen:  # a reused screen is plotted once
                seen.add(key)
                points.append((stage.name, r, c == state.final))
    if points:
        fig, axis = plt.subplots(figsize=(3.5, 2.6))
        for name in dict.fromkeys(name for name, _, _ in points):
            xs = [r.clean for n, r, _ in points if n == name]
            ys = [r.robust for n, r, _ in points if n == name]
            axis.scatter(xs, ys, s=14, label=name)
        for _, r, is_final in points:
            if is_final:
                axis.scatter([r.clean], [r.robust], s=80, marker="*", color="black", label="MPLC*", zorder=3)
                break
        initial = next((r.initial_clean for _, r, _ in points if r.initial_clean is not None), None)
        if initial is not None:
            axis.axvline(initial - budget, color="tab:red", linestyle="--", linewidth=0.8, label=f"{budget:g}-pt budget")
        axis.set_xlabel("clean R@1 (val)")
        axis.set_ylabel("robust R@1 (val)")
        axis.legend(frameon=False)
        save(fig, "fig_tradeoff")

    curves = [(f"1a seed {c['seed']}", results(c)) for c in state.stages[0].candidates]
    if state.final is not None:
        curves.append(("MPLC*", results(state.final)))
    curves = [(name, r) for name, r in curves if r is not None and r.curve]
    if curves:
        fig, (left, right) = plt.subplots(1, 2, figsize=(3.5, 1.9))
        for name, r in curves:
            epochs = [p["epoch"] + 1 for p in r.curve]
            left.plot(epochs, [p["clean"] for p in r.curve], marker=".", label=name)
            right.plot(epochs, [p["robust"] for p in r.curve], marker=".", label=name)
        left.set_title("clean R@1 (val)")
        right.set_title("robust R@1 (val)")
        for axis in (left, right):
            axis.set_xlabel("epoch (0 = pretrained)")
        right.legend(frameon=False, fontsize=5)
        save(fig, "fig_training_curves")


# --------------------------------------------------------------------------- CLI


def _results_reader(root: Path, epochs: int, budget: float):
    cache: Dict[str, Optional[ScreenResult]] = {}

    def results(cfg):
        name = run_name(cfg, epochs)
        if name not in cache:
            cache[name] = read_screen(root / name, budget)
        return cache[name]

    return results


def ensure_config(root: Path, epochs: int, budget: float, batch_size: int, freeze: bool = True) -> None:
    """Freeze the sweep settings on first use; refuse a later run with different ones."""
    path = root / CONFIG_NAME
    wanted = sweep_settings(epochs, budget, batch_size)
    if path.is_file():
        stored = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        stored = {key: stored.get(key) for key in wanted}
        if stored != wanted:
            raise SystemExit(
                f"{path} was created with different settings:\n  stored: {stored}\n  now:    {wanted}\n"
                "Use a new STEP1_ROOT, or restore the original settings."
            )
        return
    if not freeze:
        return
    root.mkdir(parents=True, exist_ok=True)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    content = dict(wanted, git_revision=revision, created=datetime.now().isoformat(timespec="seconds"))
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def summarize(root: Path, epochs: int, budget: float, accept_noise: bool) -> SweepState:
    results = _results_reader(root, epochs, budget)
    state = plan_sweep(results, accept_noise=accept_noise)
    out = root / SUMMARY_DIR
    write_tables(state, table_rows(state, results, epochs, root), out, epochs)
    write_figures(state, results, out, budget)
    return state


def prune(root: Path, state: SweepState, epochs: int) -> List[Path]:
    """Delete the .pth files of every screen except MPLC*'s; logs and scores stay."""
    keep = root / run_name(state.final, epochs)
    removed = []
    for checkpoint in sorted(root.glob("*/*.pth")):
        if checkpoint.parent != keep:
            checkpoint.unlink()
            removed.append(checkpoint)
    return removed


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("next", "summarize", "prune"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, required=True, help="SUPERVLAD_TRAIN_BATCH_SIZE, recorded in the sweep config.")
    parser.add_argument("--accept-noise", action="store_true")
    parser.add_argument("--no-freeze", action="store_true", help="Check but do not create sweep_config.yaml.")
    args = parser.parse_args(argv)

    ensure_config(args.root, args.epochs, args.budget, args.batch_size, freeze=not args.no_freeze)
    state = summarize(args.root, args.epochs, args.budget, args.accept_noise)
    if args.command == "summarize":
        print(f"Summary written to {args.root / SUMMARY_DIR}")
    elif args.command == "next":
        # One line per pending screen of the current stage, or DONE / STOP <reason>.
        if state.pending:
            for cfg in state.pending:
                print(env_assignments(cfg, args.epochs))
        elif state.final is not None:
            print("DONE")
        else:
            print(f"STOP {state.stop_reason}")
    else:
        if state.final is None:
            raise SystemExit("The sweep is not finished; nothing pruned.")
        removed = prune(args.root, state, args.epochs)
        print(f"Removed {len(removed)} checkpoint files; kept {run_name(state.final, args.epochs)}.")


if __name__ == "__main__":
    main()
