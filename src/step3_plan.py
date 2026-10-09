"""Step 3 evaluation: plan every eval.py call from the finished Step 2 runs and the results on disk.

Models (seed 0; FARE, the clean twin and seed 1 are not evaluated): the pretrained model,
plain_at's full run (scripts/mplc_v2_plain_at_full.sh), MPLC* and the MPLC* ablations that
MPLC* does not already cover. Each trained run is evaluated for two tables: Table A at its
budget-5 checkpoint (the most robust within 5 clean R@1 points, the rule Steps 1 and 2 used)
and Table B at best_model.pth (the most robust overall). A run whose checkpoint for a table is
the initial model is reported as the pretrained model in that table and not evaluated again.

Work is split into pieces, one eval.py call per (run, dataset, model); the transfer run is
one call, since its attacks are crafted on its first model. Without --shared_attacks each
model's attack RNG depends only on (seed, dataset, model tag) and the attacked queries are
fixed, so a per-model call gives the same numbers as a call with every model. A piece is done
once its directory holds DONE, written after eval.py exits 0; a piece killed mid-way is run
again. Run order follows the runbook (§3.2), without runs 1b, 10 and 11.

The targeted pilot runs first: it fails when every main model's targeted success rate is at
most 5 % or every one is at least 95 % (the column could not tell the models apart). Then the
targeted run (3) waits for a pilot that passes at another --target-rank, and every other run
continues.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from . import step2_plan as step2
from .paired_analysis import compute_paired_ccasr
from .phase2_run_group import TERMINAL_STATES
from .robustness_curve_summary import summarize as curve_summary
from .robustness_curve_summary import worst_case

PRETRAINED = "checkpoints/SuperVLAD_base.pth"
SUMMARY_DIR = "summary"
MODELS_FILE = "models.json"
DURATIONS_FILE = "durations.csv"
DONE_FILE = "DONE"
RESULT_FILES = ("rank_eval_results.json", "rank_eval_results.csv", "per_query_ranks.csv")

DATASETS = ("msls", "pitts30k", "sped", "nordland")
ABLATION_DATASETS = ("msls", "nordland")
MAX_QUERIES = 1000
# Attacked queries per dataset at --max_queries 1000 (SPED has 607), for the time estimates.
ATTACKED_QUERIES = {"msls": 1000, "pitts30k": 1000, "sped": 607, "nordland": 1000}

# Normalised budgets: <= 1 / 2 / 4 / 8 per 255 (L-inf), pixel L2 <= 1 / 2 (runbook §3.2).
EPS4 = "0.0685"
EPS_GRID = ("0.01712", "0.03425", "0.0685", "0.137")
L2_EPS = ("4.37", "8.73")
LARGE_EPS = "0.548"

BUDGET = 5.0
TABLES = ("budget5", "best")
TABLE_TITLES = {
    "budget5": "Table A: budget-5 checkpoints (most robust within 5 clean R@1 points)",
    "best": "Table B: best checkpoints (most robust overall)",
}
INITIAL_NOTES = {
    "budget5": "no checkpoint within the 5-point budget (= pretrained)",
    "best": "no epoch beat the initial model (= pretrained)",
}
CHECKPOINT_FILES = {"budget5": "best_model_budget5.pth", "best": "best_model.pth"}
ABLATION_TAGS = {
    "no perceptual attacks": "abl_no_perceptual",
    "no anchor": "abl_no_anchor",
    "hinge loss": "abl_hinge",
    "single positive": "abl_single_pos",
}

PILOT_LOW, PILOT_HIGH = 5.0, 95.0

# Time estimates before a piece of the same (run, dataset) has been measured: the runbook's
# upper bounds per model over all four datasets (§3.1), split by attacked queries.
PGD20_HOURS_PER_EPSILON = 0.87
APGD100_HOURS = 4.3
CLEAN_HOURS = 0.33
MEASURED_MARGIN = 1.2

# Validity thresholds (runbook §3.4), in R@1 points.
APGD_VS_PGD_MAX_GAP = 2.0
MORE_STEPS_MAX_GAP = 1.0
LARGE_EPS_MAX_R1 = 1.0

BEST_LINE = re.compile(r"Best checkpoints by epoch: (\{.*\})")


class Blocked(Exception):
    """A Step 2 run that Step 3 needs is missing or not finished."""


# --------------------------------------------------------------------------- models


def load_star(step2_root: Path) -> Dict[str, str]:
    path = step2_root / step2.CONFIG_NAME
    if not path.is_file():
        raise Blocked(f"{path} is missing: Step 2 has not run in {step2_root}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {key: str(value) for key, value in config["mplc_star"].items()}


def plain_at_env(star: Mapping[str, str]) -> Dict[str, str]:
    """plain_at's full run: the Step 2 full-run settings at MPLC*'s depth and the preferred lr
    (every Step 2 plain_at screen was a loss, so the preferred setting stands)."""
    return {
        "MPLC_V2_FREEZE_TE": star["MPLC_V2_FREEZE_TE"],
        "MPLC_V2_ARMS": "plain_at",
        "MPLC_V2_LR": step2.BASELINE_LRS[0],
        "MPLC_V2_ALIGN_WEIGHT": "1.0",
        **step2.FULL,
        "MPLC_V2_SEED": "0",
    }


@dataclass(frozen=True)
class Entry:
    label: str  # row label in the tables
    group: str  # main | ablation
    tag: str  # model tag; _b5 / _best are added when both checkpoints are evaluated
    run: str  # run directory name under the Step 2 root


def entries(star: Mapping[str, str]) -> Tuple[List[Entry], List[str]]:
    """Trained runs Step 3 evaluates, in table order, and the ablations MPLC* already covers."""
    mplc = {**star, **step2.FULL, "MPLC_V2_ARMS": "mplc", "MPLC_V2_SEED": "0"}
    out = [
        Entry("plain_at", "main", "plain_at_s0", step2.run_name(plain_at_env(star))),
        Entry("MPLC*", "main", "mplc_s0", step2.run_name(mplc)),
    ]
    covered = []
    for label, key, value in step2.ABLATIONS:
        if star.get(key, step2.LAUNCHER_DEFAULTS[key]) == value:
            covered.append(label)
            continue
        out.append(Entry(label, "ablation", ABLATION_TAGS[label], step2.run_name({**mplc, key: value})))
    return out, covered


def checkpoint_epochs(run_dir: Path) -> Dict[str, int]:
    """Epochs behind best_model_budget5.pth and best_model.pth of a finished run (-1 = initial)."""
    status = run_dir / "run_status.json"
    if not status.is_file():
        raise Blocked(f"{run_dir.name} has not run")
    state = json.loads(status.read_text(encoding="utf-8")).get("state", "")
    if state not in TERMINAL_STATES:
        raise Blocked(f"{run_dir.name} is {state or 'unfinished'}")
    log = run_dir / "info.log"
    found = BEST_LINE.findall(log.read_text(encoding="utf-8", errors="replace")) if log.is_file() else []
    if not found:
        raise Blocked(f"{log} records no checkpoint epochs")
    # The log is appended across resumed sessions, so its last line is the final state.
    recorded = ast.literal_eval(found[-1])
    budget = next((value for key, value in recorded.items() if key != "best" and float(key) == BUDGET), None)
    if budget is None or "best" not in recorded:
        raise Blocked(f"{log} records no budget-{BUDGET:g} or best checkpoint: {recorded}")
    epochs = {"budget5": int(budget), "best": int(recorded["best"])}
    validated = set()
    records = run_dir / "validation_recalls.jsonl"
    if records.is_file():
        validated = {int(json.loads(line)["epoch"]) for line in records.read_text(encoding="utf-8").splitlines() if line.strip()}
    for table, epoch in epochs.items():
        if epoch not in validated:
            raise Blocked(f"{run_dir.name}: epoch {epoch} of {CHECKPOINT_FILES[table]} has no validation record")
        if epoch != -1 and not (run_dir / CHECKPOINT_FILES[table]).is_file():
            raise Blocked(f"{run_dir / CHECKPOINT_FILES[table]} is missing")
    return epochs


@dataclass(frozen=True)
class Model:
    tag: str
    path: str
    epoch: Optional[int]  # None for the pretrained model
    group: str  # main | ablation


@dataclass(frozen=True)
class Row:
    label: str
    group: str
    tag: str  # the model whose results fill the row
    checkpoint: str  # epoch, "initial" or "-"
    note: str = ""


@dataclass
class Resolution:
    models: List[Model]
    rows: Dict[str, List[Row]]  # table -> rows (main and ablation)
    blockers: List[str]
    covered: List[str]  # ablations MPLC* already covers

    def tags(self, group: str) -> List[str]:
        return [m.tag for m in self.models if m.group == group]

    def entry_tags(self, label: str) -> List[str]:
        return list(dict.fromkeys(r.tag for t in TABLES for r in self.rows[t] if r.label == label and r.checkpoint != "initial"))


def resolve(step2_root: Path, pretrained: str = PRETRAINED) -> Resolution:
    """Every model to evaluate and every table row; runs that are not finished go to blockers."""
    res = Resolution(
        models=[Model("pretrained", pretrained, None, "main")],
        rows={t: [Row("pretrained", "main", "pretrained", "-")] for t in TABLES},
        blockers=[],
        covered=[],
    )
    try:
        star = load_star(step2_root)
    except Blocked as blocked:
        res.blockers.append(str(blocked))
        return res
    found, res.covered = entries(star)
    for entry in found:
        try:
            epochs = checkpoint_epochs(step2_root / entry.run)
        except Blocked as blocked:
            res.blockers.append(str(blocked))
            continue
        trained = {epoch for epoch in epochs.values() if epoch != -1}
        for table in TABLES:
            epoch = epochs[table]
            if epoch == -1:
                res.rows[table].append(Row(entry.label, entry.group, "pretrained", "initial", INITIAL_NOTES[table]))
                continue
            tag = entry.tag if len(trained) == 1 else f"{entry.tag}_{'b5' if table == 'budget5' else 'best'}"
            if tag not in {m.tag for m in res.models}:
                # One checkpoint for both tables: name it by best_model.pth, which holds the same epoch.
                file = CHECKPOINT_FILES["best" if epoch == epochs["best"] else table]
                res.models.append(Model(tag, str(step2_root / entry.run / file), epoch, entry.group))
            res.rows[table].append(Row(entry.label, entry.group, tag, str(epoch)))
    return res


def freeze_models(root: Path, res: Resolution) -> None:
    """Record the models at the first start; refuse a later call whose checkpoints differ."""
    path = root / MODELS_FILE
    current = {m.tag: {"path": m.path, "epoch": m.epoch, "group": m.group} for m in res.models}
    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))["models"]
        if stored != current:
            raise SystemExit(
                f"{path} was planned from other checkpoints (stored {stored}, now {current}); "
                "restore them or use a new STEP3_ROOT"
            )
        return
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": current, "created": datetime.now().isoformat(timespec="seconds")}, indent=2),
                    encoding="utf-8")


# --------------------------------------------------------------------------- pieces


@dataclass(frozen=True)
class RunSpec:
    key: str  # directory name under the Step 3 root
    label: str
    attack: str = "rank_pgd_linf"
    epsilons: Tuple[str, ...] = (EPS4,)
    steps: int = 20
    restarts: int = 1
    goal: str = "untargeted"
    models: str = "main"  # main | defended | mplc | ablation | transfer
    datasets: Tuple[str, ...] = DATASETS
    shared: bool = False


def run_specs(target_rank: int) -> List[RunSpec]:
    """Step 3's runs in execution order (runbook §3.2, without 1b, 10 and 11)."""
    return [
        RunSpec(f"pilot_tr{target_rank}", f"P: targeted pilot (target rank {target_rank})", goal="targeted", datasets=("sped",)),
        RunSpec("r1_apgd", "1: rank-APGD-100, 4/255 (headline)", attack="rank_apgd_linf", steps=100),
        RunSpec(f"r3_targeted_tr{target_rank}", f"3: targeted rank-PGD-20, 4/255 (target rank {target_rank})", goal="targeted"),
        RunSpec("r4_embshift", "4: embedding-shift PGD-20, 4/255", attack="embshift_linf"),
        RunSpec("r2_pgd_grid", "2: rank-PGD-20, eps grid", epsilons=EPS_GRID),
        RunSpec("r7_transfer", "7: transfer from pretrained", models="transfer", datasets=("msls",), shared=True),
        RunSpec("r8_large_eps", "8: rank-PGD-20 at 32/255", epsilons=(LARGE_EPS,), models="defended", datasets=("msls",)),
        RunSpec("r9_apgd_steps500", "9: rank-APGD-500, 4/255", attack="rank_apgd_linf", steps=500, models="mplc", datasets=("msls",)),
        RunSpec("r9b_apgd_r3", "9b: rank-APGD-100 x 3 restarts, 4/255", attack="rank_apgd_linf", steps=100, restarts=3,
                models="mplc", datasets=("msls",)),
        RunSpec("r6_l2", "6: L2 rank-PGD-20", attack="rank_pgd_l2", epsilons=L2_EPS),
        RunSpec("rA_apgd", "A: ablations, rank-APGD-100, 4/255", attack="rank_apgd_linf", steps=100, models="ablation",
                datasets=ABLATION_DATASETS),
        RunSpec("rA_pgd_grid", "A: ablations, rank-PGD-20, eps grid", epsilons=EPS_GRID, models="ablation",
                datasets=ABLATION_DATASETS),
    ]


def model_set(res: Resolution, which: str) -> List[str]:
    main = res.tags("main")
    defended = [t for t in main if t != "pretrained"]
    return {
        "main": main,
        "defended": defended,
        "mplc": res.entry_tags("MPLC*"),
        "ablation": res.tags("ablation"),
        "transfer": ["pretrained", *defended],
    }[which]


@dataclass(frozen=True)
class Piece:
    spec: RunSpec
    dataset: str
    tags: Tuple[str, ...]

    @property
    def id(self) -> str:
        return f"{self.spec.key}/{self.dataset}/{'shared' if self.spec.shared else self.tags[0]}"

    def directory(self, root: Path) -> Path:
        return root / self.id


def pieces(res: Resolution, target_rank: int) -> List[Piece]:
    out = []
    for spec in run_specs(target_rank):
        tags = model_set(res, spec.models)
        if not tags or (spec.shared and len(tags) < 2):
            continue
        for dataset in spec.datasets:
            if spec.shared:
                out.append(Piece(spec, dataset, tuple(tags)))
            else:
                out += [Piece(spec, dataset, (tag,)) for tag in tags]
    return out


def eval_args(piece: Piece, res: Resolution, root: Path, target_rank: int) -> List[str]:
    """eval.py flags of one piece; the driver adds the model and dataset-folder flags."""
    paths = {m.tag: m.path for m in res.models}
    out_dir = piece.directory(root)
    spec = piece.spec
    args = [
        "--datasets", piece.dataset,
        "--model_paths", *[paths[t] for t in piece.tags],
        "--model_tags", *piece.tags,
        f"--rank_attack={spec.attack}",
        f"--rank_steps={spec.steps}",
        f"--rank_restarts={spec.restarts}",
        f"--rank_attack_goal={spec.goal}",
    ]
    if spec.goal == "targeted":
        args.append(f"--target_rank={target_rank}")
    args.append(f"--max_queries={MAX_QUERIES}")
    if spec.shared:
        args.append("--shared_attacks")
    args += ["--epsilons", *spec.epsilons,
             f"--output_json={out_dir / 'rank_eval_results.json'}",
             f"--output_csv={out_dir / 'rank_eval_results.csv'}"]
    return args


def done_info(root: Path, piece: Piece) -> Optional[Dict[str, object]]:
    path = piece.directory(root) / DONE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def record_done(root: Path, piece: Piece, seconds: float) -> Path:
    """Mark a piece done after eval.py exited 0: its newest complete result directory.

    eval.py writes each attempt into a new timestamped directory, so an attempt killed
    mid-way leaves an incomplete one behind; it is kept and ignored.
    """
    directory = piece.directory(root)
    complete = [d for d in sorted(directory.glob("*/")) if all((d / f).is_file() for f in RESULT_FILES)]
    if not complete:
        raise SystemExit(f"{directory} holds no complete eval.py result ({', '.join(RESULT_FILES)})")
    result = complete[-1]
    info = {"result_dir": str(result), "seconds": round(float(seconds)), "finished": datetime.now().isoformat(timespec="seconds")}
    (directory / DONE_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")
    path = root / DURATIONS_FILE
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(["piece", "run", "dataset", "models", "seconds"])
        writer.writerow([piece.id, piece.spec.key, piece.dataset, len(piece.tags), info["seconds"]])
    return result


# --------------------------------------------------------------------------- estimates


def default_seconds(piece: Piece) -> float:
    spec = piece.spec
    if spec.attack == "rank_apgd_linf":
        hours = APGD100_HOURS * spec.steps / 100 * spec.restarts
    else:
        hours = PGD20_HOURS_PER_EPSILON * len(spec.epsilons) * spec.steps / 20
    share = ATTACKED_QUERIES[piece.dataset] / sum(ATTACKED_QUERIES.values())
    return 3600 * len(piece.tags) * (hours + CLEAN_HOURS) * share


def measured_per_model(root: Path) -> Dict[Tuple[str, str], float]:
    path = root / DURATIONS_FILE
    out: Dict[Tuple[str, str], float] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                key = (row["run"], row["dataset"])
                out[key] = max(out.get(key, 0.0), float(row["seconds"]) / max(1, int(row["models"])))
    return out


def estimate_seconds(piece: Piece, measured: Mapping[Tuple[str, str], float]) -> float:
    per_model = measured.get((piece.spec.key, piece.dataset))
    if per_model is None:
        return default_seconds(piece)
    return MEASURED_MARGIN * per_model * len(piece.tags)


# --------------------------------------------------------------------------- results


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Results:
    """Finished pieces' results, by (run key, dataset, model tag)."""

    def __init__(self, root: Path, all_pieces: Sequence[Piece]):
        self.root = root
        self.dirs: Dict[Tuple[str, str, str], Path] = {}
        for piece in all_pieces:
            info = done_info(root, piece)
            if info is None:
                continue
            for tag in piece.tags:
                self.dirs[(piece.spec.key, piece.dataset, tag)] = Path(str(info["result_dir"]))
        self._rows: Dict[Path, List[Dict[str, str]]] = {}

    def rows(self, run: str, dataset: str, tag: str) -> Optional[List[Dict[str, str]]]:
        directory = self.dirs.get((run, dataset, tag))
        if directory is None:
            return None
        if directory not in self._rows:
            self._rows[directory] = read_csv(directory / "rank_eval_results.csv")
        return [r for r in self._rows[directory] if r["model"] == tag and r["dataset"] == dataset]

    def per_query(self, run: str, dataset: str, tag: str) -> Optional[Path]:
        directory = self.dirs.get((run, dataset, tag))
        return None if directory is None else directory / "per_query_ranks.csv"

    def attacked(self, run: str, dataset: str, tag: str, epsilon: str, field: str = "R@1") -> Optional[float]:
        for row in self.rows(run, dataset, tag) or []:
            if row.get("epsilon") and math.isclose(float(row["epsilon"]), float(epsilon)) and row.get(field, "") != "":
                return float(row[field])
        return None

    def clean(self, dataset: str, tag: str) -> Optional[float]:
        for (run, d, t) in self.dirs:
            if d == dataset and t == tag:
                for row in self.rows(run, d, t) or []:
                    if row["condition"] == "clean_all_queries":
                        return float(row["R@1"])
        return None

    def auc(self, run: str, dataset: str, tag: str) -> Optional[float]:
        rows = self.rows(run, dataset, tag)
        if not rows:
            return None
        return curve_summary(rows)[0]["auc"]


def pilot_state(res: Resolution, results: Results, target_rank: int) -> Tuple[str, Dict[str, Optional[float]]]:
    """pending | pass | fail, and the targeted success rate (%) of each main model."""
    key = run_specs(target_rank)[0].key
    rates = {tag: results.attacked(key, "sped", tag, EPS4, "targeted_success_rate") for tag in model_set(res, "main")}
    if not rates or any(rate is None for rate in rates.values()):
        return "pending", rates
    values = list(rates.values())
    if all(v <= PILOT_LOW for v in values) or all(v >= PILOT_HIGH for v in values):
        return "fail", rates
    return "pass", rates


def pilot_advice(rates: Mapping[str, Optional[float]], target_rank: int) -> str:
    shown = ", ".join(f"{tag} {rate:.1f} %" for tag, rate in rates.items() if rate is not None)
    if all(rate is not None and rate >= PILOT_HIGH for rate in rates.values()):
        hint = f"every model is at or above {PILOT_HIGH:g} %: pick a farther target, e.g. STEP3_TARGET_RANK={target_rank * 5}"
    else:
        hint = f"every model is at or below {PILOT_LOW:g} %: pick a closer target, e.g. STEP3_TARGET_RANK={max(1, target_rank // 3)}"
    return f"targeted pilot failed at target rank {target_rank} ({shown}); {hint}"


def mcnemar_p(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar p-value from the discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / 2**n)


def correct_by_query(path: Path, tag: str, dataset: str) -> Dict[str, bool]:
    return {
        row["query_id"]: row["attacked_correct_at_1"] == "True"
        for row in read_csv(path)
        if row["model_tag"] == tag and row["dataset"] == dataset and row["attacked_rank"] != "-1"
    }


# --------------------------------------------------------------------------- summary


def fmt(value: Optional[float], spec: str = ".2f") -> str:
    return "–" if value is None else format(value, spec)


def markdown_table(fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> List[str]:
    return ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields),
            *["| " + " | ".join(str(row[f]) for f in fields) + " |" for row in rows]]


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


MAIN_FIELDS = [
    "dataset", "model", "checkpoint", "clean R@1", "clean drop", "APGD R@1", "worst-case R@1",
    "targeted success %", "emb-shift R@1", "PGD-20 R@1", "PGD-20 AUC", "L2 R@1 (1 px)", "L2 R@1 (2 px)", "note",
]
ABLATION_FIELDS = ["dataset", "model", "checkpoint", "clean R@1", "clean drop", "APGD R@1", "PGD-20 R@1", "PGD-20 AUC", "note"]


def worst_cases(res: Resolution, results: Results, target_rank: int) -> Tuple[Dict[Tuple[str, str], float], List[str]]:
    """Worst-case R@1 at 4/255 over runs 1, 3 and 4, per (dataset, tag) where all three are done."""
    runs = ("r1_apgd", f"r3_targeted_tr{target_rank}", "r4_embshift")
    out, errors = {}, []
    for dataset in DATASETS:
        for tag in model_set(res, "main"):
            paths = [results.per_query(run, dataset, tag) for run in runs]
            if any(p is None for p in paths):
                continue
            rows = [row for p in paths for row in read_csv(p) if row["model_tag"] == tag and row["dataset"] == dataset]
            try:
                summary = worst_case(rows)
            except ValueError as error:
                errors.append(f"{dataset}/{tag}: {error}")
                continue
            for item in summary:
                if math.isclose(float(item["epsilon"]), float(EPS4)):
                    out[(dataset, tag)] = float(item["worst_case_r1"])
    return out, errors


def main_rows(res: Resolution, results: Results, table: str, worst: Mapping, target_rank: int) -> List[Dict[str, str]]:
    out = []
    for dataset in DATASETS:
        reference = results.clean(dataset, "pretrained")
        for row in res.rows[table]:
            if row.group != "main":
                continue
            clean = results.clean(dataset, row.tag)
            out.append({
                "dataset": dataset, "model": row.label, "checkpoint": row.checkpoint,
                "clean R@1": fmt(clean),
                "clean drop": fmt(None if clean is None or reference is None else reference - clean),
                "APGD R@1": fmt(results.attacked("r1_apgd", dataset, row.tag, EPS4)),
                "worst-case R@1": fmt(worst.get((dataset, row.tag))),
                "targeted success %": fmt(results.attacked(f"r3_targeted_tr{target_rank}", dataset, row.tag, EPS4,
                                                           "targeted_success_rate")),
                "emb-shift R@1": fmt(results.attacked("r4_embshift", dataset, row.tag, EPS4)),
                "PGD-20 R@1": fmt(results.attacked("r2_pgd_grid", dataset, row.tag, EPS4)),
                "PGD-20 AUC": fmt(results.auc("r2_pgd_grid", dataset, row.tag)),
                "L2 R@1 (1 px)": fmt(results.attacked("r6_l2", dataset, row.tag, L2_EPS[0])),
                "L2 R@1 (2 px)": fmt(results.attacked("r6_l2", dataset, row.tag, L2_EPS[1])),
                "note": row.note,
            })
    return out


def ablation_rows(res: Resolution, results: Results, table: str) -> List[Dict[str, str]]:
    out = []
    rows = [r for r in res.rows[table] if r.label == "MPLC*"] + [r for r in res.rows[table] if r.group == "ablation"]
    for dataset in ABLATION_DATASETS:
        reference = results.clean(dataset, "pretrained")
        for row in rows:
            run_apgd, run_pgd = ("r1_apgd", "r2_pgd_grid") if row.group == "main" or row.tag == "pretrained" else ("rA_apgd", "rA_pgd_grid")
            clean = results.clean(dataset, row.tag)
            label = row.label if row.label == "MPLC*" else f"MPLC* without: {row.label}"
            out.append({
                "dataset": dataset, "model": label, "checkpoint": row.checkpoint,
                "clean R@1": fmt(clean),
                "clean drop": fmt(None if clean is None or reference is None else reference - clean),
                "APGD R@1": fmt(results.attacked(run_apgd, dataset, row.tag, EPS4)),
                "PGD-20 R@1": fmt(results.attacked(run_pgd, dataset, row.tag, EPS4)),
                "PGD-20 AUC": fmt(results.auc(run_pgd, dataset, row.tag)),
                "note": row.note,
            })
        for label in res.covered:
            out.append({**{f: "" for f in ABLATION_FIELDS}, "dataset": dataset, "model": f"MPLC* without: {label}",
                        "note": "= MPLC* (MPLC* already trains without it); see the Step 1 screens (validation)"})
    return out


PAIRED_FIELDS = ["table", "dataset", "model_a", "model_b", "queries", "a_r1", "b_r1", "a_only_correct",
                 "b_only_correct", "mcnemar_p", "ccasr_intersection_a", "ccasr_intersection_b", "n_intersection"]


def paired_rows(res: Resolution, results: Results) -> List[Dict[str, object]]:
    """MPLC* against plain_at on run 1's attacked queries, per table and dataset."""
    out = []
    condition = f"rank_apgd_linf_eps_{float(EPS4):g}"
    for table in TABLES:
        tags = {r.label: r.tag for r in res.rows[table]}
        a, b = tags.get("MPLC*"), tags.get("plain_at")
        if a is None or b is None or a == b:
            continue
        for dataset in DATASETS:
            pa, pb = results.per_query("r1_apgd", dataset, a), results.per_query("r1_apgd", dataset, b)
            if pa is None or pb is None:
                continue
            ca, cb = correct_by_query(pa, a, dataset), correct_by_query(pb, b, dataset)
            common = sorted(set(ca) & set(cb))
            a_only = sum(1 for q in common if ca[q] and not cb[q])
            b_only = sum(1 for q in common if cb[q] and not ca[q])
            ccasr = compute_paired_ccasr(pb, pa, base_model_tag=b, trained_model_tag=a, dataset=dataset, condition=condition)
            out.append({
                "table": table, "dataset": dataset, "model_a": a, "model_b": b, "queries": len(common),
                "a_r1": fmt(100.0 * sum(ca[q] for q in common) / len(common) if common else None),
                "b_r1": fmt(100.0 * sum(cb[q] for q in common) / len(common) if common else None),
                "a_only_correct": a_only, "b_only_correct": b_only,
                "mcnemar_p": f"{mcnemar_p(a_only, b_only):.3g}",
                "ccasr_intersection_a": fmt(ccasr["ccasr_intersection_trained"]),
                "ccasr_intersection_b": fmt(ccasr["ccasr_intersection_base"]),
                "n_intersection": ccasr["n_intersection"],
            })
    return out


VALIDITY_FIELDS = ["check", "model", "dataset", "measured", "result"]


def verdict(ok: Optional[bool]) -> str:
    return "pending" if ok is None else "pass" if ok else "**fail**"


def validity_rows(res: Resolution, results: Results) -> List[Dict[str, str]]:
    out = []
    defended = model_set(res, "defended")
    for tag in defended:
        for dataset in DATASETS:
            apgd = results.attacked("r1_apgd", dataset, tag, EPS4)
            pgd = results.attacked("r2_pgd_grid", dataset, tag, EPS4)
            gap = None if apgd is None or pgd is None else pgd - apgd
            out.append({"check": f"APGD-100 at most {APGD_VS_PGD_MAX_GAP:g} below PGD-20 (4/255)", "model": tag,
                        "dataset": dataset, "measured": f"PGD-20 − APGD = {fmt(gap)}",
                        "result": verdict(None if gap is None else gap <= APGD_VS_PGD_MAX_GAP)})
    for tag in model_set(res, "mplc"):
        base = results.attacked("r1_apgd", "msls", tag, EPS4)
        for run, what in (("r9_apgd_steps500", "500 steps"), ("r9b_apgd_r3", "3 restarts")):
            more = results.attacked(run, "msls", tag, EPS4)
            gap = None if base is None or more is None else base - more
            out.append({"check": f"APGD with {what} within {MORE_STEPS_MAX_GAP:g} of run 1", "model": tag, "dataset": "msls",
                        "measured": f"run 1 − {what} = {fmt(gap)}",
                        "result": verdict(None if gap is None else gap <= MORE_STEPS_MAX_GAP)})
    for tag in defended:
        large = results.attacked("r8_large_eps", "msls", tag, LARGE_EPS)
        out.append({"check": f"R@1 ≈ 0 at 32/255 (≤ {LARGE_EPS_MAX_R1:g})", "model": tag, "dataset": "msls",
                    "measured": fmt(large), "result": verdict(None if large is None else large <= LARGE_EPS_MAX_R1)})
        transfer = results.attacked("r7_transfer", "msls", tag, EPS4)
        white = results.attacked("r2_pgd_grid", "msls", tag, EPS4)
        out.append({"check": "transfer from pretrained leaves R@1 ≥ white-box PGD-20 (4/255)", "model": tag, "dataset": "msls",
                    "measured": f"transfer {fmt(transfer)}, white-box {fmt(white)}",
                    "result": verdict(None if transfer is None or white is None else transfer >= white)})
    for run, group, datasets in (("r2_pgd_grid", "main", DATASETS), ("rA_pgd_grid", "ablation", ABLATION_DATASETS)):
        for tag in model_set(res, group):
            for dataset in datasets:
                curve = [results.attacked(run, dataset, tag, eps) for eps in EPS_GRID]
                ok = None if any(v is None for v in curve) else all(b <= a for a, b in zip(curve, curve[1:]))
                out.append({"check": "attacked R@1 never rises with eps (PGD-20)", "model": tag, "dataset": dataset,
                            "measured": " / ".join(fmt(v) for v in curve), "result": verdict(ok)})
    return out


def curve_rows(results: Results, runs: Sequence[str]) -> List[Dict[str, object]]:
    rows = []
    for (run, dataset, tag) in sorted(results.dirs):
        if run in runs:
            rows += results.rows(run, dataset, tag) or []
    out = []
    for item in curve_summary(rows, "pretrained") if rows else []:
        record = {"dataset": item["dataset"], "model": item["model"], "family": item["family"],
                  "clean_r1": fmt(item["clean_r1"]), "auc": fmt(item["auc"]),
                  "clean_drop_vs_pretrained": fmt(item["clean_drop_vs_reference"])}
        record.update({f"R@1@eps={eps:g}": fmt(value) for eps, value in item["r1_by_epsilon"].items()})
        out.append(record)
    return out


def write_curves(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["dataset", "model", "family", "clean_r1"]
    fields += sorted({k for r in rows for k in r if k.startswith("R@1@eps=")}, key=lambda k: float(k.split("=")[1]))
    fields += ["auc", "clean_drop_vs_pretrained"]
    write_csv(path, fields, [{f: r.get(f, "") for f in fields} for r in rows])


def write_summary(root: Path, res: Resolution, all_pieces: Sequence[Piece], target_rank: int) -> None:
    out = root / SUMMARY_DIR
    out.mkdir(parents=True, exist_ok=True)
    results = Results(root, all_pieces)
    worst, worst_errors = worst_cases(res, results, target_rank)

    for table in TABLES:
        rows = main_rows(res, results, table, worst, target_rank)
        write_csv(out / f"table_{table}.csv", MAIN_FIELDS, rows)
        lines = [f"# {TABLE_TITLES[table]}", "",
                 f"Attacks at ≤ 4/255 unless named; {MAX_QUERIES} attacked queries per dataset (SPED: all 607). "
                 f"Worst case: a query counts only if runs 1, 3 and 4 all fail to break it. "
                 f"Targeted: target rank {target_rank}. Clean drop: against the pretrained model.", "",
                 *markdown_table(MAIN_FIELDS, rows)]
        (out / f"table_{table}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        rows = ablation_rows(res, results, table)
        write_csv(out / f"ablations_{table}.csv", ABLATION_FIELDS, rows)
        lines = [f"# Ablations, {TABLE_TITLES[table][9:]}", "", *markdown_table(ABLATION_FIELDS, rows)]
        (out / f"ablations_{table}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_curves(out / "curves_pgd.csv", curve_rows(results, ("r2_pgd_grid", "rA_pgd_grid")))
    write_curves(out / "curves_l2.csv", curve_rows(results, ("r6_l2",)))
    write_csv(out / "worst_case.csv", ["dataset", "model", "worst_case_r1"],
              [{"dataset": d, "model": t, "worst_case_r1": fmt(v)} for (d, t), v in sorted(worst.items())])
    write_csv(out / "paired.csv", PAIRED_FIELDS, paired_rows(res, results))

    validity = validity_rows(res, results)
    lines = ["# Validity checks (runbook §3.4)", "", *markdown_table(VALIDITY_FIELDS, validity)]
    if worst_errors:
        lines += ["", "Worst case could not be computed:", *[f"- {e}" for e in worst_errors]]
    (out / "validity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    state, rates = pilot_state(res, results, target_rank)
    measured = measured_per_model(root)
    piece_rows = []
    for piece in all_pieces:
        info = done_info(root, piece)
        blocked = piece.spec.key.startswith("r3_") and state != "pass"
        piece_rows.append({
            "piece": piece.id, "run": piece.spec.label,
            "state": "done" if info else "waits for the pilot" if blocked else "pending",
            "hours": fmt(info["seconds"] / 3600 if info else estimate_seconds(piece, measured) / 3600, ".1f")
            + ("" if info else " (est.)"),
            "result": info["result_dir"] if info else "",
        })
    done = sum(1 for r in piece_rows if r["state"] == "done")
    model_rows = [{"tag": m.tag, "group": m.group, "epoch": "-" if m.epoch is None else m.epoch, "path": m.path} for m in res.models]
    shown_rates = ", ".join(f"{t} {fmt(r, '.1f')} %" for t, r in rates.items())
    lines = [
        "# Step 3 status", "",
        f"- Pieces done: {done} / {len(piece_rows)}",
        f"- Targeted pilot (target rank {target_rank}): {state} ({shown_rates})",
        *[f"- **Blocked:** {b}" for b in res.blockers],
        *[f"- Ablation '{c}' is covered by MPLC* itself" for c in res.covered],
        "", "## Models", "", *markdown_table(["tag", "group", "epoch", "path"], model_rows),
        "", "## Table rows", "",
        *markdown_table(["table", "row", "model", "checkpoint", "note"],
                        [{"table": t, "row": r.label, "model": r.tag, "checkpoint": r.checkpoint, "note": r.note}
                         for t in TABLES for r in res.rows[t]]),
        "", "## Pieces", "", *markdown_table(["piece", "run", "state", "hours", "result"], piece_rows),
    ]
    (out / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- next


def pending(root: Path, res: Resolution, all_pieces: Sequence[Piece], target_rank: int) -> Tuple[List[Piece], List[Piece], str, Dict]:
    """Pieces that may run now, pieces waiting for the pilot, the pilot state and its rates."""
    results = Results(root, all_pieces)
    state, rates = pilot_state(res, results, target_rank)
    runnable, waiting = [], []
    for piece in all_pieces:
        if done_info(root, piece) is not None:
            continue
        if piece.spec.key.startswith("r3_") and state != "pass":
            waiting.append(piece)
        else:
            runnable.append(piece)
    return runnable, waiting, state, rates


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("next", "list", "record", "summarize", "plain-at-env"))
    parser.add_argument("piece", nargs="?", help="record: the piece id that finished.")
    parser.add_argument("--root", type=Path, default=Path("output/mplc_v2_step3"), help="Step 3 output root (STEP3_ROOT).")
    parser.add_argument("--step2-root", type=Path, default=Path("logs/mplc_v2_step2"), help="Step 2 run directory (STEP2_ROOT).")
    parser.add_argument("--pretrained", default=PRETRAINED, help="Pretrained SuperVLAD checkpoint.")
    parser.add_argument("--target-rank", type=int, default=10, help="Targeted attack's target rank (STEP3_TARGET_RANK).")
    parser.add_argument("--time-left", type=float, default=math.inf, help="next: seconds left before the session's stop-by time.")
    parser.add_argument("--seconds", type=float, default=0.0, help="record: the piece's wall time.")
    args = parser.parse_args(argv)
    if args.target_rank < 1:
        raise SystemExit("--target-rank must be at least 1")

    if args.command == "plain-at-env":
        print(step2.env_assignments(plain_at_env(load_star(args.step2_root))))
        print(step2.run_name(plain_at_env(load_star(args.step2_root))))
        return

    res = resolve(args.step2_root, args.pretrained)
    all_pieces = pieces(res, args.target_rank)

    if args.command == "record":
        piece = next((p for p in all_pieces if p.id == args.piece), None)
        if piece is None:
            raise SystemExit(f"record: {args.piece!r} is not a Step 3 piece")
        print(record_done(args.root, piece, args.seconds))
        write_summary(args.root, res, all_pieces, args.target_rank)
        return

    if args.command == "summarize":
        write_summary(args.root, res, all_pieces, args.target_rank)
        print(f"Tables written to {args.root / SUMMARY_DIR}")
        return

    if args.command == "list":
        for blocker in res.blockers:
            print(f"# blocked: {blocker}")
        runnable, waiting, state, _ = pending(args.root, res, all_pieces, args.target_rank)
        measured = measured_per_model(args.root)
        for piece in runnable + waiting:
            note = " (waits for the pilot)" if piece in waiting else ""
            print(f"=== {piece.id}: {piece.spec.label}, ~{estimate_seconds(piece, measured) / 3600:.1f} h{note}")
            print("+ eval.py " + " ".join(shlex.quote(a) for a in eval_args(piece, res, args.root, args.target_rank)))
        print(f"# {len(runnable) + len(waiting)} pending pieces; targeted pilot: {state}")
        return

    # next
    if res.blockers:
        print("STOP Step 3 needs every run finished: " + "; ".join(res.blockers))
        return
    try:
        freeze_models(args.root, res)
    except SystemExit as stop:
        print(f"STOP {stop}")
        return
    runnable, waiting, state, rates = pending(args.root, res, all_pieces, args.target_rank)
    if state == "fail":
        print(f"WARNING: {pilot_advice(rates, args.target_rank)}; run 3 waits, every other run continues", file=sys.stderr)
    if not runnable:
        if waiting:
            print(f"STOP {pilot_advice(rates, args.target_rank)}")
        else:
            print("DONE")
        return
    measured = measured_per_model(args.root)
    for piece in runnable:
        estimate = estimate_seconds(piece, measured)
        if estimate <= args.time_left:
            print(f"PIECE {piece.id}")
            print(round(estimate))
            print(piece.directory(args.root))
            print("\n".join(eval_args(piece, res, args.root, args.target_rank)))
            return
    shortest = min(estimate_seconds(p, measured) for p in runnable)
    print(f"WAIT the shortest pending piece needs ~{shortest / 3600:.1f} h")


if __name__ == "__main__":
    main()
