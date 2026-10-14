"""Follow-up evaluation: plan every eval.py call for the follow-up models, from their runs and the results on disk.

A Step 3 clone (src/step3_plan.py) with its own model table and output root; the pieces, the
DONE markers, the time estimates and the summary writers are Step 3's.

Models: the pretrained model, always, and the rows of MODELS below. Each row comes from one
training run: in FOLLOWUP_ROOT (scripts/mplc_v2_followup_train.sh) or in the Step 2 root. A
run counts only once its run_status.json records a terminal state; until then the row is "not
trained yet" and skipped. The checkpoints are chosen from the run's validation_recalls.jsonl:

  Table A (budget-5): the most robust trained epoch within the 5-point clean budget
      (eligible_budgets holds 5.0; the rule of step1_sweep.read_screen). None: the row shows
      the pretrained model.
  Table B (best): the trained epoch with the highest robust score (ties: higher clean R@1).
  Both are the files the trainer wrote, best_model_budget5.pth and best_model.pth: one
      checkpoint (AB, best_model.pth) when A and B are the same epoch, else one per table.
      An epoch that does not beat the initial validation's robust score is not a checkpoint
      (the trainer then keeps the initial model in the file), and A and B are one checkpoint
      only when B's robust score is not tied, since the trainer keeps the first of tied epochs.
  The clean twin: last_model.pth for both tables, evaluated for clean recall only.

--model KEY:TABLES:PATH (Step 3's syntax) overrides a row's automatic choice. Every checkpoint is
recorded in models.json under the evaluation root as in Step 3, and the work is planned from
everything recorded: a later call adds the rows trained since, a recorded row is not chosen
again, and the same row and table with another path is refused.

The clean twin is evaluated with rank-PGD-20 at the smallest grid epsilon (1/255) on the
1000-query cap: eval.py has no clean-only mode, and the clean recall on all queries
(clean_all_queries) comes with every call.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import step1_sweep as sweep
from . import step3_plan as step3
from .paired_analysis import compute_paired_ccasr
from .phase2_run_group import TERMINAL_STATES
from .step3_plan import (
    ABLATION_DATASETS,
    ATTACKED_QUERIES,
    DATASETS,
    EPS4,
    EPS_GRID,
    INITIAL_NOTES,
    L2_EPS,
    LARGE_EPS,
    MAX_QUERIES,
    PRETRAINED_PATH,
    SUMMARY_DIR,
    TABLE_LETTERS,
    TABLE_TITLES,
    TABLES,
    TAG_SUFFIXES,
    Checkpoint,
    Model,
    Piece,
    Refused,
    Resolution,
    Results,
    Row,
    RunSpec,
    fmt,
    markdown_table,
    read_csv,
    verdict,
    write_csv,
)

DEFAULT_ROOT = "output/mplc_v2_followup"
FOLLOWUP_ROOT = "logs/mplc_v2_followup"
STEP2_ROOT = "logs/mplc_v2_step2"

EPS2 = EPS_GRID[1]  # <= 2/255
APGD_EPSILONS = (EPS2, EPS4)
CLEAN_EPS = EPS_GRID[0]  # the twin's attack, a by-product of its clean recall
BUDGET = 5.0


@dataclass(frozen=True)
class Entry:
    key: str
    label: str  # the table row
    group: str  # main | ablation | clean
    source: str  # followup | step2: the root that holds the run
    run: str


# In table order. The model tag of a row is its key, plus _b5 / _best when A and B differ.
ENTRIES = (
    Entry("sp", "ARAFT (hardest positive), seed 0", "main", "step2", "mplc_v2_supervlad_mplc_mixlinf_sp_fte0_s0"),
    Entry("sp_s1", "ARAFT, seed 1", "main", "followup", "mplc_v2_supervlad_mplc_mixlinf_sp_fte0_s1"),
    Entry("sp_eps2", "ARAFT, trained at 2/255", "main", "followup", "mplc_v2_supervlad_mplc_eps0.0342_mixlinf_sp_fte0_s0"),
    Entry("fare_eps2", "FARE, trained at 2/255", "main", "followup", "mplc_v2_supervlad_fare_eps0.0342_fte0_s0"),
    Entry("plain_at_eps2", "plain_at, trained at 2/255", "main", "followup", "mplc_v2_supervlad_plain_at_eps0.0342_fte0_s0"),
    Entry("sp_hinge", "hinge loss (hardest positive)", "ablation", "followup", "mplc_v2_supervlad_mplc_mixlinf_hinge_sp_fte0_s0"),
    Entry("sp_aw2", "anchor weight 2", "ablation", "followup", "mplc_v2_supervlad_mplc_aw2_mixlinf_sp_fte0_s0"),
    Entry("sp_aw3", "anchor weight 3", "ablation", "followup", "mplc_v2_supervlad_mplc_aw3_mixlinf_sp_fte0_s0"),
    Entry("twin", "clean twin (last_model.pth)", "clean", "step2", "mplc_v2_supervlad_clean_ft_fte0_s0"),
)
MODELS = {entry.key: entry for entry in ENTRIES}
MAIN_KEYS = tuple(entry.key for entry in ENTRIES if entry.group == "main")
HEADLINE = "sp"  # the new main method: runs 9 and 9b, the ablation tables' reference, the paired tests
TWIN = "twin"
# Paired per-query comparisons on run 1's attacked queries: (model a, model b).
PAIRS = (("sp", "fare_eps2"), ("sp", "plain_at_eps2"), ("sp", "sp_s1"))

# Time estimates as in Step 3, but APGD's per-epsilon cost counts once per epsilon.
APGD100_HOURS = step3.APGD100_HOURS
PGD20_HOURS_PER_EPSILON = step3.PGD20_HOURS_PER_EPSILON
CLEAN_HOURS = step3.CLEAN_HOURS


# --------------------------------------------------------------------------- models


def run_directory(entry: Entry, followup_root: Path, step2_root: Path) -> Path:
    return Path(step2_root if entry.source == "step2" else followup_root) / entry.run


def run_state(run_dir: Path) -> str:
    path = run_dir / "run_status.json"
    if not path.is_file():
        return "no run_status.json" if run_dir.is_dir() else "not started"
    return str(json.loads(path.read_text(encoding="utf-8")).get("state", "")) or "no state"


def parse_model(spec: str) -> Checkpoint:
    """KEY:TABLES:PATH as in Step 3; a malformed spec or a missing file is refused (exit 2)."""
    parts = spec.split(":", 2)
    if len(parts) != 3 or not parts[2]:
        raise Refused(f"--model {spec!r}: expected KEY:TABLES:PATH", 2)
    key, tables, path = parts[0], parts[1].upper(), parts[2]
    if key not in MODELS:
        raise Refused(f"--model {spec!r}: KEY must be one of {', '.join(MODELS)}", 2)
    if tables not in TABLE_LETTERS:
        raise Refused(f"--model {spec!r}: TABLES must be A, B or AB", 2)
    if key == TWIN and tables != "AB":
        raise Refused(f"--model {spec!r}: the clean twin takes one checkpoint for both tables (TABLES AB)", 2)
    if path != PRETRAINED_PATH:
        path = os.path.normpath(path)
        if not Path(path).is_file():
            raise Refused(f"--model {spec!r}: {path} is not a file", 2)
    return Checkpoint(key, tables, path)


def merge(known: Sequence[Checkpoint], new: Sequence[Checkpoint]) -> List[Checkpoint]:
    """step3.merge: another checkpoint for a recorded row and table is refused (exit 3)."""
    try:
        return step3.merge(known, new)
    except Refused as refused:
        raise Refused(str(refused).replace("STEP3_ROOT", "FOLLOWUP_EVAL_ROOT"), refused.code) from None


@dataclass(frozen=True)
class Choice:
    checkpoints: Tuple[Checkpoint, ...]
    note: str  # the epochs behind the choice


def choose(entry: Entry, run_dir: Path) -> Optional[Choice]:
    """A row's automatic checkpoints, or None while its run is not trained (not in a terminal state).

    A terminal run whose chosen checkpoint file is missing is refused (exit 3).
    """
    state = run_state(run_dir)
    if state not in TERMINAL_STATES:
        return None

    def checkpoint(tables: str, name: Optional[str]) -> Checkpoint:
        if name is None:
            return Checkpoint(entry.key, tables, PRETRAINED_PATH)
        path = os.path.normpath(str(run_dir / name))
        if not Path(path).is_file():
            raise Refused(f"{entry.key}: {run_dir} is trained ({state}) but its checkpoint file is missing: {path}", 3)
        return Checkpoint(entry.key, tables, path)

    if entry.group == "clean":
        return Choice((checkpoint("AB", "last_model.pth"),), "clean twin: last_model.pth")

    records = sweep._read_records(run_dir)
    initial = next((r for r in records if int(r["epoch"]) == -1), None)
    floor = -math.inf if initial is None else float(initial["robust_score"])
    trained = [r for r in records if int(r["epoch"]) >= 0]
    screen = sweep.read_screen(run_dir, BUDGET)
    a = screen.epoch if screen is not None and screen.robust is not None and screen.robust > floor else None
    best = max(trained, key=lambda r: (float(r["robust_score"]), float(r["clean"]["R@1"])), default=None)
    b = None if best is None or float(best["robust_score"]) <= floor else int(best["epoch"])
    tied = b is not None and sum(float(r["robust_score"]) == float(best["robust_score"]) for r in trained) > 1

    shown = {None: "none (= pretrained)"}
    note = f"budget-5 epoch {shown.get(a, a)}, best epoch {shown.get(b, b)}"
    if a is None and b is None:
        return Choice((checkpoint("AB", None),), note)
    if a is not None and a == b and not tied:
        return Choice((checkpoint("AB", "best_model.pth"),), note)
    if a is not None and a == b:
        note += " (tied robust score: one file per table)"
    return Choice((checkpoint("A", None if a is None else "best_model_budget5.pth"),
                   checkpoint("B", None if b is None else "best_model.pth")), note)


def requested_keys(text: str) -> List[str]:
    """FOLLOWUP_EVAL_MODELS, in table order; empty: every row. An unknown key is refused (exit 2)."""
    keys = text.split()
    unknown = [key for key in keys if key not in MODELS]
    if unknown:
        raise Refused(f"FOLLOWUP_EVAL_MODELS: unknown key {', '.join(unknown)} (keys: {', '.join(MODELS)})", 2)
    return [key for key in MODELS if key in keys] if keys else list(MODELS)


@dataclass
class Plan:
    known: List[Checkpoint]  # recorded in models.json
    checkpoints: List[Checkpoint]  # known plus this call's
    chosen: Dict[str, str] = field(default_factory=dict)  # key -> why, for the rows chosen in this call
    untrained: Dict[str, str] = field(default_factory=dict)  # requested key -> its run and state
    runs: Dict[str, str] = field(default_factory=dict)  # key -> run directory
    unrecorded: Dict[str, str] = field(default_factory=dict)  # requested key, trained but not recorded -> its run


def gather(root: Path, keys: Sequence[str], overrides: Sequence[Checkpoint], followup_root: Path, step2_root: Path,
           choose_new: bool) -> Plan:
    """The recorded checkpoints plus the overrides and, with choose_new, the automatic choice for every
    requested row that is neither recorded nor overridden and whose run is trained. Without choose_new
    such a row is only noted as unrecorded (trained since the models were last recorded)."""
    known = step3.recorded(root)
    runs = {key: str(run_directory(MODELS[key], followup_root, step2_root)) for key in MODELS}
    new = list(overrides)
    chosen, untrained, unrecorded = {}, {}, {}
    for key in keys:
        if any(c.key == key for c in [*known, *overrides]):
            continue
        run_dir = Path(runs[key])
        state = run_state(run_dir)
        if state not in TERMINAL_STATES:
            untrained[key] = f"{run_dir}: {state}"
        elif not choose_new:
            unrecorded[key] = f"{run_dir}: {state}"
        else:
            choice = choose(MODELS[key], run_dir)
            new += choice.checkpoints
            chosen[key] = choice.note
    return Plan(known, merge(known, new), chosen, untrained, runs, unrecorded)


def resolve(checkpoints: Sequence[Checkpoint], pretrained: str = step3.PRETRAINED) -> Resolution:
    """step3.resolve over the follow-up rows; no ablation is covered by the headline model."""
    res = Resolution(
        models=[Model("pretrained", pretrained, "main")],
        rows={t: [Row("pretrained", "main", "pretrained", "-")] for t in TABLES},
        checkpoints=list(checkpoints),
        covered=[],
    )
    for entry in ENTRIES:
        given = [c for c in checkpoints if c.key == entry.key]
        for table in TABLES:
            checkpoint = next((c for c in given if table in TABLE_LETTERS[c.tables]), None)
            if checkpoint is None:
                continue
            if checkpoint.path == PRETRAINED_PATH:
                res.rows[table].append(Row(entry.label, entry.group, "pretrained", "initial", INITIAL_NOTES[table]))
                continue
            tag = entry.key + TAG_SUFFIXES[checkpoint.tables]
            if tag not in {m.tag for m in res.models}:
                res.models.append(Model(tag, checkpoint.path, entry.group))
            res.rows[table].append(Row(entry.label, entry.group, tag, Path(checkpoint.path).name))
    return res


def transfer_waits_for(res: Resolution) -> List[str]:
    """Run 7 is one call over every main model, so it waits until every main row is recorded."""
    keys = {c.key for c in res.checkpoints}
    return [key for key in MAIN_KEYS if key not in keys]


# --------------------------------------------------------------------------- pieces


def run_specs(target_rank: int) -> List[RunSpec]:
    """The follow-up runs in execution order (Step 3's keys; no pilot: Step 3's passed at rank 10)."""
    return [
        RunSpec("r1_apgd", "1: rank-APGD-100, 2/255 and 4/255 (headline)", attack="rank_apgd_linf", steps=100,
                epsilons=APGD_EPSILONS),
        RunSpec(f"r3_targeted_tr{target_rank}", f"3: targeted rank-PGD-20, 4/255 (target rank {target_rank})", goal="targeted"),
        RunSpec("r4_embshift", "4: embedding-shift PGD-20, 4/255", attack="embshift_linf"),
        RunSpec("r2_pgd_grid", "2: rank-PGD-20, eps grid", epsilons=EPS_GRID),
        RunSpec("clean_twin", "T: clean twin, clean recall (rank-PGD-20 at 1/255 rides along)", epsilons=(CLEAN_EPS,),
                models="clean"),
        RunSpec("r7_transfer", "7: transfer from pretrained", models="transfer", datasets=("msls",), shared=True),
        RunSpec("r8_large_eps", "8: rank-PGD-20 at 32/255", epsilons=(LARGE_EPS,), models="defended", datasets=("msls",)),
        RunSpec("r9_apgd_steps500", "9: rank-APGD-500, 4/255", attack="rank_apgd_linf", steps=500, models=HEADLINE,
                datasets=("msls",)),
        RunSpec("r9b_apgd_r3", "9b: rank-APGD-100 x 3 restarts, 4/255", attack="rank_apgd_linf", steps=100, restarts=3,
                models=HEADLINE, datasets=("msls",)),
        RunSpec("r6_l2", "6: L2 rank-PGD-20", attack="rank_pgd_l2", epsilons=L2_EPS),
        RunSpec("rA_apgd", "A: ablations, rank-APGD-100, 2/255 and 4/255", attack="rank_apgd_linf", steps=100,
                epsilons=APGD_EPSILONS, models="ablation", datasets=ABLATION_DATASETS),
        RunSpec("rA_pgd_grid", "A: ablations, rank-PGD-20, eps grid", epsilons=EPS_GRID, models="ablation",
                datasets=ABLATION_DATASETS),
    ]


def model_set(res: Resolution, which: str) -> List[str]:
    main = res.tags("main")
    defended = [t for t in main if t != "pretrained"]
    return {
        "main": main,
        "defended": defended,
        HEADLINE: res.entry_tags(MODELS[HEADLINE].label),
        "ablation": res.tags("ablation"),
        "clean": res.tags("clean"),
        "transfer": ["pretrained", *defended],
    }[which]


def pieces(res: Resolution, target_rank: int) -> List[Piece]:
    out = []
    for spec in run_specs(target_rank):
        tags = model_set(res, spec.models)
        if not tags or (spec.shared and len(tags) < 2) or (spec.models == "transfer" and transfer_waits_for(res)):
            continue
        for dataset in spec.datasets:
            if spec.shared:
                out.append(Piece(spec, dataset, tuple(tags)))
            else:
                out += [Piece(spec, dataset, (tag,)) for tag in tags]
    return out


def default_seconds(piece: Piece) -> float:
    spec = piece.spec
    if spec.attack == "rank_apgd_linf":
        hours = APGD100_HOURS * len(spec.epsilons) * spec.steps / 100 * spec.restarts
    else:
        hours = PGD20_HOURS_PER_EPSILON * len(spec.epsilons) * spec.steps / 20
    share = ATTACKED_QUERIES[piece.dataset] / sum(ATTACKED_QUERIES.values())
    return 3600 * len(piece.tags) * (hours + CLEAN_HOURS) * share


def estimate_seconds(piece: Piece, measured: Mapping[Tuple[str, str], float]) -> float:
    per_model = measured.get((piece.spec.key, piece.dataset))
    if per_model is None:
        return default_seconds(piece)
    return step3.MEASURED_MARGIN * per_model * len(piece.tags)


# --------------------------------------------------------------------------- summary

APGD2_FIELD = "APGD R@1 (2/255)"
MAIN_FIELDS = [f for name in step3.MAIN_FIELDS for f in ((name, APGD2_FIELD) if name == "APGD R@1" else (name,))]
ABLATION_FIELDS = [f for name in step3.ABLATION_FIELDS for f in ((name, APGD2_FIELD) if name == "APGD R@1" else (name,))]
CLEAN_TWIN_FIELDS = ["dataset", "model", "tables", "tag", "checkpoint", "clean R@1", "clean R@5",
                     "drop vs pretrained", "drop vs twin"]
PAIRED_FIELDS = [*step3.PAIRED_FIELDS[:2], "epsilon", *step3.PAIRED_FIELDS[2:]]
LETTERS = {"budget5": "A", "best": "B"}


def clean_recall(results: Results, dataset: str, tag: Optional[str], field_name: str = "R@1") -> Optional[float]:
    """Results.clean for any recall: clean recall on all queries, from any finished run of the model."""
    for (run, d, t) in results.dirs:
        if d == dataset and t == tag:
            for row in results.rows(run, d, t) or []:
                if row["condition"] == "clean_all_queries" and row.get(field_name, "") != "":
                    return float(row[field_name])
    return None


def main_rows(res: Resolution, results: Results, table: str, worst: Mapping, target_rank: int) -> List[Dict[str, str]]:
    """Step 3's main table rows plus APGD at 2/255; step3.main_rows walks datasets, then main rows."""
    rows = step3.main_rows(res, results, table, worst, target_rank)
    tags = [(dataset, row.tag) for dataset in DATASETS for row in res.rows[table] if row.group == "main"]
    assert len(rows) == len(tags)
    return [{**row, APGD2_FIELD: fmt(results.attacked("r1_apgd", dataset, tag, EPS2))} for row, (dataset, tag) in zip(rows, tags)]


def ablation_rows(res: Resolution, results: Results, table: str) -> List[Dict[str, str]]:
    """The headline model, then each ablation, on MSLS and Nordland (step3.ablation_rows names MPLC*)."""
    head = MODELS[HEADLINE].label
    rows = [r for r in res.rows[table] if r.label == head] + [r for r in res.rows[table] if r.group == "ablation"]
    out = []
    for dataset in ABLATION_DATASETS:
        reference = results.clean(dataset, "pretrained")
        for row in rows:
            own = row.group == "ablation" and row.tag != "pretrained"
            run_apgd, run_pgd = ("rA_apgd", "rA_pgd_grid") if own else ("r1_apgd", "r2_pgd_grid")
            clean = results.clean(dataset, row.tag)
            out.append({
                "dataset": dataset, "model": row.label, "checkpoint": row.checkpoint,
                "clean R@1": fmt(clean),
                "clean drop": fmt(None if clean is None or reference is None else reference - clean),
                "APGD R@1": fmt(results.attacked(run_apgd, dataset, row.tag, EPS4)),
                APGD2_FIELD: fmt(results.attacked(run_apgd, dataset, row.tag, EPS2)),
                "PGD-20 R@1": fmt(results.attacked(run_pgd, dataset, row.tag, EPS4)),
                "PGD-20 AUC": fmt(results.auc(run_pgd, dataset, row.tag)),
                "note": row.note,
            })
    return out


def clean_twin_rows(res: Resolution, results: Results) -> List[Dict[str, str]]:
    """Clean R@1 / R@5 on all queries: pretrained, the clean twin and the headline model's checkpoints."""
    entries = [("pretrained", "-", "pretrained", "-")]
    for key in (TWIN, HEADLINE):
        by_tag: Dict[str, Tuple[str, List[str]]] = {}
        for table in TABLES:
            for row in res.rows[table]:
                if row.label == MODELS[key].label:
                    by_tag.setdefault(row.tag, (row.checkpoint, []))[1].append(LETTERS[table])
        entries += [(MODELS[key].label, "-" if key == TWIN else ", ".join(tables), tag, checkpoint)
                    for tag, (checkpoint, tables) in by_tag.items()]
    twin_tag = next((tag for label, _, tag, _ in entries if label == MODELS[TWIN].label), None)
    out = []
    for dataset in DATASETS:
        pretrained = clean_recall(results, dataset, "pretrained")
        twin = clean_recall(results, dataset, twin_tag) if twin_tag else None
        for label, tables, tag, checkpoint in entries:
            r1 = clean_recall(results, dataset, tag)
            out.append({
                "dataset": dataset, "model": label, "tables": tables, "tag": tag, "checkpoint": checkpoint,
                "clean R@1": fmt(r1), "clean R@5": fmt(clean_recall(results, dataset, tag, "R@5")),
                "drop vs pretrained": fmt(None if r1 is None or pretrained is None else pretrained - r1),
                "drop vs twin": fmt(None if r1 is None or twin is None else twin - r1),
            })
    return out


def correct_by_query(path: Path, tag: str, dataset: str, condition: str) -> Dict[str, bool]:
    """step3.correct_by_query for one condition: run 1 holds two epsilons."""
    return {
        row["query_id"]: row["attacked_correct_at_1"] == "True"
        for row in read_csv(path)
        if row["model_tag"] == tag and row["dataset"] == dataset and row["condition"] == condition
        and row["attacked_rank"] != "-1"
    }


def paired_rows(res: Resolution, results: Results) -> List[Dict[str, object]]:
    """PAIRS on run 1's attacked queries, per table, dataset and epsilon; exact McNemar as Step 3."""
    out = []
    for table in TABLES:
        tags = {r.label: r.tag for r in res.rows[table]}
        for key_a, key_b in PAIRS:
            a, b = tags.get(MODELS[key_a].label), tags.get(MODELS[key_b].label)
            if a is None or b is None or a == b:
                continue
            for dataset in DATASETS:
                pa, pb = results.per_query("r1_apgd", dataset, a), results.per_query("r1_apgd", dataset, b)
                if pa is None or pb is None:
                    continue
                for eps in APGD_EPSILONS:
                    condition = f"rank_apgd_linf_eps_{float(eps):g}"
                    ca, cb = correct_by_query(pa, a, dataset, condition), correct_by_query(pb, b, dataset, condition)
                    common = sorted(set(ca) & set(cb))
                    if not common:
                        continue
                    a_only = sum(1 for q in common if ca[q] and not cb[q])
                    b_only = sum(1 for q in common if cb[q] and not ca[q])
                    ccasr = compute_paired_ccasr(pb, pa, base_model_tag=b, trained_model_tag=a, dataset=dataset,
                                                 condition=condition)
                    out.append({
                        "table": table, "dataset": dataset, "epsilon": eps, "model_a": a, "model_b": b,
                        "queries": len(common),
                        "a_r1": fmt(100.0 * sum(ca[q] for q in common) / len(common)),
                        "b_r1": fmt(100.0 * sum(cb[q] for q in common) / len(common)),
                        "a_only_correct": a_only, "b_only_correct": b_only,
                        "mcnemar_p": f"{step3.mcnemar_p(a_only, b_only):.3g}",
                        "ccasr_intersection_a": fmt(ccasr["ccasr_intersection_trained"]),
                        "ccasr_intersection_b": fmt(ccasr["ccasr_intersection_base"]),
                        "n_intersection": ccasr["n_intersection"],
                    })
    return out


def validity_rows(res: Resolution, results: Results) -> List[Dict[str, str]]:
    """Step 3's checks (APGD vs PGD-20 at 4/255, 32/255, transfer, monotonic curves), plus APGD vs
    PGD-20 at 2/255 and the headline model's 500-step / 3-restart convergence checks."""
    out = step3.validity_rows(res, results)
    for tag in model_set(res, "defended"):
        for dataset in DATASETS:
            apgd = results.attacked("r1_apgd", dataset, tag, EPS2)
            pgd = results.attacked("r2_pgd_grid", dataset, tag, EPS2)
            gap = None if apgd is None or pgd is None else pgd - apgd
            out.append({"check": f"APGD-100 at most {step3.APGD_VS_PGD_MAX_GAP:g} below PGD-20 (2/255)", "model": tag,
                        "dataset": dataset, "measured": f"PGD-20 − APGD = {fmt(gap)}",
                        "result": verdict(None if gap is None else gap <= step3.APGD_VS_PGD_MAX_GAP)})
    for tag in model_set(res, HEADLINE):
        base = results.attacked("r1_apgd", "msls", tag, EPS4)
        for run, what in (("r9_apgd_steps500", "500 steps"), ("r9b_apgd_r3", "3 restarts")):
            more = results.attacked(run, "msls", tag, EPS4)
            gap = None if base is None or more is None else base - more
            out.append({"check": f"APGD with {what} within {step3.MORE_STEPS_MAX_GAP:g} of run 1", "model": tag,
                        "dataset": "msls", "measured": f"run 1 − {what} = {fmt(gap)}",
                        "result": verdict(None if gap is None else gap <= step3.MORE_STEPS_MAX_GAP)})
    return out


def write_table(out: Path, name: str, title: str, preamble: Sequence[str], fields: Sequence[str], rows) -> None:
    write_csv(out / f"{name}.csv", fields, rows)
    lines = [f"# {title}", "", *preamble, *markdown_table(fields, rows)]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(root: Path, res: Resolution, all_pieces: Sequence[Piece], target_rank: int,
                  plan: Optional[Plan] = None) -> None:
    """Step 3's summary for the follow-up rows; plan (gather's) adds the rows' runs and states to status.md."""
    plan = plan or Plan([], list(res.checkpoints))
    out = root / SUMMARY_DIR
    out.mkdir(parents=True, exist_ok=True)
    results = Results(root, all_pieces)
    worst, worst_errors = step3.worst_cases(res, results, target_rank)

    for table in TABLES:
        note = [f"Attacks at ≤ 4/255 unless named; {MAX_QUERIES} attacked queries per dataset (SPED: all 607). "
                f"Worst case (4/255): a query counts only if runs 1, 3 and 4 all fail to break it. "
                f"Targeted: target rank {target_rank}. Clean drop: against the pretrained model "
                "(against the clean twin: clean_twin.md).", ""]
        write_table(out, f"table_{table}", TABLE_TITLES[table], note, MAIN_FIELDS,
                    main_rows(res, results, table, worst, target_rank))
        write_table(out, f"ablations_{table}", f"Ablations, {TABLE_TITLES[table][9:]}", [], ABLATION_FIELDS,
                    ablation_rows(res, results, table))
    write_table(out, "clean_twin", "Clean recall: clean twin, pretrained and the headline model",
                ["Clean R@1 / R@5 on all queries of each test set; the twin is its last_model.pth.", ""],
                CLEAN_TWIN_FIELDS, clean_twin_rows(res, results))

    step3.write_curves(out / "curves_pgd.csv", step3.curve_rows(results, ("r2_pgd_grid", "rA_pgd_grid")))
    step3.write_curves(out / "curves_l2.csv", step3.curve_rows(results, ("r6_l2",)))
    write_csv(out / "worst_case.csv", ["dataset", "model", "worst_case_r1"],
              [{"dataset": d, "model": t, "worst_case_r1": fmt(v)} for (d, t), v in sorted(worst.items())])
    write_csv(out / "paired.csv", PAIRED_FIELDS, paired_rows(res, results))

    lines = ["# Validity checks (runbook §3.4)", "", *markdown_table(step3.VALIDITY_FIELDS, validity_rows(res, results))]
    if worst_errors:
        lines += ["", "Worst case could not be computed:", *[f"- {e}" for e in worst_errors]]
    (out / "validity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    measured = step3.measured_per_model(root)
    piece_rows = []
    for piece in all_pieces:
        info = step3.done_info(root, piece)
        piece_rows.append({
            "piece": piece.id, "run": piece.spec.label, "state": "done" if info else "pending",
            "hours": fmt(info["seconds"] / 3600 if info else estimate_seconds(piece, measured) / 3600, ".1f")
            + ("" if info else " (est.)"),
            "result": info["result_dir"] if info else "",
        })
    done = sum(1 for r in piece_rows if r["state"] == "done")
    recorded_tables = {}
    for c in res.checkpoints:
        recorded_tables.setdefault(c.key, []).append(c.tables)
    def entry_status(key: str) -> str:
        if key in recorded_tables:
            return f"recorded ({', '.join(recorded_tables[key])})"
        if key in plan.untrained:
            return f"not trained yet ({plan.untrained[key]})"
        if key in plan.unrecorded:
            return f"trained, not recorded yet ({plan.unrecorded[key]})"
        return "not requested (FOLLOWUP_EVAL_MODELS)"

    entry_rows = [{"key": e.key, "row": e.label, "group": e.group, "run": plan.runs.get(e.key, e.run),
                   "status": entry_status(e.key)} for e in ENTRIES]
    waits = transfer_waits_for(res)
    lines = [
        "# Follow-up evaluation status", "",
        f"- Pieces done: {done} / {len(piece_rows)}",
        f"- Targeted attack: target rank {target_rank}",
        *([f"- Not trained yet: {', '.join(plan.untrained)}; rerun after scripts/mplc_v2_followup_train.sh finishes them"]
          if plan.untrained else []),
        *([f"- Trained, not recorded yet: {', '.join(plan.unrecorded)}; the driver adds them when it starts or "
           "runs out of pieces"] if plan.unrecorded else []),
        *([f"- Run 7 (transfer from pretrained) waits for {' and '.join(waits)}"] if waits else []),
        "", "## Follow-up models", "", *markdown_table(["key", "row", "group", "run", "status"], entry_rows),
        "", "## Checkpoints recorded", "",
        *markdown_table(["key", "tables", "path"], [{"key": c.key, "tables": c.tables, "path": c.path} for c in res.checkpoints]),
        "", "## Models", "", *markdown_table(["tag", "group", "path"], [{"tag": m.tag, "group": m.group, "path": m.path}
                                                                        for m in res.models]),
        "", "## Table rows", "",
        *markdown_table(["table", "row", "model", "checkpoint", "note"],
                        [{"table": t, "row": r.label, "model": r.tag, "checkpoint": r.checkpoint, "note": r.note}
                         for t in TABLES for r in res.rows[t]]),
        "", "## Pieces", "", *markdown_table(["piece", "run", "state", "hours", "result"], piece_rows),
    ]
    (out / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("register", "next", "list", "record", "summarize"))
    parser.add_argument("piece", nargs="?", help="record: the piece id that finished.")
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT), help="Evaluation output root (FOLLOWUP_EVAL_ROOT).")
    parser.add_argument("--followup-root", type=Path, default=Path(FOLLOWUP_ROOT), help="Follow-up runs (FOLLOWUP_ROOT).")
    parser.add_argument("--step2-root", type=Path, default=Path(STEP2_ROOT), help="Step 2 runs (FOLLOWUP_STEP2_ROOT).")
    parser.add_argument("--keys", default="", help="Rows to evaluate, space-separated (FOLLOWUP_EVAL_MODELS); default all.")
    parser.add_argument("--model", dest="overrides", action="append", default=[], metavar="KEY:TABLES:PATH",
                        help="A checkpoint that overrides a row's automatic choice (repeatable); register records it.")
    parser.add_argument("--pretrained", default=step3.PRETRAINED, help="Pretrained SuperVLAD checkpoint.")
    parser.add_argument("--target-rank", type=int, default=10, help="Targeted attack's target rank (FOLLOWUP_EVAL_TARGET_RANK).")
    parser.add_argument("--time-left", type=float, default=math.inf, help="next: seconds left before the session's stop-by time.")
    parser.add_argument("--seconds", type=float, default=0.0, help="record: the piece's wall time.")
    args = parser.parse_args(argv)

    try:
        if args.target_rank < 1:
            raise Refused("FOLLOWUP_EVAL_TARGET_RANK must be at least 1", 2)
        keys = requested_keys(args.keys)
        overrides = [parse_model(spec) for spec in args.overrides]
        plan = gather(args.root, keys, overrides, args.followup_root, args.step2_root,
                      choose_new=args.command in ("register", "list"))
    except Refused as refused:
        print(refused, file=sys.stderr)
        sys.exit(refused.code)

    if args.command == "register":
        step3.register(args.root, plan.checkpoints)
        for checkpoint in plan.checkpoints:
            print(f"model {checkpoint.spec}{'' if checkpoint in plan.known else ' (new)'}")
        for key, note in plan.chosen.items():
            print(f"#   {key}: {note}")
        for key, note in plan.untrained.items():
            print(f"# {key} not trained yet, skipped: {note}")
        return

    res = resolve(plan.checkpoints, args.pretrained)
    all_pieces = pieces(res, args.target_rank)
    waits = transfer_waits_for(res)

    if args.command == "record":
        piece = next((p for p in all_pieces if p.id == args.piece), None)
        if piece is None:
            raise SystemExit(f"record: {args.piece!r} is not a follow-up evaluation piece")
        print(step3.record_done(args.root, piece, args.seconds))
        write_summary(args.root, res, all_pieces, args.target_rank, plan)
        return

    if args.command == "summarize":
        write_summary(args.root, res, all_pieces, args.target_rank, plan)
        print(f"Tables written to {args.root / SUMMARY_DIR}")
        return

    runnable = [p for p in all_pieces if step3.done_info(args.root, p) is None]
    measured = step3.measured_per_model(args.root)

    if args.command == "list":
        for checkpoint in plan.checkpoints:
            print(f"# model {checkpoint.spec}{'' if checkpoint in plan.known else ' (not recorded yet)'}")
        for key, note in plan.chosen.items():
            print(f"#   {key}: {note}")
        for key, note in plan.untrained.items():
            print(f"# {key} not trained yet, skipped: {note}")
        if waits:
            print(f"# run 7 (transfer) waits for {' and '.join(waits)}")
        for piece in runnable:
            print(f"=== {piece.id}: {piece.spec.label}, ~{estimate_seconds(piece, measured) / 3600:.1f} h")
            print("+ eval.py " + " ".join(shlex.quote(a) for a in step3.eval_args(piece, res, args.root, args.target_rank)))
        total = sum(estimate_seconds(p, measured) for p in runnable) / 3600
        print(f"# {len(runnable)} pending pieces, ~{total:.0f} h")
        return

    # next
    missing = [m.path for m in res.models if not Path(m.path).is_file()]
    if missing:
        print("STOP recorded checkpoint files are missing: " + ", ".join(missing))
        return
    other = [rank for rank in step3.run3_ranks(args.root) if rank != args.target_rank]
    if other:
        print(f"STOP run 3 has finished pieces at target rank {other[0]}; set FOLLOWUP_EVAL_TARGET_RANK={other[0]} "
              "so every model is attacked at the same rank")
        return
    if not runnable:
        if plan.unrecorded:  # trained since the models were recorded: the driver records them and goes on
            print("REGISTER " + " ".join(plan.unrecorded))
        elif plan.untrained:
            print("UNTRAINED " + " ".join(plan.untrained))
            for key, note in plan.untrained.items():
                print(f"{key}: {note}")
        else:
            print("DONE")
            if waits:
                print(f"run 7 (transfer) waits for {' and '.join(waits)}: "
                      "add them to FOLLOWUP_EVAL_MODELS or give them with --model")
        return
    for piece in runnable:
        estimate = estimate_seconds(piece, measured)
        if estimate <= args.time_left:
            print(f"PIECE {piece.id}")
            print(round(estimate))
            print(piece.directory(args.root))
            print("\n".join(step3.eval_args(piece, res, args.root, args.target_rank)))
            return
    shortest = min(estimate_seconds(p, measured) for p in runnable)
    print(f"WAIT the shortest pending piece needs ~{shortest / 3600:.1f} h")


if __name__ == "__main__":
    main()
