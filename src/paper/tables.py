"""The paper's tables and in-text numbers from the runs in logs/paper and the evaluations in
output/paper (scripts/train_*.sh, scripts/eval_*.sh).

Run from the repository root:
    python -m src.paper.tables [--logs logs/paper] [--output output/paper] [--out output/paper/summary]

Writes tab_budget.csv, tab_main.csv, tab_ablation.csv, tab_screens.csv, tab_clean.csv and
numbers.md (trade-off, paired tests, budget sweep and gradient-masking checks).
"""

import argparse
import csv
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

DATASETS = ("msls", "pitts30k", "sped", "nordland")
EPS2 = {"supervlad": 0.03425, "boq": 0.0342}  # 2/255 as evaluated per model
EPS4 = 0.0685
EPS_GRID = (0.01712, 0.03425, 0.0685, 0.137)
EPS_LARGE = 0.548
BUDGET = 5.0
SCREENS = (  # Appendix Table 5, in stage order
    ("attack set", "linf+perc., 8 frozen", "screen_supervlad_araft_closest_mixall_fte8_s0"),
    ("attack set", "linf, 8 frozen", "screen_supervlad_araft_closest_fte8_s0"),
    ("depth, lr", "8 frozen, 3e-6", "screen_supervlad_araft_closest_fte8_lr3e-6_s0"),
    ("depth, lr", "4 frozen, 1e-5", "screen_supervlad_araft_closest_fte4_s0"),
    ("depth, lr", "4 frozen, 3e-6", "screen_supervlad_araft_closest_fte4_lr3e-6_s0"),
    ("depth, lr", "0 frozen, 1e-5", "screen_supervlad_araft_closest_s0"),
    ("depth, lr", "0 frozen, 3e-6", "screen_supervlad_araft_closest_lr3e-6_s0"),
    ("lambda_anc", "0", "screen_supervlad_araft_closest_aw0_s0"),
    ("lambda_anc", "0.1", "screen_supervlad_araft_closest_aw0.1_s0"),
    ("lambda_anc", "1", "screen_supervlad_araft_closest_s0"),
    ("lambda_anc", "10", "screen_supervlad_araft_closest_aw10_s0"),
    ("re-check", "linf+perc., 0 frozen", "screen_supervlad_araft_closest_mixall_s0"),
    ("plain AT", "lr 1e-5", "screen_supervlad_plain_at_s0"),
    ("plain AT", "lr 3e-6", "screen_supervlad_plain_at_lr3e-6_s0"),
)
ABLATIONS = (  # Table 4: (variant, run, eval tag)
    ("ARAFT", "supervlad_araft_s0", "araft"),
    ("Closest positive", "supervlad_araft_closest_s0", "araft_closest"),
    ("Closest positive, hinge", "supervlad_araft_closest_hinge_s0", "araft_closest_hinge"),
    ("No anchor", "supervlad_araft_closest_aw0_s0", "araft_closest_aw0"),
    ("Stronger anchor", "supervlad_araft_aw2_s0", "araft_aw2"),
)


# ----------------------------------------------------------------------------- validation


def validation(logs: Path, run: str):
    """(c0, [(epoch, clean drop, robust score)]) with the initial validation as epoch -1."""
    records = [json.loads(line) for line in (logs / run / "validation_recalls.jsonl").open()]
    c0 = records[0]["initial_clean_r1"]
    return c0, [(r["epoch"], c0 - r["clean_score"], r["robust_score"]) for r in records]


def selected(points, budget=None):
    """The epoch train.py keeps: the first strict maximum of the robust score, among the
    epochs within the clean budget when one is given. Epoch -1 is the pretrained model."""
    best = None
    for point in points:
        if budget is not None and point[1] > budget:
            continue
        if best is None or point[2] > best[2]:
            best = point
    return best


# ----------------------------------------------------------------------------- test sets


class Results:
    def __init__(self, root: Path):
        self.root = root

    def result_dir(self, model, attack, dataset, tag) -> Path:
        done = self.root / model / attack / dataset / tag / "DONE"
        return Path(json.loads(done.read_text())["result_dir"])

    def rows(self, model, attack, dataset, tag):
        with (self.result_dir(model, attack, dataset, tag) / "rank_eval_results.csv").open() as fh:
            return list(csv.DictReader(fh))

    def clean(self, model, attack, dataset, tag) -> float:
        return next(float(r["R@1"]) for r in self.rows(model, attack, dataset, tag) if r["condition"] == "clean_all_queries")

    def attacked(self, model, attack, dataset, tag, eps) -> float:
        return next(float(r["R@1"]) for r in self.rows(model, attack, dataset, tag)
                    if r["epsilon"] and math.isclose(float(r["epsilon"]), eps))

    def correct(self, model, attack, dataset, tag, eps):
        with (self.result_dir(model, attack, dataset, tag) / "per_query_ranks.csv").open() as fh:
            return {r["query_id"]: r["attacked_correct_at_1"] == "True" for r in csv.DictReader(fh)
                    if r["epsilon"] and math.isclose(float(r["epsilon"]), eps)}


def mcnemar_p(a_only: int, b_only: int) -> float:
    """Exact two-sided McNemar p-value from the discordant pairs."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n)


def paired(results, model, dataset, tag_a, tag_b, eps):
    a = results.correct(model, "apgd", dataset, tag_a, eps)
    b = results.correct(model, "apgd", dataset, tag_b, eps)
    assert a.keys() == b.keys(), (model, dataset, tag_a, tag_b)
    a_only = sum(a[q] and not b[q] for q in a)
    b_only = sum(b[q] and not a[q] for q in a)
    return a_only, b_only, mcnemar_p(a_only, b_only)


# ----------------------------------------------------------------------------- tables


def write_csv(path: Path, rows):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=1):
    """The paper's rounding: to two decimals, then half-up to one (47.446 -> 47.45 -> 47.5)."""
    two = Decimal(f"{value:.2f}")
    return str(two if digits == 2 else two.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def checkpoint_tag(logs, model, method, budget):
    """Eval tag of a method's budget-b (or, budget None, most robust) checkpoint. Only the most
    robust checkpoint is evaluated; a budget-5 checkpoint is reported only when it is that one
    or the pretrained model."""
    _, points = validation(logs, f"{model}_{method}_s0")
    best = selected(points)
    if budget is None:
        return method
    chosen = selected(points, budget)
    if chosen[0] == -1:
        return "pretrained"
    if chosen[0] == best[0]:
        return method
    raise SystemExit(f"{model} {method}: budget-{budget:g} checkpoint (epoch {chosen[0]}) is not evaluated")


def result_rows(logs, results, budget):
    rows = []
    for model in ("supervlad", "boq"):
        for method, label in (("plain_at", "Plain AT"), ("araft", "ARAFT")):
            tag = checkpoint_tag(logs, model, method, budget)
            row = {"model": model, "method": label, "evaluated": tag}
            values = {"clean": [], "2": [], "4": []}
            for ds in DATASETS:
                for key, value in (("clean", results.clean(model, "apgd", ds, tag)),
                                   ("2", results.attacked(model, "apgd", ds, tag, EPS2[model])),
                                   ("4", results.attacked(model, "apgd", ds, tag, EPS4))):
                    row[f"{ds}_{key}"] = fmt(value)
                    values[key].append(value)
            for key, vals in values.items():
                row[f"mean_{key}"] = fmt(sum(vals) / len(vals))
            rows.append(row)
    return rows


def ablation_rows(logs, results):
    rows = []
    for label, run, tag in ABLATIONS:
        row = {"variant": label}
        for ds in ("msls", "nordland"):
            row[f"{ds}_apgd4"] = fmt(results.attacked("supervlad", "apgd", ds, tag, EPS4))
            row[f"{ds}_clean"] = fmt(results.clean("supervlad", "apgd", ds, tag))
        _, points = validation(logs, run)
        b5 = selected(points, BUDGET)
        best = selected(points)
        row["val_budget5_drop"], row["val_budget5_robust"] = fmt(b5[1], 2), fmt(b5[2], 2)
        row["val_best_drop"], row["val_best_robust"] = fmt(best[1], 2), fmt(best[2], 2)
        rows.append(row)
    return rows


def screen_rows(logs):
    rows = []
    for stage, setting, run in SCREENS:
        if not (logs / run / "validation_recalls.jsonl").is_file():
            rows.append({"stage": stage, "setting": setting, "drop": "missing", "robust": "missing"})
            continue
        epoch, drop, robust = selected(validation(logs, run)[1], BUDGET)
        in_budget = epoch != -1
        rows.append({"stage": stage, "setting": setting, "drop": fmt(drop, 2) if in_budget else "none in budget",
                     "robust": fmt(robust, 2) if in_budget else "none in budget"})
    return rows


def clean_rows(results):
    rows = []
    for model, entries in (("supervlad", (("Pretrained", "apgd", "pretrained"), ("Clean twin", "pgd", "clean_twin"),
                                          ("ARAFT", "apgd", "araft"), ("Plain AT", "apgd", "plain_at"))),
                           ("boq", (("Pretrained", "apgd", "pretrained"), ("ARAFT", "apgd", "araft"),
                                    ("Plain AT", "apgd", "plain_at")))):
        values = {label: [results.clean(model, attack, ds, tag) for ds in DATASETS] for label, attack, tag in entries}
        for label, vals in values.items():
            rows.append({"model": model, "row": label, **{ds: fmt(v, 2) for ds, v in zip(DATASETS, vals)}})
        for label in ("ARAFT", "Plain AT"):
            rows.append({"model": model, "row": f"{label} drop vs. pretrained",
                         **{ds: fmt(p - v, 2) for ds, p, v in zip(DATASETS, values["Pretrained"], values[label])}})
        if "Clean twin" in values:
            rows.append({"model": model, "row": "ARAFT drop vs. twin",
                         **{ds: fmt(t - v, 2) for ds, t, v in zip(DATASETS, values["Clean twin"], values["ARAFT"])}})
    return rows


# ----------------------------------------------------------------------------- in-text numbers


def tradeoff_lines(logs):
    lines = ["## Trade-off on MSLS-val (Sect. 5.2, Fig. 1)", "",
             "| run | <=3 | <=5 | <=7 | <=10 | most robust (drop) | drop at epoch 3 |", "|---|---|---|---|---|---|---|"]
    frontier = {}
    for run in ("supervlad_araft_s0", "supervlad_araft_s1", "supervlad_plain_at_s0", "boq_araft_s0", "boq_plain_at_s0"):
        _, points = validation(logs, run)
        trained = [p for p in points if p[0] >= 0]
        frontier[run] = trained
        cells = []
        for budget in (3, 5, 7, 10):
            within = [p for p in trained if p[1] <= budget]
            cells.append(fmt(max(p[2] for p in within)) if within else "—")
        best = selected(points)
        epoch3 = next(p[1] for p in trained if p[0] == 3)
        lines.append(f"| {run} | {' | '.join(cells)} | {fmt(best[2])} ({fmt(best[1])}) | {fmt(epoch3)} |")
    lines.append("")
    for model in ("supervlad", "boq"):
        araft_best = max(p[2] for p in frontier[f"{model}_araft_s0"])
        crossing = min((p[1] for p in frontier[f"{model}_plain_at_s0"] if p[2] > araft_best), default=None)
        lines.append(f"- {model}: plain AT first exceeds ARAFT's best robust score ({fmt(araft_best)}) at a clean drop "
                     f"of {fmt(crossing) if crossing is not None else 'never'}.")
    return lines + [""]


def paired_lines(results):
    lines = ["## Paired McNemar tests on the attacked queries (rank-APGD-100)", "",
             "| model | dataset | eps | A | B | A R@1 | B R@1 | A only | B only | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    comparisons = [(m, "araft", "plain_at", eps) for m in ("supervlad", "boq") for eps in (EPS2[m], EPS4)]
    comparisons.append(("supervlad", "araft", "araft_closest", EPS4))
    for model, a, b, eps in comparisons:
        for ds in DATASETS:
            a_only, b_only, p = paired(results, model, ds, a, b, eps)
            lines.append(f"| {model} | {ds} | {eps:g} | {a} | {b} | {fmt(results.attacked(model, 'apgd', ds, a, eps))} | "
                         f"{fmt(results.attacked(model, 'apgd', ds, b, eps))} | {a_only} | {b_only} | {p:.3g} |")
    return lines + [""]


def curve_lines(results):
    lines = ["## Rank-PGD-20 over the budget grid (Sect. 5.1, 5.3, 5.4)", "",
             "| model | dataset | tag | " + " | ".join(f"{e:g}" for e in EPS_GRID) + " | monotone |",
             "|---|---|---|" + "---|" * (len(EPS_GRID) + 1)]
    for model in ("supervlad", "boq"):
        for ds in DATASETS:
            for tag in ("pretrained", "araft", "plain_at"):
                vals = [results.attacked(model, "pgd", ds, tag, e) for e in EPS_GRID]
                monotone = all(x >= y for x, y in zip(vals, vals[1:]))
                lines.append(f"| {model} | {ds} | {tag} | " + " | ".join(fmt(v) for v in vals) + f" | {monotone} |")
    return lines + [""]


def masking_lines(results):
    lines = ["## Gradient-masking checks, SuperVLAD ARAFT (Sect. 5.4)", ""]
    gaps = {ds: results.attacked("supervlad", "apgd", ds, "araft", EPS4) - results.attacked("supervlad", "pgd", ds, "araft", EPS4)
            for ds in DATASETS}
    lines.append("- (i) APGD-100 minus PGD-20 at 4/255: " + ", ".join(f"{ds} {g:+.1f}" for ds, g in gaps.items()))
    base = results.attacked("supervlad", "apgd", "msls", "araft", EPS4)
    for attack, label in (("apgd500", "500 steps"), ("apgd_r3", "3 restarts")):
        delta = results.attacked("supervlad", attack, "msls", "araft", EPS4) - base
        lines.append(f"- (ii) MSLS, {label} vs. APGD-100: {delta:+.2f}")
    lines.append(f"- (iii) MSLS, rank-PGD-20 at eps {EPS_LARGE:g}: "
                 f"{fmt(results.attacked('supervlad', 'pgd', 'msls', 'araft_eps32', EPS_LARGE), 2)}")
    return lines + [""]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=Path("logs/paper"))
    parser.add_argument("--output", type=Path, default=Path("output/paper"))
    parser.add_argument("--out", type=Path, default=Path("output/paper/summary"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = Results(args.output)
    write_csv(args.out / "tab_budget.csv", result_rows(args.logs, results, BUDGET))
    write_csv(args.out / "tab_main.csv", result_rows(args.logs, results, None))
    write_csv(args.out / "tab_ablation.csv", ablation_rows(args.logs, results))
    write_csv(args.out / "tab_screens.csv", screen_rows(args.logs))
    write_csv(args.out / "tab_clean.csv", clean_rows(results))
    lines = ["# Paper numbers", ""] + tradeoff_lines(args.logs) + paired_lines(results) + curve_lines(results) + masking_lines(results)
    (args.out / "numbers.md").write_text("\n".join(lines))
    print(f"Wrote tables and numbers.md to {args.out}")


if __name__ == "__main__":
    main()
