#!/usr/bin/env python3
"""Compare short screening runs at one validated epoch.

Usage:
    python scripts/screen_summary.py logs/ --epoch 12 --budgets 5 7

Every subdirectory of ROOT holding a validation_recalls.jsonl is one run. The
table shows, at the requested epoch (matched literally against each record's
"epoch" field), the clean R@1, its drop from the initial model (epoch -1), the
R@1 under each attack and the robust score. Each b<N> column names the most
robust validated epoch whose clean drop is at most N, as "<epoch> (<robust>)".
When a record at that epoch holds clean R@1 on out-of-domain sets ("ood_clean"),
one "ood <set>" column per set follows the robust score.
"""
import argparse
import json
from pathlib import Path

ATTACK_PREFIX = "rank_pgd_linf_eps_"
LABELS = {"last_epoch": "last epoch", "clean_r1": "clean R@1", "clean_drop": "clean drop"}


def _attack_column(name):
    return "R@1 " + (name[len(ATTACK_PREFIX):] if name.startswith(ATTACK_PREFIX) else name)


def _load_run(run_dir):
    with open(run_dir / "validation_recalls.jsonl") as f:
        # A resumed run may log an epoch twice; the later record wins.
        records = {rec["epoch"]: rec for rec in map(json.loads, filter(str.strip, f))}
    status_path = run_dir / "run_status.json"
    state = json.loads(status_path.read_text()).get("state", "-") if status_path.exists() else "-"
    return records, state


def _initial_clean(records):
    if -1 in records:
        return records[-1]["clean"]["R@1"]
    return next((rec["initial_clean_r1"] for rec in records.values() if "initial_clean_r1" in rec), None)


def _drop(initial, rec):
    return None if initial is None else initial - rec["clean"]["R@1"]


def summarise(root: Path, epoch: int, budgets=(5.0, 7.0)) -> list:
    runs = {
        run_dir.name: _load_run(run_dir)
        for run_dir in sorted(Path(root).iterdir())
        if (run_dir / "validation_recalls.jsonl").is_file()
    }
    attack_names = sorted({name for records, _ in runs.values() for rec in records.values() for name in rec.get("attacks", {})})
    # Set names in first-seen order, from the compared epoch only.
    ood_names = list(dict.fromkeys(
        name for records, _ in runs.values() for name in (records.get(epoch) or {}).get("ood_clean", {})
    ))
    rows = []
    for run_name, (records, state) in runs.items():
        initial = _initial_clean(records)
        rec = records.get(epoch)
        row = {"run": run_name, "state": state, "last_epoch": max(records) if records else "-"}
        if rec is None:
            row.update({"clean_r1": "-", "clean_drop": "-"})
            row.update({_attack_column(name): "-" for name in attack_names})
            row["robust"] = "-"
            row.update({f"ood {name}": "-" for name in ood_names})
        else:
            drop = _drop(initial, rec)
            row["clean_r1"] = rec["clean"]["R@1"]
            row["clean_drop"] = "-" if drop is None else drop
            attacks = rec.get("attacks", {})
            row.update({_attack_column(name): attacks[name]["R@1"] if name in attacks else "-" for name in attack_names})
            row["robust"] = rec["robust_score"]
            ood_clean = rec.get("ood_clean", {})
            row.update({f"ood {name}": ood_clean.get(name, "-") for name in ood_names})
        trained = [r for e, r in records.items() if e >= 0]
        for budget in budgets:
            eligible = [r for r in trained if (d := _drop(initial, r)) is not None and d <= budget]
            best = max(eligible, key=lambda r: r["robust_score"], default=None)
            row[f"b{budget:g}"] = "-" if best is None else f"{best['epoch']} ({best['robust_score']:.2f})"
        rows.append(row)
    return rows


def _cell(value):
    return f"{value:.2f}" if isinstance(value, float) else str(value)


def format_table(rows) -> str:
    if not rows:
        return "no runs found"
    keys = list(rows[0])
    lines = [
        "| " + " | ".join(LABELS.get(key, key) for key in keys) + " |",
        "|" + "|".join("---" for _ in keys) + "|",
    ]
    lines += ["| " + " | ".join(_cell(row[key]) for key in keys) + " |" for row in rows]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare screening runs at one validated epoch.")
    parser.add_argument("root", type=Path, help="directory whose subdirectories are training runs")
    parser.add_argument("--epoch", type=int, default=12, help="epoch value to compare, as logged in validation_recalls.jsonl")
    parser.add_argument("--budgets", type=float, nargs="+", default=[5.0, 7.0], help="clean R@1 drop budgets")
    args = parser.parse_args(argv)
    print(format_table(summarise(args.root, args.epoch, tuple(args.budgets))))


if __name__ == "__main__":
    main()
