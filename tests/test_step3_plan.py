import csv
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import yaml

from src import step2_plan as step2
from src import step3_plan as step3
from tests.test_step1_sweep import REPO_ROOT, TRAIN_SCRIPT, run_script

STEP3_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_step3.sh"
PLAIN_AT_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_plain_at_full.sh"
STAR = {"MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_FREEZE_TE": "0", "MPLC_V2_LR": "1e-5", "MPLC_V2_ALIGN_WEIGHT": "1.0"}
PLAIN_AT = "mplc_v2_supervlad_plain_at_fte0_s0"
MPLC = "mplc_v2_supervlad_mplc_mixlinf_fte0_s0"
NO_ANCHOR = "mplc_v2_supervlad_mplc_aw0_mixlinf_fte0_s0"
HINGE = "mplc_v2_supervlad_mplc_mixlinf_hinge_fte0_s0"
SINGLE_POS = "mplc_v2_supervlad_mplc_mixlinf_sp_fte0_s0"


def write_run(root, name, budget5, best, state="early_stopped", earlier=None):
    """A Step 2 run whose last log line puts budget-5 and best at these epochs (-1 = initial)."""
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    lines = []
    if earlier is not None:
        lines.append(f"2026-10-02 10:00:00   Best checkpoints by epoch: {{'1': -1, '3': -1, '5': {earlier}, 'best': {earlier}}}")
    lines.append(f"2026-10-02 22:30:33   Best checkpoints by epoch: {{'1': -1, '3': -1, '5': {budget5}, 'best': {best}}}")
    (run_dir / "info.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    epochs = sorted({-1, budget5, best, *([earlier] if earlier is not None else [])})
    (run_dir / "validation_recalls.jsonl").write_text("".join(json.dumps({"epoch": e}) + "\n" for e in epochs), encoding="utf-8")
    for file in ("best_model.pth", "best_model_budget5.pth", "last_model.pth"):
        (run_dir / file).write_bytes(b"")
    return run_dir


def write_step2(root, plain_at=(-1, 12), sp=(30, 40), skip=()):
    """A finished Step 2 (plus the plain_at full run) like the server's, with MPLC* = linf, fte 0."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / step2.CONFIG_NAME).write_text(yaml.safe_dump({"mplc_star": STAR}), encoding="utf-8")
    runs = {PLAIN_AT: plain_at, MPLC: (63, 63), NO_ANCHOR: (-1, 96), HINGE: (57, 57), SINGLE_POS: sp}
    for name, (budget5, best) in runs.items():
        if name not in skip:
            write_run(root, name, budget5, best)
    return root


class PlainAtRunTests(unittest.TestCase):
    def test_env_names_the_run_beside_step2_with_full_length_settings(self):
        env = step3.plain_at_env(STAR)
        self.assertEqual(step2.run_name(env), PLAIN_AT)
        with tempfile.TemporaryDirectory() as tmp:
            result = run_script(TRAIN_SCRIPT, {"MPLC_V2_DRY_RUN": "1", "MPLC_V2_RUN_ROOT": tmp, **env})
        self.assertEqual(result.returncode, 0, result.stdout)
        command = result.stdout.split()
        for flag in (f"--save_dir={PLAIN_AT}", "--val_every=3", "--patience=6", "--lr_plateau_patience=3",
                     "--freeze_te=0", "--seed=0", "--defense_loss=hinge"):
            self.assertIn(flag, command)
        self.assertNotIn("--num_epochs=12", command)

    def test_entries_name_every_step2_run(self):
        found, covered = step3.entries(STAR)
        self.assertEqual([e.run for e in found], [PLAIN_AT, MPLC, NO_ANCHOR, HINGE, SINGLE_POS])
        self.assertEqual(covered, ["no perceptual attacks"])


class CheckpointTests(unittest.TestCase):
    def test_last_logged_line_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = write_run(tmp, "r", 57, 57, earlier=3)
            self.assertEqual(step3.checkpoint_epochs(run_dir), {"budget5": 57, "best": 57})

    def test_unfinished_missing_or_unrecorded_runs_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(step3.Blocked, "has not run"):
                step3.checkpoint_epochs(Path(tmp) / "absent")
            with self.assertRaisesRegex(step3.Blocked, "interrupted"):
                step3.checkpoint_epochs(write_run(tmp, "paused", 54, 54, state="interrupted"))
            run_dir = write_run(tmp, "unvalidated", 5, 5)
            (run_dir / "validation_recalls.jsonl").write_text(json.dumps({"epoch": -1}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(step3.Blocked, "no validation record"):
                step3.checkpoint_epochs(run_dir)
            run_dir = write_run(tmp, "no_file", 5, 5)
            (run_dir / "best_model.pth").unlink()
            with self.assertRaisesRegex(step3.Blocked, "missing"):
                step3.checkpoint_epochs(run_dir)


class ResolveTests(unittest.TestCase):
    def test_tables_share_a_checkpoint_or_split_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = step3.resolve(write_step2(tmp))
        self.assertEqual(res.blockers, [])
        models = {m.tag: m for m in res.models}
        self.assertEqual(list(models), ["pretrained", "plain_at_s0", "mplc_s0", "abl_no_anchor", "abl_hinge",
                                        "abl_single_pos_b5", "abl_single_pos_best"])
        # Same epoch for both tables: one model, named by best_model.pth.
        self.assertTrue(models["mplc_s0"].path.endswith(f"{MPLC}/best_model.pth"))
        self.assertEqual(models["mplc_s0"].epoch, 63)
        # Distinct trained epochs: both evaluated.
        self.assertTrue(models["abl_single_pos_b5"].path.endswith("best_model_budget5.pth"))
        self.assertEqual((models["abl_single_pos_b5"].epoch, models["abl_single_pos_best"].epoch), (30, 40))
        rows = {(t, r.label): r for t in step3.TABLES for r in res.rows[t]}
        # Initial-model budget-5: the pretrained model fills Table A, best_model.pth Table B.
        self.assertEqual((rows["budget5", "plain_at"].tag, rows["budget5", "plain_at"].checkpoint), ("pretrained", "initial"))
        self.assertIn("5-point budget", rows["budget5", "plain_at"].note)
        self.assertEqual((rows["best", "plain_at"].tag, rows["best", "plain_at"].checkpoint), ("plain_at_s0", "12"))
        self.assertEqual(rows["budget5", "single positive"].tag, "abl_single_pos_b5")
        self.assertEqual(rows["best", "single positive"].tag, "abl_single_pos_best")
        self.assertEqual(res.tags("main"), ["pretrained", "plain_at_s0", "mplc_s0"])

    def test_unfinished_runs_are_listed_as_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = write_step2(tmp, skip=(SINGLE_POS, PLAIN_AT))
            write_run(root, HINGE, 54, 54, state="interrupted")
            res = step3.resolve(root)
        self.assertEqual(len(res.blockers), 3)
        self.assertEqual(res.tags("ablation"), ["abl_no_anchor"])

    def test_missing_step2_config_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = step3.resolve(Path(tmp))
        self.assertIn("step2_config.yaml", res.blockers[0])


class PieceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.res = step3.resolve(write_step2(Path(self.tmp.name) / "s2"))
        self.root = Path(self.tmp.name) / "s3"

    def tearDown(self):
        self.tmp.cleanup()

    def test_order_models_and_datasets(self):
        found = step3.pieces(self.res, 10)
        counts = {}
        for piece in found:
            counts[piece.spec.key] = counts.get(piece.spec.key, 0) + 1
        self.assertEqual(counts, {"pilot_tr10": 3, "r1_apgd": 12, "r3_targeted_tr10": 12, "r4_embshift": 12,
                                  "r2_pgd_grid": 12, "r7_transfer": 1, "r8_large_eps": 2, "r9_apgd_steps500": 1,
                                  "r9b_apgd_r3": 1, "r6_l2": 12, "rA_apgd": 8, "rA_pgd_grid": 8})
        self.assertEqual(list(dict.fromkeys(p.spec.key for p in found))[:3], ["pilot_tr10", "r1_apgd", "r3_targeted_tr10"])
        transfer = next(p for p in found if p.spec.key == "r7_transfer")
        self.assertEqual(transfer.tags, ("pretrained", "plain_at_s0", "mplc_s0"))  # attacks crafted on pretrained
        self.assertEqual(transfer.id, "r7_transfer/msls/shared")
        self.assertEqual({p.dataset for p in found if p.spec.key == "rA_apgd"}, {"msls", "nordland"})
        self.assertEqual({p.tags for p in found if p.spec.key == "r9_apgd_steps500"}, {("mplc_s0",)})

    def test_eval_args(self):
        found = {p.id: p for p in step3.pieces(self.res, 25)}
        args = step3.eval_args(found["r3_targeted_tr25/sped/mplc_s0"], self.res, self.root, 25)
        for flag in ("--rank_attack_goal=targeted", "--target_rank=25", "--max_queries=1000", "mplc_s0"):
            self.assertIn(flag, args)
        self.assertEqual(args[args.index("--epsilons") + 1], "0.0685")
        self.assertIn(f"--output_json={self.root}/r3_targeted_tr25/sped/mplc_s0/rank_eval_results.json", args)
        args = step3.eval_args(found["r7_transfer/msls/shared"], self.res, self.root, 25)
        self.assertIn("--shared_attacks", args)
        self.assertEqual(args[args.index("--model_tags") + 1:args.index("--model_tags") + 4], ["pretrained", "plain_at_s0", "mplc_s0"])
        args = step3.eval_args(found["r9b_apgd_r3/msls/mplc_s0"], self.res, self.root, 25)
        self.assertIn("--rank_restarts=3", args)
        self.assertNotIn("--target_rank=25", args)
        args = step3.eval_args(found["r6_l2/nordland/plain_at_s0"], self.res, self.root, 25)
        self.assertIn("--rank_attack=rank_pgd_l2", args)
        self.assertEqual(args[args.index("--epsilons") + 1:args.index("--epsilons") + 3], ["4.37", "8.73"])

    def test_record_picks_the_complete_attempt_and_estimates_use_it(self):
        piece = next(p for p in step3.pieces(self.res, 10) if p.id == "r1_apgd/msls/mplc_s0")
        default = step3.estimate_seconds(piece, {})
        self.assertAlmostEqual(default, 3600 * (4.3 + 0.33) * 1000 / 3607)
        directory = piece.directory(self.root)
        (directory / "2026-10-12_01-00-00").mkdir(parents=True)  # killed attempt
        complete = directory / "2026-10-12_02-00-00"
        complete.mkdir()
        for file in step3.RESULT_FILES:
            (complete / file).write_text("{}", encoding="utf-8")
        (directory / "2026-10-12_03-00-00").mkdir()  # an attempt after it, also incomplete
        self.assertEqual(step3.record_done(self.root, piece, 1000), complete)
        self.assertEqual(step3.done_info(self.root, piece)["result_dir"], str(complete))
        measured = step3.measured_per_model(self.root)
        self.assertEqual(measured, {("r1_apgd", "msls"): 1000.0})
        self.assertAlmostEqual(step3.estimate_seconds(piece, measured), 1200.0)

    def test_record_refuses_without_a_complete_result(self):
        piece = step3.pieces(self.res, 10)[0]
        (piece.directory(self.root) / "2026-10-12_01-00-00").mkdir(parents=True)
        with self.assertRaisesRegex(SystemExit, "no complete"):
            step3.record_done(self.root, piece, 10)

    def test_models_are_frozen_at_the_first_start(self):
        step3.freeze_models(self.root, self.res)
        step3.freeze_models(self.root, self.res)
        write_run(Path(self.tmp.name) / "s2", MPLC, 60, 60)
        with self.assertRaisesRegex(SystemExit, "other checkpoints"):
            step3.freeze_models(self.root, step3.resolve(Path(self.tmp.name) / "s2"))


def fake_eval(argv):
    """Write eval.py's three result files for every model, condition and epsilon of a call."""
    def values(flag):
        i = argv.index(flag) + 1
        out = []
        while i < len(argv) and not argv[i].startswith("--"):
            out.append(argv[i])
            i += 1
        return out

    options = dict(a[2:].split("=", 1) for a in argv if a.startswith("--") and "=" in a)
    dataset, tags, epsilons = values("--datasets")[0], values("--model_tags"), values("--epsilons")
    attack, goal = options["rank_attack"], options["rank_attack_goal"]
    run_dir = Path(options["output_json"]).parent / "2026-10-12_00-00-00"
    run_dir.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "model", "condition", "attack", "epsilon", "R@1", "targeted_success_rate"]
    rows, per_query = [], []
    for tag in tags:
        clean = 90.0 if tag == "pretrained" else 80.0
        rows.append({"dataset": dataset, "model": tag, "condition": "clean_all_queries", "R@1": clean})
        rows.append({"dataset": dataset, "model": tag, "condition": "clean_attacked_subset", "R@1": clean})
        for eps in epsilons:
            r1 = 0.0 if tag == "pretrained" else max(0.0, 50.0 - 100.0 * float(eps))
            if attack == "rank_apgd_linf":
                r1 = max(0.0, r1 - 10.0)
            condition = f"{attack}_eps_{float(eps):g}" + ("_targeted" if goal == "targeted" else "")
            rows.append({"dataset": dataset, "model": tag, "condition": condition, "attack": attack, "epsilon": eps,
                         "R@1": r1, "targeted_success_rate": (90.0 if tag == "pretrained" else 30.0) if goal == "targeted" else ""})
            correct = round(r1 / 10)
            per_query += [{"query_id": q, "dataset": dataset, "model_tag": tag, "checkpoint_tag": tag, "condition": condition,
                           "epsilon": eps, "clean_rank": 0, "attacked_rank": 0 if q < correct else 5,
                           "clean_correct_at_1": True, "attacked_correct_at_1": q < correct} for q in range(10)]
    (run_dir / "rank_eval_results.json").write_text("{}", encoding="utf-8")
    for name, data, names in (("rank_eval_results.csv", rows, fields), ("per_query_ranks.csv", per_query, list(per_query[0]))):
        with (run_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(data)


def finish(root, res, target_rank, which=lambda piece: True):
    for piece in step3.pieces(res, target_rank):
        if which(piece):
            fake_eval(step3.eval_args(piece, res, root, target_rank))
            step3.record_done(root, piece, 60)


def plan_next(root, s2, *extra):
    out = io.StringIO()
    with redirect_stdout(out):
        step3.main(["next", "--root", str(root), "--step2-root", str(s2), *extra])
    return out.getvalue().splitlines()


class NextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.s2 = write_step2(Path(self.tmp.name) / "s2")
        self.res = step3.resolve(self.s2)
        self.root = Path(self.tmp.name) / "s3"

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_piece_is_the_pilot_and_models_are_frozen(self):
        lines = plan_next(self.root, self.s2)
        self.assertEqual(lines[0], "PIECE pilot_tr10/sped/pretrained")
        self.assertEqual(lines[2], str(self.root / "pilot_tr10/sped/pretrained"))
        self.assertEqual(lines[3:5], ["--datasets", "sped"])
        self.assertTrue((self.root / step3.MODELS_FILE).is_file())

    def test_blocked_step2_stops(self):
        write_run(self.s2, HINGE, 54, 54, state="interrupted")
        self.assertTrue(plan_next(self.root, self.s2)[0].startswith("STOP Step 3 needs every run finished"))
        self.assertFalse((self.root / step3.MODELS_FILE).exists())

    def test_a_piece_that_does_not_fit_gives_way_to_a_shorter_one(self):
        finish(self.root, self.res, 10, lambda p: p.spec.key.startswith("pilot"))
        lines = plan_next(self.root, self.s2, "--time-left", "3600")
        self.assertEqual(lines[0], "PIECE r1_apgd/sped/pretrained")  # MSLS and Pitts30k (~1.3 h) do not fit
        self.assertTrue(plan_next(self.root, self.s2, "--time-left", "60")[0].startswith("WAIT"))

    def test_pilot_rule(self):
        def state(rates):
            results = step3.Results(self.root, [])
            results.attacked = lambda run, dataset, tag, eps, field="R@1": rates[tag]
            return step3.pilot_state(self.res, results, 10)[0]

        self.assertEqual(state({"pretrained": 99.0, "plain_at_s0": 40.0, "mplc_s0": 20.0}), "pass")
        self.assertEqual(state({"pretrained": 99.0, "plain_at_s0": 96.0, "mplc_s0": 95.0}), "fail")
        self.assertEqual(state({"pretrained": 5.0, "plain_at_s0": 0.0, "mplc_s0": 1.0}), "fail")
        self.assertEqual(state({"pretrained": 99.0, "plain_at_s0": None, "mplc_s0": 1.0}), "pending")

    def test_failed_pilot_holds_run_3_back_until_another_target_rank(self):
        finish(self.root, self.res, 10, lambda p: p.spec.key.startswith("pilot"))
        # Rewrite the pilot so every model is at 100 %: run 3 must wait, run 1 goes on.
        for piece in step3.pieces(self.res, 10):
            if piece.spec.key.startswith("pilot"):
                path = Path(step3.done_info(self.root, piece)["result_dir"]) / "rank_eval_results.csv"
                rows = step3.read_csv(path)
                for row in rows:
                    if row.get("targeted_success_rate"):
                        row["targeted_success_rate"] = "100.0"
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        self.assertEqual(plan_next(self.root, self.s2)[0], "PIECE r1_apgd/msls/pretrained")
        finish(self.root, self.res, 10, lambda p: not p.spec.key.startswith(("pilot", "r3_")))
        stop = plan_next(self.root, self.s2)[0]
        self.assertTrue(stop.startswith("STOP targeted pilot failed at target rank 10"), stop)
        self.assertIn("STEP3_TARGET_RANK=50", stop)
        # A new target rank: its pilot runs first, then run 3 at that rank.
        self.assertEqual(plan_next(self.root, self.s2, "--target-rank", "50")[0], "PIECE pilot_tr50/sped/pretrained")
        finish(self.root, self.res, 50, lambda p: p.spec.key.startswith("pilot"))
        self.assertEqual(plan_next(self.root, self.s2, "--target-rank", "50")[0], "PIECE r3_targeted_tr50/msls/pretrained")

    def test_done_when_every_piece_is_done(self):
        finish(self.root, self.res, 10)
        self.assertEqual(plan_next(self.root, self.s2), ["DONE"])


class SummaryTests(unittest.TestCase):
    def test_tables_from_finished_pieces(self):
        with tempfile.TemporaryDirectory() as tmp:
            s2 = write_step2(Path(tmp) / "s2")
            root = Path(tmp) / "s3"
            res = step3.resolve(s2)
            finish(root, res, 10)
            step3.write_summary(root, res, step3.pieces(res, 10), 10)
            out = root / "summary"
            table_a = {(r["dataset"], r["model"]): r for r in step3.read_csv(out / "table_budget5.csv")}
            table_b = {(r["dataset"], r["model"]): r for r in step3.read_csv(out / "table_best.csv")}
            ablations_b = {(r["dataset"], r["model"]): r for r in step3.read_csv(out / "ablations_best.csv")}
            paired = step3.read_csv(out / "paired.csv")
            validity = (out / "validity.md").read_text()
            status = (out / "status.md").read_text()
            curves = step3.read_csv(out / "curves_pgd.csv")

        row = table_a["msls", "MPLC*"]
        self.assertEqual((row["checkpoint"], row["clean R@1"], row["clean drop"]), ("63", "80.00", "10.00"))
        self.assertEqual((row["APGD R@1"], row["PGD-20 R@1"], row["targeted success %"]), ("33.15", "43.15", "30.00"))
        self.assertEqual(row["worst-case R@1"], "30.00")  # 3 / 10 fake queries survive every attack
        self.assertEqual((row["L2 R@1 (1 px)"], row["L2 R@1 (2 px)"]), ("0.00", "0.00"))
        self.assertNotEqual(row["PGD-20 AUC"], "–")
        # plain_at has no budget-5 checkpoint: Table A shows the pretrained model, Table B its best.
        self.assertEqual((table_a["sped", "plain_at"]["checkpoint"], table_a["sped", "plain_at"]["APGD R@1"]), ("initial", "0.00"))
        self.assertIn("5-point budget", table_a["sped", "plain_at"]["note"])
        self.assertEqual((table_b["sped", "plain_at"]["checkpoint"], table_b["sped", "plain_at"]["APGD R@1"]), ("12", "33.15"))
        self.assertEqual(ablations_b["nordland", "MPLC* without: no anchor"]["checkpoint"], "96")
        self.assertEqual(ablations_b["nordland", "MPLC* without: single positive"]["APGD R@1"], "33.15")
        self.assertIn(("msls", "MPLC* without: no perceptual attacks"), ablations_b)
        self.assertEqual(len(ablations_b), 2 * 5)  # MPLC*, 3 ablation runs, 1 covered
        # Paired: Table A compares MPLC* with the pretrained weights plain_at falls back to.
        self.assertEqual({(r["table"], r["model_b"]) for r in paired}, {("budget5", "pretrained"), ("best", "plain_at_s0")})
        self.assertEqual(paired[0]["mcnemar_p"], f"{step3.mcnemar_p(3, 0):.3g}")
        self.assertNotIn("pending", validity)
        self.assertIn("**fail**", validity)  # fake APGD is 10 points below PGD-20
        self.assertIn("Pieces done: 84 / 84", status)
        self.assertEqual({r["model"] for r in curves if r["dataset"] == "nordland"},
                         {"pretrained", "plain_at_s0", "mplc_s0", "abl_no_anchor", "abl_hinge",
                          "abl_single_pos_b5", "abl_single_pos_best"})

    def test_mcnemar(self):
        self.assertEqual(step3.mcnemar_p(0, 0), 1.0)
        self.assertAlmostEqual(step3.mcnemar_p(5, 0), 2 / 32)
        self.assertEqual(step3.mcnemar_p(10, 10), 1.0)
        self.assertAlmostEqual(step3.mcnemar_p(1, 9), step3.mcnemar_p(9, 1))


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.s2 = write_step2(tmp / "s2")
        self.root = tmp / "s3"
        # PYTHON for the driver: eval.py calls go to fake_eval, everything else to this python.
        self.python = tmp / "python"
        self.python.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "eval.py" ]; then shift; exec "' + sys.executable + '" -c '
            '"import sys; from tests.test_step3_plan import fake_eval; fake_eval(sys.argv[1:])" "$@"; fi\n'
            'exec "' + sys.executable + '" "$@"\n', encoding="utf-8")
        self.python.chmod(self.python.stat().st_mode | stat.S_IEXEC)
        self.env = {"STEP2_ROOT": str(self.s2), "STEP3_ROOT": str(self.root), "PYTHON": str(self.python)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_lists_pieces(self):
        result = run_script(STEP3_SCRIPT, self.env, ["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("\n=== "), 84, result.stdout)
        self.assertIn("--test_method=hard_resize", result.stdout.splitlines()[0])
        self.assertFalse(self.root.exists())

    def test_session_runs_every_piece_and_resumes(self):
        # The 500-step APGD piece is estimated at ~6 h before it is measured.
        env = {**self.env, "STEP3_STOP_AT": "+12 hours"}
        result = run_script(STEP3_SCRIPT, env)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Step 3 done", result.stdout)
        self.assertIn("Pieces done: 84 / 84", (self.root / "summary" / "status.md").read_text())
        result = run_script(STEP3_SCRIPT, env)  # nothing left: done at once
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("=== Step 3 piece", result.stdout)

    def test_unfinished_step2_exits_3(self):
        write_run(self.s2, SINGLE_POS, 30, 30, state="interrupted")
        result = run_script(STEP3_SCRIPT, {**self.env, "STEP3_STOP_AT": "+3 hours"})
        self.assertEqual(result.returncode, 3, result.stdout)
        self.assertIn(SINGLE_POS, result.stdout)

    def test_stop_at_is_required_and_checked(self):
        self.assertEqual(run_script(STEP3_SCRIPT, self.env).returncode, 2)
        self.assertEqual(run_script(STEP3_SCRIPT, {**self.env, "STEP3_STOP_AT": "+30 minutes"}).returncode, 2)
        self.assertEqual(run_script(STEP3_SCRIPT, {**self.env, "STEP3_STOP_AT": "soon"}).returncode, 2)
        self.assertEqual(run_script(STEP3_SCRIPT, args=["--bogus"]).returncode, 2)

    def test_summary_only(self):
        result = run_script(STEP3_SCRIPT, self.env, ["--summary-only"])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue((self.root / "summary" / "table_budget5.md").is_file())


class PlainAtDriverTests(unittest.TestCase):
    def test_dry_run_prints_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            s2 = write_step2(Path(tmp) / "s2", skip=(PLAIN_AT,))
            result = run_script(PLAIN_AT_SCRIPT, {"STEP2_ROOT": str(s2), "STEP1_ROOT": str(Path(tmp) / "s1")}, ["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Step 2 is not done", result.stdout)  # no Step 1 here, so the planner cannot say DONE
        self.assertIn(f"--save_dir={PLAIN_AT}", result.stdout.split())
        self.assertIn(f"--run_dir={s2}/{PLAIN_AT}", result.stdout.split())

    def test_unfinished_step2_exits_3_and_options_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            s2 = write_step2(Path(tmp) / "s2", skip=(PLAIN_AT,))
            env = {"STEP2_ROOT": str(s2), "STEP1_ROOT": str(Path(tmp) / "s1")}
            self.assertEqual(run_script(PLAIN_AT_SCRIPT, env).returncode, 2)  # no PLAIN_AT_STOP_AT
            result = run_script(PLAIN_AT_SCRIPT, {**env, "PLAIN_AT_STOP_AT": "+3 hours"})
            self.assertEqual(result.returncode, 3, result.stdout)
            self.assertFalse((s2 / PLAIN_AT).exists())
        self.assertEqual(run_script(PLAIN_AT_SCRIPT, args=["--bogus"]).returncode, 2)


if __name__ == "__main__":
    unittest.main()
