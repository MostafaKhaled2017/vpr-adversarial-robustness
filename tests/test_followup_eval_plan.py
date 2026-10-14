import importlib.util
import io
import json
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src import followup_eval_plan as fu
from src import step3_plan as step3
from tests.test_step1_sweep import REPO_ROOT, run_script
from tests.test_step3_plan import fake_eval

SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_followup_eval.sh"
# The summary's PGD-20 AUC, curves and worst case come from src.robustness_curve_summary, which
# imports torch; a summary over runs 1 and T only (no grid, L2, targeted or embedding-shift
# results) does not reach it.
needs_torch = unittest.skipUnless(importlib.util.find_spec("torch") is not None,
                                  "needs torch: src.robustness_curve_summary imports it")


def record(epoch, clean, robust, budgets=(1.0, 3.0, 5.0)):
    return {"epoch": epoch, "clean": {"R@1": clean}, "robust_score": robust, "eligible_budgets": list(budgets)}


INITIAL = record(-1, 88.0, 0.0)
SAME = [INITIAL, record(3, 86.0, 10.0, (3.0, 5.0)), record(6, 84.0, 30.0, (5.0,)), record(9, 80.0, 25.0, ())]  # A = B = 6
SPLIT = [INITIAL, record(3, 85.0, 20.0, (5.0,)), record(6, 80.0, 40.0, ())]  # A = 3, B = 6
NO_BUDGET = [INITIAL, record(3, 80.0, 40.0, ())]  # A: none (pretrained), B = 3
RUNS = {"sp": SAME, "sp_s1": SPLIT, "sp_eps2": SAME, "fare_eps2": NO_BUDGET, "plain_at_eps2": SAME,
        "sp_hinge": SAME, "sp_aw2": SPLIT, "sp_aw3": SAME, "twin": SAME}
MAIN_TAGS = ["pretrained", "sp", "sp_s1_b5", "sp_s1_best", "sp_eps2", "fare_eps2_best", "plain_at_eps2"]
ABLATION_TAGS = ["sp_hinge", "sp_aw2_b5", "sp_aw2_best", "sp_aw3"]
RUN_ORDER = ["r1_apgd", "r3_targeted_tr10", "r4_embshift", "r2_pgd_grid", "clean_twin", "r7_transfer", "r8_large_eps",
             "r9_apgd_steps500", "r9b_apgd_r3", "r6_l2", "rA_apgd", "rA_pgd_grid"]
# 7 main tags x 4 datasets for runs 1, 3, 4, 2 and 6; the twin on 4; 1 transfer; 6 defended at
# 32/255; sp for 9 and 9b; 4 ablation tags x 2 datasets for A twice.
COUNTS = {"r1_apgd": 28, "r3_targeted_tr10": 28, "r4_embshift": 28, "r2_pgd_grid": 28, "clean_twin": 4, "r7_transfer": 1,
          "r8_large_eps": 6, "r9_apgd_steps500": 1, "r9b_apgd_r3": 1, "r6_l2": 28, "rA_apgd": 8, "rA_pgd_grid": 8}
ALL_PIECES = sum(COUNTS.values())  # 169


def write_run(run_dir, records, state="early_stopped", files=("best_model.pth", "best_model_budget5.pth", "last_model.pth")):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    (run_dir / "validation_recalls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    for name in files:
        (run_dir / name).write_bytes(b"")
    return run_dir


class Layout:
    """Follow-up and Step 2 runs for every row (RUNS) except skip, a stand-in pretrained checkpoint
    and an evaluation root, all under tmp."""

    def __init__(self, tmp, skip=(), states=None):
        tmp = Path(tmp)
        self.followup, self.step2, self.root = tmp / "followup", tmp / "step2", tmp / "eval"
        self.pretrained = tmp / "SuperVLAD_base.pth"
        self.pretrained.write_bytes(b"")
        for key, records in RUNS.items():
            if key not in skip:
                write_run(self.run(key), records, (states or {}).get(key, "early_stopped"))

    def run(self, key):
        return fu.run_directory(fu.MODELS[key], self.followup, self.step2)

    def flags(self):
        return ["--root", str(self.root), "--followup-root", str(self.followup), "--step2-root", str(self.step2),
                "--pretrained", str(self.pretrained)]

    def env(self):
        return {"FOLLOWUP_EVAL_ROOT": str(self.root), "FOLLOWUP_ROOT": str(self.followup),
                "FOLLOWUP_STEP2_ROOT": str(self.step2), "FOLLOWUP_EVAL_PRETRAINED": str(self.pretrained)}

    def plan(self, command, *extra):
        out = io.StringIO()
        with redirect_stdout(out):
            fu.main([command, *self.flags(), *extra])
        return out.getvalue().splitlines()

    def resolution(self):
        return fu.resolve(step3.recorded(self.root), str(self.pretrained))


def finish(root, res, which=lambda piece: True):
    """Run every chosen piece with Step 3's fake eval.py and mark it done (no summary)."""
    for piece in fu.pieces(res, 10):
        if which(piece):
            fake_eval(step3.eval_args(piece, res, root, 10))
            step3.record_done(root, piece, 60)


class ChoiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def choose(self, records, key="sp", **kwargs):
        run_dir = write_run(self.tmp_path / key, records, **kwargs)
        return run_dir, fu.choose(fu.MODELS[key], run_dir)

    def test_the_three_checkpoint_cases(self):
        run_dir, choice = self.choose(SAME)
        self.assertEqual(choice.checkpoints, (step3.Checkpoint("sp", "AB", str(run_dir / "best_model.pth")),))
        self.assertEqual(choice.note, "budget-5 epoch 6, best epoch 6")
        run_dir, choice = self.choose(SPLIT, key="sp_s1")
        self.assertEqual(choice.checkpoints, (step3.Checkpoint("sp_s1", "A", str(run_dir / "best_model_budget5.pth")),
                                              step3.Checkpoint("sp_s1", "B", str(run_dir / "best_model.pth"))))
        run_dir, choice = self.choose(NO_BUDGET, key="fare_eps2")
        self.assertEqual(choice.checkpoints, (step3.Checkpoint("fare_eps2", "A", "pretrained"),
                                              step3.Checkpoint("fare_eps2", "B", str(run_dir / "best_model.pth"))))
        self.assertIn("budget-5 epoch none", choice.note)

    def test_the_twin_is_its_last_model(self):
        run_dir, choice = self.choose(SAME, key="twin")
        self.assertEqual(choice.checkpoints, (step3.Checkpoint("twin", "AB", str(run_dir / "last_model.pth")),))

    def test_what_the_trainer_keeps(self):
        # No trained epoch beats the initial validation: both files hold the initial model.
        _, choice = self.choose([record(-1, 88.0, 5.0), record(3, 87.0, 4.0, (5.0,))])
        self.assertEqual(choice.checkpoints, (step3.Checkpoint("sp", "AB", "pretrained"),))
        # A tied best robust score: the trainer keeps the first tied epoch, so one file per table.
        run_dir, choice = self.choose([INITIAL, record(3, 85.0, 40.0, (5.0,)), record(6, 86.0, 40.0, (5.0,))], key="sp_aw2")
        self.assertEqual([c.tables for c in choice.checkpoints], ["A", "B"])
        self.assertIn("tied", choice.note)
        # A resumed epoch replaces the earlier record of it.
        _, choice = self.choose([*SPLIT, record(6, 85.0, 45.0, (5.0,))], key="sp_aw3")
        self.assertEqual([c.tables for c in choice.checkpoints], ["AB"])

    def test_untrained_runs_are_skipped(self):
        for state in ("training", "paused", "failed"):
            self.assertIsNone(self.choose(SAME, state=state)[1], state)
        self.assertIsNone(fu.choose(fu.MODELS["sp_s1"], self.tmp_path / "absent"))
        self.assertEqual(fu.run_state(self.tmp_path / "absent"), "not started")
        _, choice = self.choose(SAME, key="sp_hinge", state="collapse_aborted")
        self.assertIsNotNone(choice)

    def test_a_trained_run_without_its_checkpoint_file_is_refused(self):
        with self.assertRaises(step3.Refused) as caught:
            self.choose(SPLIT, files=("best_model.pth",))
        self.assertEqual(caught.exception.code, 3)
        self.assertIn("best_model_budget5.pth", str(caught.exception))
        with self.assertRaises(step3.Refused) as caught:
            self.choose(SAME, key="twin", files=("best_model.pth",))
        self.assertIn("last_model.pth", str(caught.exception))


class RegisterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_untrained_rows_are_skipped_and_reported(self):
        layout = Layout(self.tmp.name, skip=("sp_s1",), states={"sp_eps2": "training"})
        lines = layout.plan("register")
        self.assertIn(f"# sp_s1 not trained yet, skipped: {layout.run('sp_s1')}: not started", lines)
        self.assertIn(f"# sp_eps2 not trained yet, skipped: {layout.run('sp_eps2')}: training", lines)
        self.assertEqual({c.key for c in step3.recorded(layout.root)}, set(fu.MODELS) - {"sp_s1", "sp_eps2"})
        res = layout.resolution()
        self.assertEqual(fu.transfer_waits_for(res), ["sp_s1", "sp_eps2"])
        # Every piece of the recorded rows done: the planner says which rows are still untrained.
        finish(layout.root, res)
        lines = layout.plan("next")
        self.assertEqual(lines[0], "UNTRAINED sp_s1 sp_eps2")
        self.assertEqual(lines[2], f"sp_eps2: {layout.run('sp_eps2')}: training")
        # sp_eps2 finishes training: the driver records it (register) and evaluates it.
        write_run(layout.run("sp_eps2"), SAME)
        self.assertEqual(layout.plan("next"), ["REGISTER sp_eps2"])
        self.assertIn(f"model sp_eps2:AB:{layout.run('sp_eps2') / 'best_model.pth'} (new)", layout.plan("register"))
        self.assertEqual(layout.plan("next")[0], "PIECE r1_apgd/msls/sp_eps2")
        write_run(layout.run("sp_s1"), SPLIT)
        layout.plan("register")
        res = layout.resolution()
        self.assertEqual(fu.transfer_waits_for(res), [])
        finish(layout.root, res, lambda p: step3.done_info(layout.root, p) is None)
        self.assertEqual(layout.plan("next"), ["DONE"])
        self.assertEqual(len(step3.read_csv(layout.root / step3.DURATIONS_FILE)), ALL_PIECES)  # no piece ran twice

    def test_an_override_wins_and_conflicts_are_refused(self):
        layout = Layout(self.tmp.name)
        last = str(layout.run("sp") / "last_model.pth")
        lines = layout.plan("register", "--model", f"sp:ab:{last}")
        self.assertIn(f"model sp:AB:{last} (new)", lines)
        recorded = step3.recorded(layout.root)
        self.assertEqual([c for c in recorded if c.key == "sp"], [step3.Checkpoint("sp", "AB", last)])
        self.assertEqual(len(recorded), 12)  # sp_s1, fare_eps2 and sp_aw2 take two checkpoints
        # A recorded row is not chosen again, so the override stays without being repeated.
        self.assertTrue(all(not line.endswith("(new)") for line in layout.plan("register")))
        self.assertEqual(step3.recorded(layout.root), recorded)
        for spec, code in ((f"sp:B:{layout.run('sp') / 'best_model.pth'}", 3), ("sp:AB:missing.pth", 2),
                           (f"mplc:AB:{last}", 2), (f"twin:A:{last}", 2), ("sp:AB", 2)):
            with self.assertRaises(SystemExit) as caught, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                layout.plan("register", "--model", spec)
            self.assertEqual(caught.exception.code, code, spec)
        self.assertEqual(step3.recorded(layout.root), recorded)

    def test_models_subset_and_unknown_keys(self):
        layout = Layout(self.tmp.name, skip=("sp_hinge",))
        lines = layout.plan("register", "--keys", "twin sp")
        self.assertEqual([line for line in lines if line.startswith("model")],
                         [f"model sp:AB:{layout.run('sp') / 'best_model.pth'} (new)",
                          f"model twin:AB:{layout.run('twin') / 'last_model.pth'} (new)"])
        self.assertNotIn("sp_hinge", "\n".join(lines))  # not requested, so not reported as untrained
        self.assertEqual(fu.pieces(layout.resolution(), 10)[-1].id, "r6_l2/nordland/sp")  # no ablations
        self.assertEqual(layout.plan("next", "--keys", "twin sp")[0], "PIECE r1_apgd/msls/pretrained")
        for keys in ("sp nope", "SP"):
            with self.assertRaises(SystemExit) as caught, redirect_stderr(io.StringIO()):
                layout.plan("register", "--keys", keys)
            self.assertEqual(caught.exception.code, 2)
        # The default adds the other trained rows to the recorded ones.
        lines = layout.plan("register")
        self.assertEqual(sum(line.endswith("(new)") for line in lines), 9)
        self.assertIn(f"# sp_hinge not trained yet, skipped: {layout.run('sp_hinge')}: not started", lines)


class PieceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = Layout(self.tmp.name)
        self.layout.plan("register")
        self.res = self.layout.resolution()
        self.pieces = fu.pieces(self.res, 10)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, piece_id):
        piece = next(p for p in self.pieces if p.id == piece_id)
        return step3.eval_args(piece, self.res, self.layout.root, 10)

    def test_order_and_model_sets(self):
        counts = {}
        for piece in self.pieces:
            counts[piece.spec.key] = counts.get(piece.spec.key, 0) + 1
        self.assertEqual(counts, COUNTS)
        self.assertEqual(list(dict.fromkeys(p.spec.key for p in self.pieces)), RUN_ORDER)
        self.assertEqual(len(self.pieces), ALL_PIECES)
        self.assertEqual(self.res.tags("main"), MAIN_TAGS)
        self.assertEqual(self.res.tags("ablation"), ABLATION_TAGS)
        by_run = {}
        for piece in self.pieces:
            by_run.setdefault(piece.spec.key, []).append(piece)
        for run in ("r1_apgd", "r3_targeted_tr10", "r4_embshift", "r2_pgd_grid", "r6_l2"):
            self.assertEqual({p.tags[0] for p in by_run[run]}, set(MAIN_TAGS), run)
            self.assertEqual({p.dataset for p in by_run[run]}, set(step3.DATASETS), run)
        for run in ("rA_apgd", "rA_pgd_grid"):
            self.assertEqual({p.tags[0] for p in by_run[run]}, set(ABLATION_TAGS), run)
            self.assertEqual({p.dataset for p in by_run[run]}, {"msls", "nordland"}, run)
        self.assertEqual([p.id for p in by_run["clean_twin"]], [f"clean_twin/{d}/twin" for d in step3.DATASETS])
        self.assertEqual({p.tags for p in by_run["r9_apgd_steps500"] + by_run["r9b_apgd_r3"]}, {("sp",)})
        self.assertEqual(by_run["r7_transfer"][0].tags, tuple(MAIN_TAGS))  # crafted on pretrained, replayed on the rest
        self.assertEqual({p.tags[0] for p in by_run["r8_large_eps"]}, set(MAIN_TAGS[1:]))
        self.assertNotIn("twin", {t for p in self.pieces if p.spec.key != "clean_twin" for t in p.tags})

    def test_eval_args(self):
        args = self.args("r1_apgd/pitts30k/sp")
        self.assertEqual(args[args.index("--epsilons") + 1:args.index("--epsilons") + 3], ["0.03425", "0.0685"])
        self.assertIn("--rank_attack=rank_apgd_linf", args)
        self.assertIn("--rank_steps=100", args)
        args = self.args("clean_twin/sped/twin")
        self.assertEqual(args[args.index("--model_paths") + 1], str(self.layout.run("twin") / "last_model.pth"))
        out = self.layout.root / "clean_twin/sped/twin"
        self.assertEqual(args[args.index("--epsilons") + 1:], ["0.01712", f"--output_json={out}/rank_eval_results.json",
                                                               f"--output_csv={out}/rank_eval_results.csv"])
        for flag in ("--rank_attack=rank_pgd_linf", "--rank_steps=20", "--max_queries=1000"):
            self.assertIn(flag, args)
        args = self.args("r3_targeted_tr10/msls/fare_eps2_best")
        self.assertIn("--target_rank=10", args)
        self.assertEqual(args[args.index("--model_paths") + 1], str(self.layout.run("fare_eps2") / "best_model.pth"))
        args = self.args("rA_apgd/nordland/sp_aw2_b5")
        self.assertEqual(args[args.index("--model_paths") + 1], str(self.layout.run("sp_aw2") / "best_model_budget5.pth"))
        self.assertIn("--shared_attacks", self.args("r7_transfer/msls/shared"))
        self.assertIn("--rank_restarts=3", self.args("r9b_apgd_r3/msls/sp"))

    def test_estimates(self):
        piece = next(p for p in self.pieces if p.id == "r1_apgd/msls/sp")
        self.assertAlmostEqual(fu.estimate_seconds(piece, {}), 3600 * (2 * 4.3 + 0.33) * 1000 / 3607)
        piece = next(p for p in self.pieces if p.id == "r9_apgd_steps500/msls/sp")
        self.assertAlmostEqual(fu.estimate_seconds(piece, {}), step3.default_seconds(piece))
        self.assertAlmostEqual(fu.estimate_seconds(piece, {("r9_apgd_steps500", "msls"): 100.0}), 120.0)


class NextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = Layout(self.tmp.name)
        self.layout.plan("register")

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_piece_and_a_shorter_one_when_it_does_not_fit(self):
        lines = self.layout.plan("next")
        self.assertEqual(lines[0], "PIECE r1_apgd/msls/pretrained")
        self.assertEqual(lines[2], str(self.layout.root / "r1_apgd/msls/pretrained"))
        self.assertEqual(lines[3:5], ["--datasets", "msls"])
        # Run 1 (~2.5 h per model on MSLS) does not fit in an hour; run 3 (~0.3 h) does.
        self.assertEqual(self.layout.plan("next", "--time-left", "3600")[0], "PIECE r3_targeted_tr10/msls/pretrained")
        self.assertTrue(self.layout.plan("next", "--time-left", "60")[0].startswith("WAIT"))

    def test_done_missing_checkpoint_and_target_rank(self):
        res = self.layout.resolution()
        finish(self.layout.root, res)
        self.assertEqual(self.layout.plan("next"), ["DONE"])
        stop = self.layout.plan("next", "--target-rank", "20")[0]
        self.assertTrue(stop.startswith("STOP run 3 has finished pieces at target rank 10"), stop)
        self.assertIn("FOLLOWUP_EVAL_TARGET_RANK=10", stop)
        (self.layout.run("sp_aw2") / "best_model_budget5.pth").unlink()
        stop = self.layout.plan("next")[0]
        self.assertTrue(stop.startswith("STOP recorded checkpoint files are missing"), stop)
        self.assertIn("best_model_budget5.pth", stop)


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.layout = Layout(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def summary(self, res, plan=None):
        fu.write_summary(self.layout.root, res, fu.pieces(res, 10), 10, plan)
        return self.layout.root / "summary"

    def test_an_empty_root(self):
        out = self.summary(fu.resolve([], str(self.layout.pretrained)),
                           fu.Plan([], [], untrained={"sp": "not started"}, unrecorded={"twin": "early_stopped"}))
        for name in ("status.md", "table_budget5.md", "table_best.csv", "ablations_best.md", "clean_twin.csv",
                     "curves_pgd.csv", "curves_l2.csv", "worst_case.csv", "paired.csv", "validity.md"):
            self.assertTrue((out / name).is_file(), name)
        status = (out / "status.md").read_text()
        self.assertIn("Pieces done: 0 / 20", status)  # the pretrained model alone in runs 1, 3, 4, 2 and 6
        self.assertIn("Not trained yet: sp;", status)
        self.assertIn("Trained, not recorded yet: twin;", status)
        self.assertIn("| sp_aw2 | anchor weight 2 | ablation | mplc_v2_supervlad_mplc_aw2_mixlinf_sp_fte0_s0 | "
                      "not requested (FOLLOWUP_EVAL_MODELS) |", status)

    def test_run_1_and_the_twin(self):
        self.layout.plan("register")
        res = self.layout.resolution()
        finish(self.layout.root, res, lambda p: p.spec.key in ("r1_apgd", "clean_twin"))
        out = self.summary(res)
        table_a = {(r["dataset"], r["model"]): r for r in step3.read_csv(out / "table_budget5.csv")}
        twin = {(r["dataset"], r["tag"]): r for r in step3.read_csv(out / "clean_twin.csv")}
        paired = step3.read_csv(out / "paired.csv")
        validity = (out / "validity.md").read_text()
        status = (out / "status.md").read_text()

        self.assertEqual(list(table_a["msls", fu.MODELS["sp"].label])[:8],
                         ["dataset", "model", "checkpoint", "clean R@1", "clean drop", "APGD R@1", "APGD R@1 (2/255)",
                          "worst-case R@1"])
        row = table_a["msls", fu.MODELS["sp"].label]
        self.assertEqual((row["APGD R@1"], row["APGD R@1 (2/255)"]),
                         (format(50.0 - 100.0 * 0.0685 - 10.0, ".2f"), format(50.0 - 100.0 * 0.03425 - 10.0, ".2f")))
        self.assertEqual((row["clean R@1"], row["clean drop"], row["PGD-20 R@1"]), ("80.00", "10.00", "–"))
        self.assertEqual(table_a["sped", fu.MODELS["fare_eps2"].label]["checkpoint"], "initial")
        self.assertEqual(twin["nordland", "twin"]["clean R@1"], "80.00")
        sp = twin["nordland", "sp"]
        self.assertEqual((sp["drop vs pretrained"], sp["drop vs twin"]), ("10.00", "0.00"))
        self.assertEqual(twin["msls", "sp"]["tables"], "A, B")
        self.assertEqual(len(twin), 4 * 3)
        # sp against fare_eps2 (Table A: its pretrained fallback), plain_at_eps2 and sp_s1, per table, dataset and eps.
        self.assertEqual({(r["table"], r["model_b"]) for r in paired},
                         {("budget5", "pretrained"), ("budget5", "plain_at_eps2"), ("budget5", "sp_s1_b5"),
                          ("best", "fare_eps2_best"), ("best", "plain_at_eps2"), ("best", "sp_s1_best")})
        self.assertEqual(len(paired), 6 * 4 * 2)
        first = next(r for r in paired if r["model_b"] == "pretrained" and r["epsilon"] == "0.0685")
        self.assertEqual((first["a_only_correct"], first["mcnemar_p"]), ("3", f"{step3.mcnemar_p(3, 0):.3g}"))
        self.assertIn("APGD with 500 steps within 1 of run 1 | sp | msls", validity)
        self.assertIn("below PGD-20 (2/255)", validity)
        self.assertIn(f"Pieces done: 32 / {ALL_PIECES}", status)

    @needs_torch
    def test_every_piece(self):
        self.layout.plan("register")
        res = self.layout.resolution()
        finish(self.layout.root, res)
        out = self.summary(res)
        table_b = {(r["dataset"], r["model"]): r for r in step3.read_csv(out / "table_best.csv")}
        ablations = {(r["dataset"], r["model"]): r for r in step3.read_csv(out / "ablations_best.csv")}
        row = table_b["msls", fu.MODELS["sp"].label]
        self.assertEqual(row["worst-case R@1"], "30.00")  # 3 / 10 fake queries survive every attack
        self.assertNotEqual(row["PGD-20 AUC"], "–")
        self.assertEqual(ablations["nordland", "anchor weight 2"]["APGD R@1"], format(50.0 - 100.0 * 0.0685 - 10.0, ".2f"))
        self.assertEqual(len(ablations), 2 * 4)  # sp, hinge, aw2 (its best checkpoint) and aw3
        self.assertNotIn("pending", (out / "validity.md").read_text())
        self.assertIn(f"Pieces done: {ALL_PIECES} / {ALL_PIECES}", (out / "status.md").read_text())
        curves = step3.read_csv(out / "curves_pgd.csv")
        self.assertEqual({r["model"] for r in curves if r["dataset"] == "nordland"}, set(MAIN_TAGS + ABLATION_TAGS))


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.layout = Layout(tmp)
        # PYTHON for the driver: eval.py calls go to Step 3's fake_eval, everything else to this python.
        # FINISH_RUN: a run_status.json that every eval.py call marks finished (training ends mid-session).
        self.python = tmp / "python"
        self.python.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "eval.py" ] && [ -n "${FINISH_RUN:-}" ]; then echo \'{"state": "early_stopped"}\' > "${FINISH_RUN}"; fi\n'
            'if [ "$1" = "eval.py" ]; then shift; exec "' + sys.executable + '" -c '
            '"import sys; from tests.test_step3_plan import fake_eval; fake_eval(sys.argv[1:])" "$@"; fi\n'
            'exec "' + sys.executable + '" "$@"\n', encoding="utf-8")
        self.python.chmod(self.python.stat().st_mode | stat.S_IEXEC)
        self.env = {**self.layout.env(), "PYTHON": sys.executable}

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_lists_pieces_without_a_session(self):
        result = run_script(SCRIPT, self.env, ["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout.count("\n=== "), ALL_PIECES, result.stdout)
        self.assertIn("--test_method=hard_resize", result.stdout.splitlines()[0])
        self.assertIn(f"# model sp:AB:{self.layout.run('sp') / 'best_model.pth'} (not recorded yet)", result.stdout)
        self.assertIn("+ eval.py --datasets msls --model_paths", result.stdout)
        self.assertFalse(self.layout.root.exists())
        result = run_script(SCRIPT, {**self.env, "FOLLOWUP_EVAL_MODELS": "sp twin"}, ["--dry-run"])
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("# run 7 (transfer) waits for sp_s1 and sp_eps2", result.stdout)
        self.assertNotIn("sp_hinge", result.stdout)

    def test_options_and_stop_at_are_checked(self):
        self.assertEqual(run_script(SCRIPT, self.env).returncode, 2)  # no FOLLOWUP_EVAL_STOP_AT
        self.assertEqual(run_script(SCRIPT, {**self.env, "FOLLOWUP_EVAL_STOP_AT": "+30 minutes"}).returncode, 2)
        self.assertEqual(run_script(SCRIPT, {**self.env, "FOLLOWUP_EVAL_STOP_AT": "soon"}).returncode, 2)
        self.assertEqual(run_script(SCRIPT, self.env, ["--bogus"]).returncode, 2)
        self.assertEqual(run_script(SCRIPT, self.env, ["--dry-run", "--model"]).returncode, 2)
        self.assertEqual(run_script(SCRIPT, self.env, ["--dry-run", "--model", "sp:AB:missing.pth"]).returncode, 2)
        self.assertEqual(run_script(SCRIPT, {**self.env, "FOLLOWUP_EVAL_MODELS": "sp mplc"}, ["--dry-run"]).returncode, 2)
        self.assertEqual(run_script(SCRIPT, {**self.env, "FOLLOWUP_EVAL_TARGET_RANK": "0"}, ["--dry-run"]).returncode, 2)
        self.assertEqual(run_script(SCRIPT, self.env, ["--summary-only", "--model", "sp:AB:pretrained"]).returncode, 2)
        (self.layout.run("sp") / "best_model.pth").unlink()
        result = run_script(SCRIPT, self.env, ["--dry-run"])
        self.assertEqual(result.returncode, 3, result.stdout)
        self.assertIn("checkpoint file is missing", result.stdout)
        help_text = run_script(SCRIPT, args=["-h"])
        self.assertEqual(help_text.returncode, 0)
        self.assertIn("FOLLOWUP_EVAL_STOP_AT", help_text.stdout)
        self.assertNotIn("set -euo", help_text.stdout)

    def test_summary_only_on_an_empty_root(self):
        result = run_script(SCRIPT, self.env, ["--summary-only"])
        self.assertEqual(result.returncode, 0, result.stdout)
        status = (self.layout.root / "summary" / "status.md").read_text()
        self.assertIn("Pieces done: 0 / 20", status)
        self.assertIn("Trained, not recorded yet: sp, sp_s1, sp_eps2", status)  # nothing recorded: --summary-only records nothing
        self.assertFalse((self.layout.root / step3.MODELS_FILE).exists())

    @needs_torch
    def test_session_runs_every_piece_and_waits_for_untrained_rows(self):
        # sp_hinge finishes training during the session, sp_aw3 does not.
        write_run(self.layout.run("sp_hinge"), SAME, state="training")
        write_run(self.layout.run("sp_aw3"), SAME, state="training")
        env = {**self.env, "PYTHON": str(self.python), "FOLLOWUP_EVAL_STOP_AT": "+12 hours",
               "FOLLOWUP_EVAL_MODELS": "sp twin sp_hinge sp_aw3"}
        result = run_script(SCRIPT, {**env, "FINISH_RUN": str(self.layout.run("sp_hinge") / "run_status.json")})
        self.assertEqual(result.returncode, 3, result.stdout)
        self.assertIn("=== Trained since this session started: sp_hinge", result.stdout)
        self.assertIn("not trained yet: sp_aw3", result.stdout)
        self.assertIn("Rerun after scripts/mplc_v2_followup_train.sh finishes them", result.stdout)
        self.assertIn("Pieces done: 51 / 51", (self.layout.root / "summary" / "status.md").read_text())  # 47 + sp_hinge's 4
        write_run(self.layout.run("sp_aw3"), SAME)
        result = run_script(SCRIPT, env)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Follow-up evaluation done", result.stdout)
        self.assertIn("run 7 (transfer) waits for sp_s1", result.stdout)
        self.assertIn("Pieces done: 55 / 55", (self.layout.root / "summary" / "status.md").read_text())
        result = run_script(SCRIPT, {**env, "FOLLOWUP_EVAL_MODELS": ""}, ["--dry-run"])
        self.assertEqual(result.stdout.count("\n=== "), ALL_PIECES - 55, result.stdout)


if __name__ == "__main__":
    unittest.main()
