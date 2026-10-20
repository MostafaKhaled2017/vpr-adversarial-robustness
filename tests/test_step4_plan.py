import contextlib
import fcntl
import io
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from src import step2_plan as step2
from src import step4_plan as step4
from tests.test_followup_plan import commands, record, save_dir
from tests.test_step1_sweep import REPO_ROOT, TRAIN_SCRIPT, run_script

STEP4_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_step4.sh"
EXPECTED = {
    "boq": {
        "araft": "mplc_v2_boq_mplc_mixlinf_sp_fte0_s0",
        "plain_at": "mplc_v2_boq_plain_at_fte0_s0",
        "fare": "mplc_v2_boq_fare_fte0_s0",
        "twin": "mplc_v2_boq_clean_ft_fte0_s0",
        "fare_eps2": "mplc_v2_boq_fare_eps0.0342_fte0_s0",
        "screen_lr1e-5": "mplc_v2_boq_mplc_mixlinf_sp_fte0_ep9_s0",
        "screen_lr3e-6": "mplc_v2_boq_mplc_mixlinf_sp_fte0_lr3e-6_ep9_s0",
    },
    "mixvpr": {
        "araft": "mplc_v2_mixvpr_mplc_mixlinf_sp_s0",
        "plain_at": "mplc_v2_mixvpr_plain_at_s0",
        "fare": "mplc_v2_mixvpr_fare_s0",
        "twin": "mplc_v2_mixvpr_clean_ft_s0",
        "fare_eps2": "mplc_v2_mixvpr_fare_eps0.0342_s0",
        "screen_lr1e-5": "mplc_v2_mixvpr_mplc_mixlinf_sp_ep9_s0",
        "screen_lr3e-6": "mplc_v2_mixvpr_mplc_mixlinf_sp_lr3e-6_ep9_s0",
    },
}
DEFAULT = ["araft", "plain_at", "fare", "twin"]
DATASETS = ["msls", "pitts30k", "sped", "nordland"]


def plan(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        step4.main(list(argv))
    return out.getvalue().splitlines()


def write_run(root, model, key, state="early_stopped", records=None, files=("best_model.pth", "best_model_budget5.pth",
                                                                           "last_model.pth")):
    run_dir = Path(root) / EXPECTED[model][key]
    run_dir.mkdir(parents=True, exist_ok=True)
    if state is not None:
        (run_dir / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    if records is None:  # budget-5 epoch 3, best epoch 6: two checkpoints
        records = [record(-1, 90.0, 0.0, (1, 3, 5, 7, 10)), record(3, 87.0, 30.0, (5, 7, 10)),
                   record(6, 80.0, 45.0, (10,))]
    (run_dir / "validation_recalls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    for name in files:
        (run_dir / name).write_text("", encoding="utf-8")
    return run_dir


def fake_eval(argv):
    """eval.py for the driver tests: writes --output_json; FAKE_EVAL=fail exits 1 instead."""
    import os

    if os.environ.get("FAKE_EVAL") == "fail":
        sys.exit(1)
    out = Path(next(a for a in argv if a.startswith("--output_json=")).split("=", 1)[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"argv": argv}), encoding="utf-8")
    with (out.parent / "eval_calls.txt").open("a", encoding="utf-8") as handle:
        handle.write(" ".join(argv) + "\n")


class RunNameTests(unittest.TestCase):
    def test_names_match_the_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model, names in EXPECTED.items():
                runs = step4.runs(model)
                self.assertEqual(list(runs), list(names))
                for key, env in runs.items():
                    with self.subTest(model=model, key=key):
                        self.assertEqual(step4.run_name(env), names[key])
                        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_DRY_RUN": "1", "MPLC_V2_RUN_ROOT": tmp, **env})
                        self.assertEqual(result.returncode, 0, result.stdout)
                        self.assertEqual(len(commands(result.stdout)), 1, result.stdout)
                        self.assertEqual(save_dir(commands(result.stdout)[0]), names[key])
                        self.assertIn(f"--model={model}", result.stdout.split())
        self.assertEqual(plan("name", "--model", "boq", "araft"), [EXPECTED["boq"]["araft"]])

    def test_full_runs_use_araft_settings(self):
        env = step4.runs("boq")["araft"]
        for key, value in {"MPLC_V2_ARMS": "mplc", "MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_MULTI_POSITIVE": "0",
                           "MPLC_V2_FREEZE_TE": "0", "MPLC_V2_LR": "1e-5", "MPLC_V2_VAL_EVERY": "3",
                           "MPLC_V2_PATIENCE": "6", "MPLC_V2_LR_PLATEAU_PATIENCE": "3"}.items():
            self.assertEqual(env[key], value, key)
        self.assertNotIn("MPLC_V2_FREEZE_TE", step4.runs("mixvpr")["araft"])
        self.assertEqual(step4.runs("boq")["fare"]["MPLC_V2_ALIGN_WEIGHT"], "1.0")


class NextTests(unittest.TestCase):
    def test_default_runs_in_order_then_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = step4.runs("boq")
            pending = plan("next", "--model", "boq", "--root", tmp, "--all")
            self.assertEqual(pending, [step2.env_assignments(runs[k]) for k in DEFAULT])
            write_run(tmp, "boq", "araft")
            write_run(tmp, "boq", "plain_at", state="paused")
            self.assertEqual(plan("next", "--model", "boq", "--root", tmp), [step2.env_assignments(runs["plain_at"])])
            for key in DEFAULT:
                write_run(tmp, "boq", key)
            self.assertEqual(plan("next", "--model", "boq", "--root", tmp), ["DONE"])
            self.assertEqual(plan("next", "--model", "boq", "--root", tmp, "--runs", "fare_eps2"),
                             [step2.env_assignments(runs["fare_eps2"])])

    def test_unknown_model_or_key_exits_2(self):
        for argv in (["next", "--model", "netvlad", "--root", "x"], ["name", "--model", "boq", "bogus"],
                     ["next", "--model", "boq", "--root", "x", "--runs", "bogus"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as stop:
                plan(*argv)
            self.assertEqual(stop.exception.code, 2)


class ModelsTests(unittest.TestCase):
    def test_trained_rows_give_their_table_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            araft = write_run(tmp, "boq", "araft")  # budget-5 epoch 3, best epoch 6
            same = [record(-1, 90.0, 0.0, (5,)), record(3, 88.0, 30.0, (5,))]
            plain = write_run(tmp, "boq", "plain_at", records=same)  # one checkpoint for both tables
            fare = write_run(tmp, "boq", "fare", records=[record(-1, 90.0, 0.0, (5,)), record(3, 70.0, 30.0, ())])
            twin = write_run(tmp, "boq", "twin")
            write_run(tmp, "boq", "fare_eps2", state="paused")
            lines = plan("models", "--model", "boq", "--root", tmp, "--pretrained", "ck/boq.pth")
        self.assertEqual(lines, [
            "main pretrained=ck/boq.pth",
            f"main araft_b5={araft}/best_model_budget5.pth",
            f"main araft_best={araft}/best_model.pth",
            f"main plain_at={plain}/best_model.pth",
            f"main fare_best={fare}/best_model.pth",  # no epoch within budget: Table A is the pretrained row
            f"twin twin={twin}/last_model.pth",
            "pending fare_eps2",
        ])

    def test_untrained_default_rows_are_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            lines = plan("models", "--model", "mixvpr", "--root", tmp, "--pretrained", "ck/m")
        self.assertEqual(lines, ["main pretrained=ck/m", *[f"pending {k}" for k in DEFAULT]])


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.root = tmp / "step4"
        self.eval_root = tmp / "eval"
        self.python = tmp / "python"
        self.python.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "train.py" ]; then shift; exec "' + sys.executable + '" -c '
            '"import sys; from tests.test_followup_plan import fake_train; fake_train(sys.argv[1:])" "$@"; fi\n'
            'if [ "$1" = "eval.py" ]; then shift; exec "' + sys.executable + '" -c '
            '"import sys; from tests.test_step4_plan import fake_eval; fake_eval(sys.argv[1:])" "$@"; fi\n'
            'exec "' + sys.executable + '" "$@"\n', encoding="utf-8")
        self.python.chmod(self.python.stat().st_mode | stat.S_IEXEC)
        self.env = {"STEP4_ROOT": str(self.root), "STEP4_EVAL_ROOT": str(self.eval_root), "STEP4_MODEL": "boq",
                    "PYTHON": str(self.python), "STEP4_STOP_AT": "+12 hours", "BOQ_BASE_CHECKPOINT": str(tmp / "boq.pth")}
        (tmp / "boq.pth").write_text("", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_driver(self, *args, **env):
        return run_script(STEP4_SCRIPT, {**self.env, **env}, list(args))

    def test_train_dry_run_prints_default_runs_in_order(self):
        result = self.run_driver("train", "--dry-run", MPLC_V2_MODEL="mixvpr", MPLC_V2_TAU="0.1")  # ignored
        self.assertEqual(result.returncode, 0, result.stdout)
        lines = commands(result.stdout)
        self.assertEqual([save_dir(line) for line in lines], [EXPECTED["boq"][k] for k in DEFAULT])
        for line in lines:
            self.assertIn("--model=boq", line.split())
            self.assertIn("--selection_clean_budgets 1 3 5 7 10 --val_rank_epsilons 0.0342 0.0685", line)
            self.assertIn(f"--run_dir={self.root}/", line)
            self.assertNotIn("--listwise_tau=0.1", line)
        self.assertFalse(self.root.exists())

    def test_options_are_checked(self):
        cases = [
            (("train",), {"STEP4_MODEL": ""}),
            (("train",), {"STEP4_MODEL": "netvlad"}),
            (("train",), {"STEP4_STOP_AT": ""}),
            (("train",), {"STEP4_STOP_AT": "+30 minutes"}),
            (("train",), {"STEP4_RUNS": "araft bogus"}),
            (("eval",), {"STEP4_STOP_AT": ""}),
            (("bogus",), {}),
            ((), {}),
            (("train", "--bogus"), {}),
        ]
        for args, env in cases:
            with self.subTest(args=args, env=env):
                self.assertEqual(self.run_driver(*args, **env).returncode, 2)
        result = run_script(STEP4_SCRIPT, args=["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("STEP4_STOP_AT", result.stdout)
        self.assertFalse(self.root.exists())

    def test_train_session_trains_then_is_done_and_resumes_after_pause(self):
        result = self.run_driver("train", FAKE_TRAIN="session")
        self.assertEqual(result.returncode, 5, result.stdout)
        result = self.run_driver("train")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"=== RESUME {EXPECTED['boq']['araft']} from", result.stdout)
        self.assertEqual(result.stdout.count("=== Step 4 run: "), 4, result.stdout)
        call = (self.root / EXPECTED["boq"]["araft"] / "train_calls.txt").read_text(encoding="utf-8")
        self.assertIn("--stop_by=", call)
        text = (self.root / "summary" / "runs.md").read_text(encoding="utf-8")
        self.assertIn("# Step 4 training runs (boq)", text)
        self.assertIn(f"| fare_eps2 | {EXPECTED['boq']['fare_eps2']} | pending |", text)
        result = self.run_driver("train")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("=== Step 4 run: ", result.stdout)

    def test_train_lock_held_exits_1(self):
        self.root.mkdir(parents=True)
        with open(self.root / ".lock", "w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_driver("train")
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_eval_waits_for_untrained_rows_then_resumes_piece_by_piece(self):
        result = self.run_driver("eval")  # nothing trained: the pretrained row only, then exit 3
        self.assertEqual(result.returncode, 3, result.stdout)
        base = self.eval_root / "boq"
        self.assertTrue((base / "r1_apgd" / "pretrained" / "msls.done").is_file())
        for key in DEFAULT:
            write_run(self.root, "boq", key)
        result = self.run_driver("eval", "--dry-run")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("pretrained", "".join(commands(result.stdout)))  # finished pieces are skipped
        result = self.run_driver("eval", FAKE_EVAL="fail")
        self.assertEqual(result.returncode, 1, result.stdout)
        result = self.run_driver("eval")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("=== Step 4 evaluation done", result.stdout)
        tags = ["pretrained", "araft_b5", "araft_best", "plain_at_b5", "plain_at_best", "fare_b5", "fare_best"]
        for piece, suffix in (("r1_apgd", "_rank_apgd_linf_steps100_q1000"), ("r2_pgd_grid", "_q1000")):
            for tag in tags:
                for dataset in DATASETS:
                    with self.subTest(piece=piece, tag=tag, dataset=dataset):
                        self.assertTrue((base / piece / tag / f"boq{'_' + dataset}{suffix}" /
                                         "rank_eval_results.json").is_file())
        call = (base / "r1_apgd" / "araft_b5" / "boq_msls_rank_apgd_linf_steps100_q1000" / "eval_calls.txt").read_text()
        for flag in ("--model_type=boq", "--rank_attack=rank_apgd_linf", "--rank_steps=100", "--max_queries=1000",
                     "--epsilons 0.03425 0.0685", "best_model_budget5.pth"):
            self.assertIn(flag, call)
        for dataset in DATASETS:
            self.assertTrue((base / "twin_clean" / "twin" / f"boq_{dataset}_q1000" / "rank_eval_results.json").is_file())
        result = self.run_driver("eval")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("=== Piece", result.stdout)


if __name__ == "__main__":
    unittest.main()
