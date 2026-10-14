import contextlib
import csv
import fcntl
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

from src import followup_plan as followup
from src import step2_plan as step2
from tests.test_step1_sweep import REPO_ROOT, TRAIN_SCRIPT, run_script

FOLLOWUP_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_followup_train.sh"
EXPECTED = {
    "sp_s1": "mplc_v2_supervlad_mplc_mixlinf_sp_fte0_s1",
    "sp_hinge": "mplc_v2_supervlad_mplc_mixlinf_hinge_sp_fte0_s0",
    "sp_aw2": "mplc_v2_supervlad_mplc_aw2_mixlinf_sp_fte0_s0",
    "fare_eps2": "mplc_v2_supervlad_fare_eps0.0342_fte0_s0",
    "plain_at_eps2": "mplc_v2_supervlad_plain_at_eps0.0342_fte0_s0",
    "sp_eps2": "mplc_v2_supervlad_mplc_eps0.0342_mixlinf_sp_fte0_s0",
    "sp_aw3": "mplc_v2_supervlad_mplc_aw3_mixlinf_sp_fte0_s0",
}
KEYS = list(EXPECTED)


def record(epoch, clean, robust, budgets, eps2=None, eps4=None):
    """A validation_recalls.jsonl record; R@1 at eps 0.0342 / 0.0685 default to robust +- 10."""
    eps2 = robust + 10.0 if eps2 is None else eps2
    eps4 = robust - 10.0 if eps4 is None else eps4
    return {
        "epoch": epoch, "clean": {"R@1": clean}, "robust_score": robust, "eligible_budgets": list(budgets),
        "attacks": {"rank_pgd_linf_eps_0.0342": {"R@1": eps2}, "rank_pgd_linf_eps_0.0685": {"R@1": eps4}},
    }


def write_run(root, key, state="early_stopped", records=()):
    run_dir = Path(root) / EXPECTED[key]
    run_dir.mkdir(parents=True, exist_ok=True)
    if state is not None:
        (run_dir / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    (run_dir / "validation_recalls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return run_dir


def plan(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        followup.main(list(argv))
    return out.getvalue().splitlines()


def commands(output):
    return [line for line in output.splitlines() if line.startswith("+ ")]


def save_dir(command):
    return next(word for word in command.split() if word.startswith("--save_dir=")).split("=", 1)[1]


def fake_train(argv):
    """train.py for the driver tests: FAKE_TRAIN=finish (early_stopped, exit 0), pause (paused,
    exit 0, a run that did not finish) or session (paused, exit 5)."""
    run_dir = Path(next(a for a in argv if a.startswith("--run_dir=")).split("=", 1)[1])
    run_dir.mkdir(parents=True, exist_ok=True)
    mode = os.environ.get("FAKE_TRAIN", "finish")
    with (run_dir / "train_calls.txt").open("a", encoding="utf-8") as handle:
        handle.write(" ".join(argv) + "\n")
    (run_dir / "last_model.pth").write_text("", encoding="utf-8")
    records = [record(-1, 90.0, 0.0, (1, 3, 5, 7, 10)), record(3, 87.0, 30.0, (5, 7, 10))]
    (run_dir / "validation_recalls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    state = "early_stopped" if mode == "finish" else "paused"
    (run_dir / "run_status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
    sys.exit(5 if mode == "session" else 0)


class RunNameTests(unittest.TestCase):
    def test_names_match_the_launcher(self):
        self.assertEqual(list(followup.RUNS), KEYS)
        with tempfile.TemporaryDirectory() as tmp:
            for key, env in followup.RUNS.items():
                with self.subTest(key=key):
                    self.assertEqual(followup.run_name(env), EXPECTED[key])
                    result = run_script(TRAIN_SCRIPT, {"MPLC_V2_DRY_RUN": "1", "MPLC_V2_RUN_ROOT": tmp, **env})
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertEqual(len(commands(result.stdout)), 1, result.stdout)  # one arm, no clean twin
                    words = result.stdout.split()
                    for flag in (f"--save_dir={EXPECTED[key]}", f"--run_dir={tmp}/{EXPECTED[key]}", "--freeze_te=0",
                                 "--val_every=3", "--patience=6", "--lr_plateau_patience=3", "--grad_checkpointing"):
                        self.assertIn(flag, words)
                    self.assertNotIn("--num_epochs=12", words)
                    train_eps = env.get("MPLC_V2_TRAIN_EPS", "0.0685")
                    self.assertIn(f"epsilon={train_eps},", result.stdout)
                    if env["MPLC_V2_ARMS"] == "mplc":  # single positive, rank L-inf only
                        self.assertNotIn("--multi_positive", words)
                        self.assertNotIn("PerceptualPGDAttack", result.stdout)
        self.assertEqual(plan("name", "sp_eps2"), [EXPECTED["sp_eps2"]])

    def test_eps_suffix_follows_the_launcher_for_other_settings(self):
        cases = [
            {"MPLC_V2_ARMS": "fare", "MPLC_V2_ALIGN_WEIGHT": "10", "MPLC_V2_TRAIN_EPS": "0.137", "MPLC_V2_LR": "3e-6"},
            {"MPLC_V2_ARMS": "clean_ft", "MPLC_V2_TRAIN_EPS": "0.137", "MPLC_V2_SEED": "2"},
            {"MPLC_V2_ARMS": "mplc", "MPLC_V2_ALIGN_WEIGHT": "0", "MPLC_V2_TRAIN_EPS": "0.137",
             "MPLC_V2_DEFENSE_LOSS": "hinge"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for env in cases:
                with self.subTest(env=env):
                    result = run_script(TRAIN_SCRIPT, {"MPLC_V2_DRY_RUN": "1", "MPLC_V2_RUN_ROOT": tmp, **env})
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn(f"--save_dir={followup.run_name(env)}", result.stdout.split())


class NextTests(unittest.TestCase):
    def test_first_unfinished_run_in_order_then_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(plan("next", "--root", tmp), [step2.env_assignments(followup.RUNS["sp_s1"])])
            write_run(root, "sp_s1")
            write_run(root, "sp_hinge", state="paused")  # started, not finished: the launcher resumes it
            self.assertEqual(plan("next", "--root", tmp), [step2.env_assignments(followup.RUNS["sp_hinge"])])
            write_run(root, "sp_hinge", state="completed")
            write_run(root, "sp_aw2", state="collapse_aborted")
            write_run(root, "plain_at_eps2", state=None)  # directory without run_status.json
            self.assertEqual(plan("next", "--root", tmp), [step2.env_assignments(followup.RUNS["fare_eps2"])])
            pending = plan("next", "--root", tmp, "--all")
            self.assertEqual(pending, [step2.env_assignments(followup.RUNS[k])
                                       for k in ("fare_eps2", "plain_at_eps2", "sp_eps2", "sp_aw3")])
            for key in KEYS:
                write_run(root, key)
            self.assertEqual(plan("next", "--root", tmp), ["DONE"])
            self.assertEqual(plan("next", "--root", tmp, "--all"), ["DONE"])

    def test_runs_subset_keeps_its_order_and_unknown_keys_exit_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, "sp_aw3")
            lines = plan("next", "--root", tmp, "--all", "--runs", "sp_aw3", "fare_eps2", "sp_s1")
            self.assertEqual(lines, [step2.env_assignments(followup.RUNS[k]) for k in ("fare_eps2", "sp_s1")])
            self.assertEqual(plan("next", "--root", tmp, "--runs", "sp_aw3"), ["DONE"])
            listed = plan("list", "--runs", "plain_at_eps2")
            self.assertEqual(listed[0], f"plain_at_eps2  {EXPECTED['plain_at_eps2']}")
            for argv in (["next", "--root", tmp, "--runs", "sp_s1", "bogus"], ["name", "bogus"],
                         ["summary", "--root", tmp, "--runs", "bogus"]):
                err = io.StringIO()
                with self.subTest(argv=argv), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop:
                    plan(*argv)
                self.assertEqual(stop.exception.code, 2)
                self.assertIn("sp_aw3", err.getvalue())  # the valid keys are listed


class SummaryTests(unittest.TestCase):
    def test_budget5_and_best_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, "sp_s1", records=[
                record(-1, 90.0, 0.0, (1, 3, 5, 7, 10), 0.0, 0.0),
                record(3, 87.0, 30.0, (5, 7, 10)),
                record(6, 84.0, 45.0, (7, 10), 55.0, 35.0),
                record(9, 88.0, 30.0, (5, 7, 10), 38.0, 22.0),  # ties epoch 3, higher clean: budget-5
                record(12, 80.0, 45.0, (10,)),  # ties epoch 6, lower clean: not best
            ])
            write_run(root, "sp_hinge", state="completed", records=[
                record(-1, 90.0, 0.0, (1, 3, 5, 7, 10)),
                record(3, 80.0, 50.0, (10,), 60.0, 40.0),
            ])
            write_run(root, "sp_aw2", state="paused", records=[record(-1, 90.0, 0.0, (5,)), record(3, 88.0, 20.0, (5,))])
            self.assertEqual(plan("summary", "--root", tmp), [f"Tables written to {root / 'summary'}"])
            with (root / "summary" / "runs.csv").open(encoding="utf-8") as handle:
                rows = {row["key"]: row for row in csv.DictReader(handle)}
            self.assertEqual(list(rows), KEYS)

            sp = rows["sp_s1"]
            self.assertEqual((sp["run"], sp["state"], sp["epochs_run"], sp["initial_clean_r1"]),
                             (EXPECTED["sp_s1"], "early_stopped", "12", "90.00"))
            self.assertEqual([sp[f"b5_{f}"] for f in followup.CHECKPOINT_FIELDS],
                             ["9", "88.00", "2.00", "38.00", "22.00", "30.00"])
            self.assertEqual([sp[f"best_{f}"] for f in followup.CHECKPOINT_FIELDS],
                             ["6", "84.00", "6.00", "55.00", "35.00", "45.00"])

            hinge = rows["sp_hinge"]  # no trained epoch within 5 points
            self.assertEqual(hinge["state"], "completed")
            self.assertEqual([hinge[f"b5_{f}"] for f in followup.CHECKPOINT_FIELDS], ["initial", "", "", "", "", ""])
            self.assertEqual([hinge[f"best_{f}"] for f in followup.CHECKPOINT_FIELDS],
                             ["3", "80.00", "10.00", "60.00", "40.00", "50.00"])

            paused = rows["sp_aw2"]  # unfinished: state and progress only
            self.assertEqual((paused["state"], paused["epochs_run"], paused["b5_epoch"], paused["best_epoch"]),
                             ("paused", "3", "", ""))
            self.assertEqual((rows["sp_aw3"]["state"], rows["sp_aw3"]["epochs_run"]), ("pending", ""))

            text = (root / "summary" / "runs.md").read_text(encoding="utf-8")
            self.assertIn(f"| sp_s1 | {EXPECTED['sp_s1']} | early_stopped | 12 |", text)
            self.assertIn("| initial |", text)
            self.assertNotIn("run_dir", text)

            plan("summary", "--root", tmp, "--runs", "sp_hinge")
            with (root / "summary" / "runs.csv").open(encoding="utf-8") as handle:
                self.assertEqual([row["key"] for row in csv.DictReader(handle)], ["sp_hinge"])


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "followup"
        self.env = {"FOLLOWUP_ROOT": str(self.root)}

    def tearDown(self):
        self.tmp.cleanup()

    def dry_run(self, **env):
        return run_script(FOLLOWUP_SCRIPT, {**self.env, **env}, ["--dry-run"])

    def test_dry_run_prints_every_pending_run_in_order(self):
        result = self.dry_run(MPLC_V2_TAU="0.1", MPLC_V2_ARMS="clean_ft mplc")  # stray settings are ignored
        self.assertEqual(result.returncode, 0, result.stdout)
        lines = commands(result.stdout)
        self.assertEqual([save_dir(line) for line in lines], list(EXPECTED.values()), result.stdout)
        for line in lines:
            self.assertIn("--selection_clean_budgets 1 3 5 7 10 --val_rank_epsilons 0.0342 0.0685", line)
            self.assertIn(f"--run_dir={self.root}/", line)
            self.assertNotIn("--pipe_loader", line.split())
            self.assertNotIn("--stop_by", line)
            self.assertNotIn("--listwise_tau=0.1", line)
        self.assertFalse(self.root.exists())

    def test_dry_run_overrides(self):
        result = self.dry_run(FOLLOWUP_RUNS="sp_aw3 sp_s1", FOLLOWUP_CLEAN_BUDGETS="1 5", FOLLOWUP_PIPE_LOADER="1")
        self.assertEqual(result.returncode, 0, result.stdout)
        lines = commands(result.stdout)
        self.assertEqual([save_dir(line) for line in lines], [EXPECTED["sp_aw3"], EXPECTED["sp_s1"]])
        for line in lines:
            self.assertIn("--selection_clean_budgets 1 5 --val_rank_epsilons", line)
            self.assertIn("--pipe_loader", line.split())

    def test_dry_run_skips_finished_runs(self):
        write_run(self.root, "sp_s1")
        write_run(self.root, "fare_eps2", state="collapse_aborted")
        result = self.dry_run()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual([save_dir(line) for line in commands(result.stdout)],
                         [EXPECTED[k] for k in KEYS if k not in ("sp_s1", "fare_eps2")])
        self.assertNotIn("SKIP", result.stdout)
        for key in KEYS:
            write_run(self.root, key)
        result = self.dry_run()
        self.assertEqual((result.returncode, commands(result.stdout)), (0, []), result.stdout)
        self.assertIn("Every follow-up run is finished", result.stdout)

    def test_options_stop_at_and_keys_are_checked(self):
        self.assertEqual(run_script(FOLLOWUP_SCRIPT, self.env).returncode, 2)  # no FOLLOWUP_TRAIN_STOP_AT
        for stop_at in ("+30 minutes", "soon"):
            self.assertEqual(run_script(FOLLOWUP_SCRIPT, {**self.env, "FOLLOWUP_TRAIN_STOP_AT": stop_at}).returncode, 2)
        result = run_script(FOLLOWUP_SCRIPT, {**self.env, "FOLLOWUP_TRAIN_STOP_AT": "+3 hours",
                                              "FOLLOWUP_RUNS": "sp_s1 bogus"})
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("sp_aw3", result.stdout)
        self.assertEqual(self.dry_run(FOLLOWUP_RUNS="bogus").returncode, 2)
        self.assertEqual(run_script(FOLLOWUP_SCRIPT, args=["--bogus"]).returncode, 2)
        self.assertEqual(run_script(FOLLOWUP_SCRIPT, args=["--dry-run", "--summary-only"]).returncode, 2)
        result = run_script(FOLLOWUP_SCRIPT, args=["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("FOLLOWUP_TRAIN_STOP_AT", result.stdout)
        self.assertFalse(self.root.exists())

    def test_summary_only_writes_tables(self):
        write_run(self.root, "sp_hinge", records=[record(-1, 90.0, 0.0, (5,)), record(3, 88.0, 20.0, (5,))])
        result = run_script(FOLLOWUP_SCRIPT, {**self.env, "FOLLOWUP_RUNS": "sp_s1"}, ["--summary-only"])
        self.assertEqual(result.returncode, 0, result.stdout)
        text = (self.root / "summary" / "runs.md").read_text(encoding="utf-8")
        self.assertIn(f"| sp_hinge | {EXPECTED['sp_hinge']} | early_stopped | 3 |", text)
        self.assertIn("| sp_aw3 |", text)  # all seven rows, whatever FOLLOWUP_RUNS selects


class DriverSessionTests(unittest.TestCase):
    """Real sessions with train.py replaced by fake_train."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.root = tmp / "followup"
        self.python = tmp / "python"
        self.python.write_text(
            "#!/usr/bin/env bash\n"
            'if [ "$1" = "train.py" ]; then shift; exec "' + sys.executable + '" -c '
            '"import sys; from tests.test_followup_plan import fake_train; fake_train(sys.argv[1:])" "$@"; fi\n'
            'exec "' + sys.executable + '" "$@"\n', encoding="utf-8")
        self.python.chmod(self.python.stat().st_mode | stat.S_IEXEC)
        self.env = {"FOLLOWUP_ROOT": str(self.root), "PYTHON": str(self.python), "FOLLOWUP_TRAIN_STOP_AT": "+12 hours"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_session_trains_every_run_then_is_done(self):
        result = run_script(FOLLOWUP_SCRIPT, {**self.env, "FOLLOWUP_RUNS": "sp_eps2 sp_s1"})
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"=== Follow-up training done. Tables: {self.root}/summary/runs.md", result.stdout)
        self.assertEqual(sorted(p.name for p in self.root.iterdir() if p.name.startswith("mplc_v2")),
                         sorted([EXPECTED["sp_eps2"], EXPECTED["sp_s1"]]))
        call = (self.root / EXPECTED["sp_eps2"] / "train_calls.txt").read_text(encoding="utf-8")
        self.assertIn("--stop_by=", call)
        self.assertIn("--selection_clean_budgets 1 3 5 7 10", call)
        self.assertLess(result.stdout.index(EXPECTED["sp_eps2"]), result.stdout.index(EXPECTED["sp_s1"]))
        text = (self.root / "summary" / "runs.md").read_text(encoding="utf-8")
        self.assertIn(f"| sp_eps2 | {EXPECTED['sp_eps2']} | early_stopped | 3 |", text)
        self.assertIn(f"| sp_aw3 | {EXPECTED['sp_aw3']} | pending |", text)

        result = run_script(FOLLOWUP_SCRIPT, self.env)  # the other five, then done
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn(f"=== Follow-up run: {EXPECTED['sp_s1']}", result.stdout)
        self.assertEqual(result.stdout.count("=== Follow-up run: "), 5, result.stdout)
        result = run_script(FOLLOWUP_SCRIPT, self.env)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("=== Follow-up run: ", result.stdout)

    def test_paused_run_ends_the_session_with_5(self):
        result = run_script(FOLLOWUP_SCRIPT, {**self.env, "FAKE_TRAIN": "session"})
        self.assertEqual(result.returncode, 5, result.stdout)
        self.assertIn("Session over: the run paused", result.stdout)
        self.assertIn(f"| sp_s1 | {EXPECTED['sp_s1']} | paused |", (self.root / "summary" / "runs.md").read_text())
        result = run_script(FOLLOWUP_SCRIPT, self.env)  # next session: resumes, then trains the rest
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(f"=== RESUME {EXPECTED['sp_s1']} from", result.stdout)

    def test_run_that_returns_0_unfinished_stops_the_loop(self):
        result = run_script(FOLLOWUP_SCRIPT, {**self.env, "FAKE_TRAIN": "pause"})
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn(f"Run {EXPECTED['sp_s1']} ran but did not finish; see {self.root}/{EXPECTED['sp_s1']}/info.log",
                      result.stdout)
        self.assertEqual(result.stdout.count("=== Follow-up run: "), 1, result.stdout)

    def test_lock_held_exits_1(self):
        self.root.mkdir(parents=True)
        with open(self.root / ".lock", "w", encoding="utf-8") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = run_script(FOLLOWUP_SCRIPT, self.env)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("Another follow-up training driver is running", result.stdout)
        self.assertFalse(any(p.name.startswith("mplc_v2") for p in self.root.iterdir()))


if __name__ == "__main__":
    unittest.main()
