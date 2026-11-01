"""The scripts in scripts/ run exactly the commands behind the paper's results.

tests/golden/ holds the parsed arguments of every training run and evaluation piece the paper
reports (see tests/golden/README.md). Each test dry-runs a script, parses the printed command
with the real parser and compares it with the golden arguments.
"""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src import rank_eval
from src.cli import parse_arguments

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = REPO_ROOT / "tests" / "golden"
# Where results go, and what never changes a result.
TRAIN_IGNORED = {"save_dir", "run_dir", "tensorboard_dir", "continue_training", "num_workers", "pipe_loader",
                 "stop_by", "selection_clean_budgets", "log_dir"}
EVAL_IGNORED = {"save_dir", "output_json", "output_csv", "num_workers", "trace_output_dir_path",
                "diagnostics_output_dir_path", "attack_image_output_dir_path", "model_tags", "recorded_in", "epsilons"}


def run_script(root, script, *args):
    environment = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable,
                   "RUN_ROOT": str(root / "runs"), "OUT_ROOT": str(root / "out")}
    return subprocess.run(["bash", str(REPO_ROOT / "scripts" / script), *args], cwd=REPO_ROOT, env=environment,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def commands(result):
    return [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]


def as_json(namespace):
    return json.loads(json.dumps(vars(namespace), default=str))


def load_golden(kind):
    return {path.stem: json.loads(path.read_text()) for path in (GOLDEN / kind).glob("*.json")}


class PaperScriptTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def assert_matches(self, name, actual, golden, ignored):
        self.assertLessEqual(set(actual) - ignored, set(golden), f"{name}: arguments missing from golden")
        diff = {key: (actual[key], golden[key]) for key in set(actual) & set(golden) - ignored
                if actual[key] != golden[key]}
        self.assertEqual(diff, {}, name)

    def test_train_commands_match_golden(self):
        runs = {}
        for script in ("train_screens.sh", "train_supervlad.sh", "train_boq.sh"):
            result = run_script(self.root, script)
            self.assertEqual(result.returncode, 0, result.stdout)
            for argv in commands(result):
                self.assertEqual(argv[1], "train.py")
                args = as_json(parse_arguments(argv[2:]))
                runs[Path(args["run_dir"]).name] = args
        golden = load_golden("train")
        self.assertEqual(sorted(runs), sorted(golden))
        for name, args in runs.items():
            self.assert_matches(name, args, golden[name], TRAIN_IGNORED)

    @unittest.skipUnless((REPO_ROOT / "checkpoints" / "SuperVLAD_base.pth").is_file(),
                         "eval.py's parser needs the released checkpoints")
    def test_eval_commands_match_golden(self):
        # eval.py's parser requires every checkpoint to exist.
        for name in load_golden("train"):
            (self.root / "runs" / name).mkdir(parents=True)
            for checkpoint in ("best_model.pth", "last_model.pth"):
                (self.root / "runs" / name / checkpoint).touch()
        pieces = {}
        for script in ("eval_supervlad.sh", "eval_boq.sh"):
            result = run_script(self.root, script)
            self.assertEqual(result.returncode, 0, result.stdout)
            for argv in commands(result):
                self.assertEqual(argv[1], "eval.py")
                args = as_json(rank_eval.finalize_arguments(rank_eval.build_parser().parse_args(argv[2:])))
                model, attack, dataset, tag = Path(args["output_json"]).parts[-5:-1]
                args["model_paths"] = [path.replace(str(self.root / "runs"), "logs/paper") for path in args["model_paths"]]
                pieces[f"{model}__{attack}__{dataset}__{tag}"] = args
        golden = load_golden("eval")
        self.assertEqual(sorted(pieces), sorted(golden))
        for name, args in pieces.items():
            self.assert_matches(name, args, golden[name], EVAL_IGNORED)
            # Every budget evaluated now was evaluated for the paper.
            self.assertLessEqual(set(args["epsilons"]), set(golden[name]["epsilons"]), name)

    def test_finished_run_is_skipped(self):
        run_dir = self.root / "runs" / "supervlad_araft_s0"
        run_dir.mkdir(parents=True)
        (run_dir / "run_status.json").write_text(json.dumps({"state": "early_stopped"}))
        result = run_script(self.root, "train.sh", "supervlad_araft_s0", "supervlad", "araft", "0")
        self.assertIn("=== SKIP supervlad_araft_s0", result.stdout)
        self.assertEqual(commands(result), [])

    def test_stopped_run_resumes_from_last_checkpoint(self):
        run_dir = self.root / "runs" / "supervlad_araft_s0"
        run_dir.mkdir(parents=True)
        (run_dir / "run_status.json").write_text(json.dumps({"state": "paused"}))
        (run_dir / "last_model.pth").touch()
        result = run_script(self.root, "train.sh", "supervlad_araft_s0", "supervlad", "araft", "0")
        [argv] = commands(result)
        args = parse_arguments(argv[2:])
        self.assertEqual(args.resume, str(run_dir / "last_model.pth"))
        self.assertTrue(args.continue_training)

    def test_done_piece_is_skipped(self):
        piece = self.root / "out" / "supervlad" / "apgd" / "sped" / "araft"
        result_dir = piece / "2026-01-01_00-00-00"
        result_dir.mkdir(parents=True)
        for name in ("rank_eval_results.json", "rank_eval_results.csv", "per_query_ranks.csv"):
            (result_dir / name).touch()
        (piece / "DONE").write_text(json.dumps({"result_dir": str(result_dir)}))
        result = run_script(self.root, "eval.sh", "supervlad", "apgd", "sped", "araft", "x.pth", "0.0685")
        self.assertIn("=== SKIP", result.stdout)
        self.assertEqual(commands(result), [])

    def test_unknown_model_or_method_fails(self):
        for model, method in (("supervlad", "fgsm"), ("mixvpr", "araft")):
            result = run_script(self.root, "train.sh", "x", model, method, "0")
            self.assertNotEqual(result.returncode, 0, (model, method))


if __name__ == "__main__":
    unittest.main()
