"""Run-skipping, resuming and argument checks of the scripts in scripts/."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.cli import parse_arguments  # noqa: E402


def run_script(root, script, *args):
    environment = {**os.environ, "DRY_RUN": "1", "PYTHON": sys.executable,
                   "RUN_ROOT": str(root / "runs"), "OUT_ROOT": str(root / "out")}
    return subprocess.run(["bash", str(REPO_ROOT / "scripts" / script), *args], cwd=REPO_ROOT, env=environment,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def commands(result):
    return [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

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
