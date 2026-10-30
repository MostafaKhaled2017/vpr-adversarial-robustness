import json
import os
import stat
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "scripts" / "mplc_v2_boq_eval.sh"


FAKE_PYTHON = """#!/usr/bin/env python3
import sys
from pathlib import Path

args = sys.argv[1:]
dataset = args[args.index("--datasets") + 1]
output_json = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--output_json="))
output_dir = Path(output_json).parent / "fake-attempt"
output_dir.mkdir(parents=True, exist_ok=True)
for name in ("rank_eval_results.json", "rank_eval_results.csv", "per_query_ranks.csv"):
    (output_dir / name).write_text("{}\\n", encoding="utf-8")
invocation_log = Path(output_json).parents[2] / "invocations.log"
with invocation_log.open("a", encoding="utf-8") as handle:
    handle.write(dataset + "\\n")
"""


class BoQResumableEvalTests(unittest.TestCase):
    def run_driver(self, root, fake_python, *args):
        env = os.environ.copy()
        env["PYTHON"] = str(fake_python)
        return subprocess.run(
            ["bash", str(DRIVER), *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def test_skips_valid_piece_and_runs_incomplete_piece(self):
        with TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            fake_python = temporary / "fake-python"
            fake_python.write_text(FAKE_PYTHON, encoding="utf-8")
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
            checkpoint = temporary / "checkpoint.pth"
            checkpoint.write_text("checkpoint", encoding="utf-8")
            root = temporary / "output"

            completed = root / "boq_msls_rank_apgd_linf_steps100_q1000" / "model" / "fake-attempt"
            completed.mkdir(parents=True)
            for name in ("rank_eval_results.json", "rank_eval_results.csv", "per_query_ranks.csv"):
                (completed / name).write_text("{}\n", encoding="utf-8")
            (completed.parent / "DONE").write_text(
                json.dumps({"result_dir": str(completed)}), encoding="utf-8"
            )

            result = self.run_driver(
                root,
                fake_python,
                "--model",
                f"model={checkpoint}",
                "--datasets",
                "msls pitts30k",
                "--output-root",
                str(root),
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("SKIP msls/model", result.stdout)
            self.assertIn("DONE pitts30k/model", result.stdout)
            self.assertEqual(
                (root / "invocations.log").read_text(encoding="utf-8").splitlines(),
                ["pitts30k"],
            )

    def test_dry_run_prints_one_command_per_piece(self):
        with TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            checkpoint = temporary / "checkpoint.pth"
            checkpoint.write_text("checkpoint", encoding="utf-8")
            result = self.run_driver(
                temporary / "output",
                temporary / "unused-python",
                "--model",
                f"model={checkpoint}",
                "--datasets",
                "msls pitts30k",
                "--dry-run",
            )

            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(result.stdout.count("--rank_attack=rank_apgd_linf"), 2)
            self.assertIn("--rank_steps=100", result.stdout)
            self.assertIn("--num_workers=2", result.stdout)
            self.assertIn("--infer_batch_size=4", result.stdout)


if __name__ == "__main__":
    unittest.main()
