import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "phase2_supervlad.sh"


class Phase2LauncherResumeTests(unittest.TestCase):
    def run_launcher(self, *arguments, **extra_env):
        environment = os.environ.copy()
        environment.update(
            {
                "PHASE2_DRY_RUN": "1",
                "PYTHON": sys.executable,
                "PHASE2_TAUS": "0.05",
                "PHASE2_KS": "1",
                "PHASE2_POOLS": "0",
                **extra_env,
            }
        )
        return subprocess.run(
            ["bash", str(LAUNCHER), *arguments],
            cwd=REPO_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    def root_from_output(self, output):
        prefix = "Pilot run root: "
        line = next(line for line in output.splitlines() if line.startswith(prefix))
        return Path(line.removeprefix(prefix))

    def test_new_pilot_creates_group_and_uses_exact_config_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_launcher("pilot", PHASE2_PILOT_ROOT_BASE=directory)

            self.assertEqual(result.returncode, 0, result.stdout)
            root = self.root_from_output(result.stdout)
            config_dir = root / "tau0.05_k1_pool0"
            self.assertTrue((root / "pilot_manifest.yaml").is_file())
            self.assertIn(f"--run_dir={config_dir}", result.stdout)
            self.assertIn("--resume_model_only", result.stdout)

    def test_explicit_root_resumes_from_last_checkpoint_without_model_only_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher("pilot", PHASE2_PILOT_ROOT_BASE=directory)
            root = self.root_from_output(first.stdout)
            config_dir = root / "tau0.05_k1_pool0"
            checkpoint = config_dir / "last_model.pth"
            checkpoint.touch()

            resumed = self.run_launcher("pilot", "--run-root", str(root))

            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            self.assertIn(f"--resume={checkpoint}", resumed.stdout)
            self.assertIn("--continue", resumed.stdout)
            command = next(line for line in resumed.stdout.splitlines() if line.startswith("+ "))
            self.assertNotIn("--resume_model_only", command)

    def test_terminal_configuration_is_skipped_even_when_best_model_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher("pilot", PHASE2_PILOT_ROOT_BASE=directory)
            root = self.root_from_output(first.stdout)
            config_dir = root / "tau0.05_k1_pool0"
            (config_dir / "best_model.pth").touch()
            (config_dir / "run_status.json").write_text(
                json.dumps({"state": "completed"}), encoding="utf-8"
            )

            resumed = self.run_launcher("pilot", "--run-root", str(root))

            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            self.assertIn("SKIP tau0.05_k1_pool0: completed", resumed.stdout)
            self.assertNotIn("train.py", resumed.stdout)

    def test_resume_uses_manifest_grid_when_no_overrides_are_given(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher("pilot", PHASE2_PILOT_ROOT_BASE=directory)
            root = self.root_from_output(first.stdout)
            environment = os.environ.copy()
            environment["PHASE2_DRY_RUN"] = "1"
            environment["PYTHON"] = sys.executable

            resumed = subprocess.run(
                ["bash", str(LAUNCHER), "pilot", "--run-root", str(root)],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )

            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            self.assertIn("1 taus x 1 ks x 1 pools", resumed.stdout)

    def test_resume_rejects_a_conflicting_fixed_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher("pilot", PHASE2_PILOT_ROOT_BASE=directory)
            root = self.root_from_output(first.stdout)

            resumed = self.run_launcher(
                "pilot",
                "--run-root",
                str(root),
                PHASE2_RAMP_EPOCHS="9",
            )

            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("PHASE2_RAMP_EPOCHS does not match", resumed.stdout)

    def test_pending_config_uses_the_base_checkpoint_frozen_in_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher(
                "pilot",
                PHASE2_PILOT_ROOT_BASE=directory,
                PHASE1_BASE_PATH="checkpoints/custom_base.pth",
            )
            root = self.root_from_output(first.stdout)

            resumed = self.run_launcher("pilot", "--run-root", str(root))

            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            command = next(line for line in resumed.stdout.splitlines() if line.startswith("+ "))
            self.assertIn("--resume=checkpoints/custom_base.pth", command)
            self.assertIn("--resume_model_only", command)

    def test_pending_config_keeps_from_scratch_initialization_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher(
                "pilot",
                PHASE2_PILOT_ROOT_BASE=directory,
                SUPERVLAD_FROM_SCRATCH="1",
            )
            root = self.root_from_output(first.stdout)

            resumed = self.run_launcher("pilot", "--run-root", str(root))

            self.assertEqual(resumed.returncode, 0, resumed.stdout)
            command = next(line for line in resumed.stdout.splitlines() if line.startswith("+ "))
            self.assertNotIn("--resume=", command)
            self.assertNotIn("--resume_model_only", command)

    def test_select_reads_only_terminal_runs_from_explicit_root(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher("pilot", PHASE2_PILOT_ROOT_BASE=directory)
            root = self.root_from_output(first.stdout)
            config_dir = root / "tau0.05_k1_pool0"
            (config_dir / "run_status.json").write_text(
                json.dumps({"state": "completed"}), encoding="utf-8"
            )
            common_eval = Path(directory) / "common_eval.json"
            common_eval.write_text("{}", encoding="utf-8")

            selected = self.run_launcher(
                "select", "--run-root", str(root), PHASE2_COMMON_EVAL_JSON=str(common_eval)
            )

            self.assertEqual(selected.returncode, 0, selected.stdout)
            self.assertIn(f"--run tau0.05_k1_pool0={config_dir}", selected.stdout)
            self.assertIn(f"--common_eval_json {common_eval}", selected.stdout)
            self.assertIn(f"--output_markdown {root / 'pilot_ranking.md'}", selected.stdout)

    def test_select_requires_common_validation_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher("pilot", PHASE2_PILOT_ROOT_BASE=directory)
            root = self.root_from_output(first.stdout)
            config_dir = root / "tau0.05_k1_pool0"
            (config_dir / "run_status.json").write_text(
                json.dumps({"state": "completed"}), encoding="utf-8"
            )

            selected = self.run_launcher("select", "--run-root", str(root))

        self.assertNotEqual(selected.returncode, 0)
        self.assertIn("PHASE2_COMMON_EVAL_JSON", selected.stdout)

    def test_select_refuses_a_root_with_pending_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher("pilot", PHASE2_PILOT_ROOT_BASE=directory)
            root = self.root_from_output(first.stdout)
            common_eval = Path(directory) / "common_eval.json"
            common_eval.write_text("{}", encoding="utf-8")

            selected = self.run_launcher(
                "select", "--run-root", str(root), PHASE2_COMMON_EVAL_JSON=str(common_eval)
            )

            self.assertNotEqual(selected.returncode, 0)
            self.assertIn("is not terminal", selected.stdout)

    def test_select_does_not_rank_a_collapse_aborted_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher("pilot", PHASE2_PILOT_ROOT_BASE=directory)
            root = self.root_from_output(first.stdout)
            config_dir = root / "tau0.05_k1_pool0"
            (config_dir / "run_status.json").write_text(
                json.dumps({"state": "collapse_aborted"}), encoding="utf-8"
            )
            common_eval = Path(directory) / "common_eval.json"
            common_eval.write_text("{}", encoding="utf-8")

            selected = self.run_launcher(
                "select", "--run-root", str(root), PHASE2_COMMON_EVAL_JSON=str(common_eval)
            )

            self.assertNotEqual(selected.returncode, 0)
            self.assertIn("EXCLUDE tau0.05_k1_pool0: collapse_aborted", selected.stdout)
            self.assertNotIn(f"--run tau0.05_k1_pool0={config_dir}", selected.stdout)

    def test_final_can_read_the_winner_from_an_explicit_pilot_root(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.run_launcher(
                "pilot",
                PHASE2_PILOT_ROOT_BASE=directory,
                PHASE2_RAMP_EPOCHS="9",
            )
            root = self.root_from_output(first.stdout)
            (root / "pilot_winner.txt").write_text("tau0.05_k1_pool0\n", encoding="utf-8")

            final = self.run_launcher("final", "--run-root", str(root), PHASE2_SEEDS="0")

            self.assertEqual(final.returncode, 0, final.stdout)
            self.assertIn("Winning configuration: tau0.05_k1_pool0", final.stdout)
            self.assertIn("--attack_ramp_epochs=9", final.stdout)


if __name__ == "__main__":
    unittest.main()
