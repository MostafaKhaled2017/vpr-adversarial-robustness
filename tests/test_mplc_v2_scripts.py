import json
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_train.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_eval.sh"


def run_script(script, extra_env=None):
    environment = os.environ.copy()
    environment.update({"MPLC_V2_DRY_RUN": "1", "PYTHON": sys.executable})
    if extra_env:
        environment.update(extra_env)
    return subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


class MplcV2TrainScriptTests(unittest.TestCase):
    def test_default_seed_trains_clean_ft_then_mplc(self):
        result = run_script(TRAIN_SCRIPT)

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--seed=0", result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_clean_ft_s0", result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_s0", result.stdout)
        self.assertIn("--validation_protocol=rank_pgd", result.stdout)
        self.assertIn("RankLinfAttack(model, epsilon=0.1, steps=5)", result.stdout)

        commands = [line for line in result.stdout.splitlines() if line.startswith("+ ")]
        clean_ft_command = next(
            line for line in commands if "mplc_v2_supervlad_clean_ft_s0" in line
        )
        self.assertNotIn("--attack", clean_ft_command)

    def test_seed_override_trains_seed_one(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_SEED": "1"})

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--seed=1", result.stdout)
        self.assertIn("_s1", result.stdout)

    def test_unknown_arm_exits_with_status_2(self):
        result = run_script(TRAIN_SCRIPT, {"MPLC_V2_ARMS": "bogus"})

        self.assertEqual(result.returncode, 2, result.stdout)

    def test_finished_arm_is_skipped(self):
        run_dir = REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s99" / "x"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_status.json").write_text(
            json.dumps({"state": "early_stopped"}), encoding="utf-8"
        )
        try:
            result = run_script(
                TRAIN_SCRIPT, {"MPLC_V2_SEED": "99", "MPLC_V2_ARMS": "clean_ft"}
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("SKIP mplc_v2_supervlad_clean_ft_s99", result.stdout)
        finally:
            shutil.rmtree(
                REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s99", ignore_errors=True
            )

    def test_mplc_tau_override_gets_distinct_save_dir(self):
        result = run_script(
            TRAIN_SCRIPT, {"MPLC_V2_ARMS": "clean_ft mplc", "MPLC_V2_TAU": "0.01"}
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_clean_ft_s0", result.stdout)
        self.assertIn("--save_dir=mplc_v2_supervlad_mplc_tau0.01_s0", result.stdout)

    def test_mplc_override_run_is_not_skipped_by_default_finished_run(self):
        run_dir = REPO_ROOT / "logs" / "mplc_v2_supervlad_mplc_s98" / "x"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_status.json").write_text(
            json.dumps({"state": "completed"}), encoding="utf-8"
        )
        try:
            result = run_script(
                TRAIN_SCRIPT,
                {"MPLC_V2_SEED": "98", "MPLC_V2_ARMS": "mplc", "MPLC_V2_TAU": "0.01"},
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn("TRAIN mplc_v2_supervlad_mplc_tau0.01_s98", result.stdout)
            self.assertNotIn("SKIP", result.stdout)
        finally:
            shutil.rmtree(
                REPO_ROOT / "logs" / "mplc_v2_supervlad_mplc_s98", ignore_errors=True
            )


class MplcV2EvalScriptTests(unittest.TestCase):
    def test_explicit_models_and_single_dataset(self):
        result = run_script(
            EVAL_SCRIPT,
            {
                "MPLC_V2_MODELS": "a=checkpoints/SuperVLAD_base.pth",
                "MPLC_V2_DATASETS": "sped",
            },
        )

        self.assertEqual(result.returncode, 0, result.stdout)
        eval_lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("+ ") and "eval.py" in line
        ]
        self.assertEqual(len(eval_lines), 1, result.stdout)
        self.assertIn("--model_tags a", eval_lines[0])
        self.assertIn("--rank_steps=20", eval_lines[0])
        self.assertIn("--datasets sped", eval_lines[0])

    def test_interrupted_run_without_status_is_not_evaluated(self):
        run_dir = REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s97" / "run_a"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_model.pth").write_text("x", encoding="utf-8")
        try:
            result = run_script(
                EVAL_SCRIPT, {"MPLC_V2_SEEDS": "97", "MPLC_V2_DATASETS": "sped"}
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(
                "WARNING: no finished checkpoint for mplc_v2_supervlad_clean_ft_s97",
                result.stdout,
            )
            self.assertNotIn(str(run_dir), result.stdout)
        finally:
            shutil.rmtree(
                REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s97", ignore_errors=True
            )

    def test_running_state_run_is_not_evaluated(self):
        run_dir = REPO_ROOT / "logs" / "mplc_v2_supervlad_mplc_s96" / "run_a"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "best_model.pth").write_text("x", encoding="utf-8")
        (run_dir / "run_status.json").write_text(
            json.dumps({"state": "running"}), encoding="utf-8"
        )
        try:
            result = run_script(
                EVAL_SCRIPT, {"MPLC_V2_SEEDS": "96", "MPLC_V2_DATASETS": "sped"}
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(
                "WARNING: no finished checkpoint for mplc_v2_supervlad_mplc_s96",
                result.stdout,
            )
            self.assertNotIn(str(run_dir), result.stdout)
        finally:
            shutil.rmtree(
                REPO_ROOT / "logs" / "mplc_v2_supervlad_mplc_s96", ignore_errors=True
            )

    def test_finished_run_is_evaluated_and_newest_wins(self):
        base = REPO_ROOT / "logs" / "mplc_v2_supervlad_clean_ft_s95"
        old_dir = base / "2024-01-01_run"
        new_dir = base / "2024-01-02_run"
        for run_dir, state in ((old_dir, "completed"), (new_dir, "early_stopped")):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "best_model.pth").write_text("x", encoding="utf-8")
            (run_dir / "run_status.json").write_text(
                json.dumps({"state": state}), encoding="utf-8"
            )
        try:
            result = run_script(
                EVAL_SCRIPT, {"MPLC_V2_SEEDS": "95", "MPLC_V2_DATASETS": "sped"}
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            eval_lines = [
                line
                for line in result.stdout.splitlines()
                if line.startswith("+ ") and "eval.py" in line
            ]
            self.assertEqual(len(eval_lines), 1, result.stdout)
            self.assertIn(
                f"{new_dir.relative_to(REPO_ROOT)}/best_model.pth", eval_lines[0]
            )
            self.assertNotIn(
                f"{old_dir.relative_to(REPO_ROOT)}/best_model.pth", eval_lines[0]
            )
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
