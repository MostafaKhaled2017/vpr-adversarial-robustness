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


if __name__ == "__main__":
    unittest.main()
