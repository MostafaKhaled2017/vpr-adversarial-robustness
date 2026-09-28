import dataclasses
import re
import tempfile
import unittest
from pathlib import Path

from src import step1_sweep as sweep
from src import step2_plan as step2
from tests.test_step1_sweep import EPOCHS, REPO_ROOT, TRAIN_SCRIPT, finished, run_script, write_run

STEP2_SCRIPT = REPO_ROOT / "scripts" / "mplc_v2_step2.sh"
BATCH = int(re.search(r"^SUPERVLAD_TRAIN_BATCH_SIZE=(\d+)",
                      (REPO_ROOT / "scripts" / "lib" / "supervlad_common.sh").read_text(), re.M).group(1))


def finish_step1(root, batch_size=BATCH):
    """Write finished Step 1 screens (all scoring 50) until the sweep reaches MPLC*."""
    sweep.ensure_config(Path(root), EPOCHS, 5.0, batch_size, 200, 3)
    while True:
        state = sweep.plan_sweep(sweep._results_reader(Path(root), EPOCHS, 5.0))
        if state.final is not None:
            return state
        for cfg in state.pending:
            records, run_state = finished(50.0)
            write_run(Path(root) / sweep.run_name(cfg, EPOCHS), state=run_state, records=records)


class RunNameTests(unittest.TestCase):
    def test_names_match_the_launcher(self):
        cases = [
            {"MPLC_V2_ARMS": "clean_ft", "MPLC_V2_FREEZE_TE": "0", "MPLC_V2_LR": "3e-6",
             "MPLC_V2_ALIGN_WEIGHT": "10", "MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_SEED": "1"},
            {"MPLC_V2_ARMS": "mplc", "MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_ALIGN_WEIGHT": "0",
             "MPLC_V2_DEFENSE_LOSS": "hinge", "MPLC_V2_MULTI_POSITIVE": "0", "MPLC_V2_FREEZE_TE": "0"},
            {"MPLC_V2_ARMS": "mplc", "MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_FREEZE_TE": "0",
             "MPLC_V2_NUM_EPOCHS": "12", "MPLC_V2_VAL_EVERY": "3", "MPLC_V2_SEED": "2"},
            {"MPLC_V2_ARMS": "plain_at", "MPLC_V2_ALIGN_WEIGHT": "10", "MPLC_V2_LR": "3e-6",
             "MPLC_V2_NUM_EPOCHS": "12", "MPLC_V2_VAL_EVERY": "3"},
            {"MPLC_V2_ARMS": "fare", "MPLC_V2_ALIGN_WEIGHT": "0.1", "MPLC_V2_FREEZE_TE": "0",
             "MPLC_V2_NUM_EPOCHS": "12", "MPLC_V2_VAL_EVERY": "3"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for env in cases:
                with self.subTest(env=env):
                    result = run_script(TRAIN_SCRIPT, {"MPLC_V2_DRY_RUN": "1", "MPLC_V2_RUN_ROOT": tmp, **env})
                    self.assertEqual(result.returncode, 0, result.stdout)
                    self.assertIn(f"--save_dir={step2.run_name(env)}", result.stdout.split())


class LoadStep1Tests(unittest.TestCase):
    def test_reads_mplc_star_noise_schedule_and_star_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            finish_step1(tmp)
            step1 = step2.load_step1(Path(tmp), BATCH, 200)
        # Every screen scores 50, so each stage keeps its preferred option: linf, 8, 1e-5, 1.0.
        self.assertEqual(step1.star, {"MPLC_V2_ATTACK_MIX": "linf", "MPLC_V2_FREEZE_TE": "8",
                                      "MPLC_V2_LR": "1e-5", "MPLC_V2_ALIGN_WEIGHT": "1.0"})
        self.assertEqual((step1.noise, step1.epochs, step1.val_every, step1.budget), (0.0, EPOCHS, 3, 5.0))
        self.assertEqual(step1.star_screen.robust, 50.0)

    def test_unfinished_missing_or_resized_step1_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(SystemExit, "mplc_v2_step1.sh"):
                step2.load_step1(root, BATCH, 200)
            sweep.ensure_config(root, EPOCHS, 5.0, BATCH, 200, 3)
            with self.assertRaisesRegex(SystemExit, "not finished"):
                step2.load_step1(root, BATCH, 200)
            finish_step1(root)
            with self.assertRaisesRegex(SystemExit, "batch"):
                step2.load_step1(root, BATCH + 4, 200)


if __name__ == "__main__":
    unittest.main()
