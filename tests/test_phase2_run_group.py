import tempfile
import unittest
from pathlib import Path

from src.phase2_run_group import (
    TERMINAL_STATES,
    classify_config_run,
    create_pilot_run_group,
    load_pilot_manifest,
)


class PilotRunGroupTests(unittest.TestCase):
    def manifest_values(self):
        return {
            "taus": ["0.01", "0.05"],
            "ks": ["1", "5"],
            "pools": ["0", "4096"],
            "seed": 0,
            "ramp_epochs": 5,
            "abort_knn": "0.15",
            "base_checkpoint": "checkpoints/SuperVLAD_base.pth",
        }

    def test_create_group_materializes_manifest_and_all_eight_config_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = create_pilot_run_group(Path(directory), self.manifest_values())

            manifest = load_pilot_manifest(run_root)

            self.assertEqual(len(manifest["configurations"]), 8)
            self.assertEqual(manifest["taus"], ["0.01", "0.05"])
            for config in manifest["configurations"]:
                self.assertTrue((run_root / config["label"]).is_dir())

    def test_existing_group_rejects_changed_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = create_pilot_run_group(Path(directory), self.manifest_values())
            changed = self.manifest_values()
            changed["taus"] = ["0.05"]

            with self.assertRaisesRegex(ValueError, "does not match"):
                create_pilot_run_group(Path(directory), changed, run_root=run_root)

    def test_best_model_without_terminal_status_is_resumable_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            (config_dir / "best_model.pth").touch()
            (config_dir / "last_model.pth").touch()

            state = classify_config_run(config_dir)

            self.assertEqual(state.state, "resumable")
            self.assertEqual(state.checkpoint, config_dir / "last_model.pth")

    def test_initial_validation_checkpoint_can_resume_epoch_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            checkpoint = config_dir / "initial_validation_model.pth"
            checkpoint.touch()

            state = classify_config_run(config_dir)

            self.assertEqual(state.state, "resumable")
            self.assertEqual(state.checkpoint, checkpoint)

    def test_terminal_status_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            config_dir = Path(directory)
            (config_dir / "run_status.json").write_text(
                '{"state": "early_stopped"}\n', encoding="utf-8"
            )

            state = classify_config_run(config_dir)

            self.assertIn(state.state, TERMINAL_STATES)
            self.assertIsNone(state.checkpoint)


if __name__ == "__main__":
    unittest.main()
