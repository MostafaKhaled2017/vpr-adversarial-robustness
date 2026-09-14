import argparse
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from src.checkpoints import (
    capture_rng_state,
    checkpoint_next_epoch,
    load_model_state_dict,
    load_model_weights,
    load_training_state,
    resolve_run_directory,
    restore_rng_state,
    save_training_config,
    write_run_status,
)


class TrainingCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.reference = nn.Linear(3, 2)
        with torch.no_grad():
            self.reference.weight.fill_(0.25)
            self.reference.bias.fill_(-0.5)

    def assert_loaded(self, checkpoint):
        target = nn.Linear(3, 2)
        load_model_state_dict(target, checkpoint)
        self.assertTrue(torch.equal(target.weight, self.reference.weight))
        self.assertTrue(torch.equal(target.bias, self.reference.bias))

    def test_supported_checkpoint_layouts(self):
        state_dict = self.reference.state_dict()
        self.assert_loaded(state_dict)
        self.assert_loaded({"model_state_dict": state_dict})
        self.assert_loaded({"state_dict": state_dict})
        self.assert_loaded({"state_dict": {f"module.{key}": value for key, value in state_dict.items()}})
        self.assert_loaded({"state_dict": {f"model.{key}": value for key, value in state_dict.items()}})

    def test_checkpoint_file_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "weights.pth"
            torch.save({"model_state_dict": self.reference.state_dict()}, path)
            target = nn.Linear(3, 2)
            load_model_weights(target, str(path), map_location="cpu")
            self.assertTrue(torch.equal(target.weight, self.reference.weight))

    def test_strict_loading_rejects_missing_keys(self):
        with self.assertRaises(RuntimeError):
            load_model_state_dict(nn.Linear(3, 2), {"weight": self.reference.weight})

    def test_resume_starts_after_the_last_completed_epoch(self):
        self.assertEqual(checkpoint_next_epoch({"epoch_num": 6}), 7)

    def test_initial_validation_checkpoint_resumes_at_epoch_zero(self):
        self.assertEqual(checkpoint_next_epoch({"epoch_num": -1}), 0)

    def test_full_training_state_restores_optimizer_scaler_and_runtime_state(self):
        source = nn.Linear(3, 2)
        optimizer = torch.optim.Adam(source.parameters(), lr=0.25)
        scaler = FakeScaler({"scale": 512.0})
        checkpoint = {
            "epoch_num": 4,
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": {"scale": 128.0},
            "best_r5": 9.5,
            "not_improved_num": 3,
            "runtime_state": {"iteration": 123, "negative_pool": {"size": 7}},
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_model.pth"
            torch.save(checkpoint, path)
            target = nn.Linear(3, 2)
            target_optimizer = torch.optim.Adam(target.parameters(), lr=1e-5)

            state = load_training_state(path, target, target_optimizer, scaler, map_location="cpu")

        self.assertEqual(state.next_epoch, 5)
        self.assertEqual(state.best_score, 9.5)
        self.assertEqual(state.not_improved, 3)
        self.assertEqual(state.runtime_state["iteration"], 123)
        self.assertTrue(state.runtime_state["resumed_from_checkpoint"])
        self.assertEqual(scaler.loaded_state, {"scale": 128.0})

    def test_rng_state_round_trip_restores_all_cpu_generators(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(())))

        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(())))

        self.assertEqual(actual, expected)

    def test_explicit_run_directory_bypasses_timestamp_nesting(self):
        with tempfile.TemporaryDirectory() as directory:
            exact = Path(directory) / "pilot" / "tau0.05_k1_pool4096"
            resolved = resolve_run_directory("logs", "ignored", exact, "2026-01-01_00-00-00")
            self.assertEqual(resolved, exact)

    def test_legacy_save_directory_keeps_timestamp_nesting(self):
        resolved = resolve_run_directory("logs", "experiment", None, "2026-01-01_00-00-00")
        self.assertEqual(resolved, Path("logs/experiment/2026-01-01_00-00-00"))

    def test_run_status_is_written_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_run_status(Path(directory), "early_stopped", final_epoch=12)
            payload = __import__("json").loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["state"], "early_stopped")
        self.assertEqual(payload["final_epoch"], 12)


class FakeScaler:
    def __init__(self, state):
        self.state = state
        self.loaded_state = None

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.loaded_state = state


class SaveTrainingConfigTests(unittest.TestCase):
    def test_saves_all_args_as_yaml_in_save_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir) / "run"
            args = argparse.Namespace(
                save_dir=str(save_dir),
                lr=1e-5,
                epochs_num=15,
                attack=["LinfAttack"],
                resume=None,
                mixed_precision=True,
                resize=(480, 640),
                dataset_path=Path("/data/gsv_cities"),
            )
            config_path = save_training_config(args)

            self.assertEqual(config_path, str(save_dir / "training_config.yaml"))
            with open(config_path) as config_file:
                config = yaml.safe_load(config_file)
            self.assertEqual(config["save_dir"], str(save_dir))
            self.assertEqual(config["lr"], 1e-5)
            self.assertEqual(config["epochs_num"], 15)
            self.assertEqual(config["attack"], ["LinfAttack"])
            self.assertIsNone(config["resume"])
            self.assertTrue(config["mixed_precision"])
            self.assertEqual(config["resize"], [480, 640])
            self.assertEqual(config["dataset_path"], "/data/gsv_cities")


if __name__ == "__main__":
    unittest.main()
