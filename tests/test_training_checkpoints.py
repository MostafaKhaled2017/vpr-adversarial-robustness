import argparse
import tempfile
import unittest
from pathlib import Path

import torch
import yaml
from torch import nn

from src.checkpoints import load_model_state_dict, load_model_weights, save_training_config


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
