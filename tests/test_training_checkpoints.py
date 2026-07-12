import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from perceptual_adv_training.checkpoints import load_model_state_dict, load_model_weights


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


if __name__ == "__main__":
    unittest.main()
