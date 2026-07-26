import sys
import unittest
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "submodules" / "perceptual-advex"))

from src.attacks import RetrievalAttackWrapper
from src.config import amp_autocast


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, inputs, queryflag=0):
        return self.linear(inputs.flatten(1)[:, :4])


class RecordingBackend(nn.Module):
    def __init__(self, proxy_model):
        super().__init__()
        self.proxy_model = proxy_model
        self.saw_autocast = None

    def forward(self, inputs, labels):
        self.saw_autocast = torch.is_autocast_enabled("cuda")
        return inputs


class RecordingAttack(RetrievalAttackWrapper):
    backend_cls = RecordingBackend


class AmpAutocastTests(unittest.TestCase):
    def test_disable_overrides_ambient_autocast(self):
        with amp_autocast(True, "cuda"):
            self.assertTrue(torch.is_autocast_enabled("cuda"))
            with amp_autocast(False, "cuda"):
                self.assertFalse(torch.is_autocast_enabled("cuda"))
            self.assertTrue(torch.is_autocast_enabled("cuda"))

    def test_cpu_device_is_noop(self):
        with amp_autocast(True, "cpu"):
            self.assertFalse(torch.is_autocast_enabled("cuda"))


class AttackGenerationAutocastTests(unittest.TestCase):
    def test_backend_runs_outside_autocast_region(self):
        attack = RecordingAttack(FakeModel(), margin=0.1, mixed_precision=True, device="cuda")
        inputs = torch.rand(2, 3, 8, 8)
        with amp_autocast(True, "cuda"):
            attack(inputs, targets=object())
        self.assertFalse(attack.backend.saw_autocast)


if __name__ == "__main__":
    unittest.main()
