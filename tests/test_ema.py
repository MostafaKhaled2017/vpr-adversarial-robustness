"""Exponential moving average of the weights: update rule, checkpoint layout and resume."""

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.checkpoints import load_training_state  # noqa: E402
from src.cli import parse_arguments  # noqa: E402
from src.train_loop import build_checkpoint_state, build_ema_model, update_ema  # noqa: E402

BASE = ["--eval_datasets_folder", "/tmp", "--device", "cpu", "--model=supervlad"]
DECAY = 0.9


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.register_buffer("count", torch.tensor([3], dtype=torch.int64))


def set_weights(model, value):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)


def checkpoint_for(model, optimizer, ema_model):
    return build_checkpoint_state(
        Namespace(tensorboard_dir="tensorboard"), model, optimizer, 0, None, None, 0.0, 0,
        iteration=7, ema_model=ema_model,
    )


class EmaTest(unittest.TestCase):
    def test_update_rule(self):
        model = Tiny()
        ema_model = build_ema_model(model, DECAY)
        start = {name: value.clone() for name, value in ema_model.state_dict().items()}
        set_weights(model, 2.0)
        update_ema(ema_model, model, DECAY)
        for name in ("linear.weight", "linear.bias"):
            expected = DECAY * start[name] + (1 - DECAY) * 2.0
            self.assertTrue(torch.allclose(ema_model.state_dict()[name], expected), name)

    def test_integer_buffer_is_copied(self):
        model = Tiny()
        ema_model = build_ema_model(model, DECAY)
        model.count.fill_(10)
        update_ema(ema_model, model, DECAY)
        self.assertEqual(ema_model.count.dtype, torch.int64)
        self.assertEqual(ema_model.count.item(), 10)

    def test_average_is_frozen_and_off_by_default(self):
        self.assertIsNone(build_ema_model(Tiny(), 0.0))
        ema_model = build_ema_model(Tiny(), DECAY)
        self.assertFalse(any(parameter.requires_grad for parameter in ema_model.parameters()))

    def test_checkpoint_carries_average_and_live_weights(self):
        model = nn.DataParallel(Tiny())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ema_model = build_ema_model(model, DECAY)
        set_weights(model, 2.0)
        update_ema(ema_model, model, DECAY)
        state = checkpoint_for(model, optimizer, ema_model)
        self.assertEqual(list(state["model_state_dict"]), list(model.state_dict()))
        self.assertTrue(torch.equal(state["model_state_dict"]["module.linear.weight"], ema_model.linear.weight))
        self.assertTrue(torch.equal(state["live_model_state_dict"]["module.linear.weight"], model.module.linear.weight))
        self.assertFalse(torch.equal(state["model_state_dict"]["module.linear.weight"], model.module.linear.weight))
        self.assertTrue(torch.equal(state["runtime_state"]["ema_model_state_dict"]["linear.weight"], ema_model.linear.weight))

    def test_checkpoint_without_average_is_unchanged(self):
        model = Tiny()
        state = checkpoint_for(model, torch.optim.SGD(model.parameters(), lr=0.1), None)
        self.assertNotIn("live_model_state_dict", state)
        self.assertNotIn("ema_model_state_dict", state["runtime_state"])

    def test_resume_continues_the_average(self):
        torch.manual_seed(0)
        model = nn.DataParallel(Tiny())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ema_model = build_ema_model(model, DECAY)
        set_weights(model, 2.0)
        update_ema(ema_model, model, DECAY)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last_model.pth"
            torch.save(checkpoint_for(model, optimizer, ema_model), path)
            resumed = nn.DataParallel(Tiny())
            state = load_training_state(path, resumed, torch.optim.SGD(resumed.parameters(), lr=0.1), strict=False)
        # Training resumes from the live weights, not the average.
        self.assertTrue(torch.equal(resumed.module.linear.weight, model.module.linear.weight))

        continued = build_ema_model(resumed, DECAY, state.runtime_state["ema_model_state_dict"])
        restarted = build_ema_model(resumed, DECAY)
        for live in (model, resumed):
            set_weights(live, 3.0)
        update_ema(ema_model, model, DECAY)
        update_ema(continued, resumed, DECAY)
        update_ema(restarted, resumed, DECAY)
        self.assertTrue(torch.allclose(continued.linear.weight, ema_model.linear.weight))
        self.assertFalse(torch.allclose(restarted.linear.weight, ema_model.linear.weight))

    def test_parser(self):
        self.assertEqual(parse_arguments(BASE).ema_decay, 0.0)
        self.assertEqual(parse_arguments(BASE + ["--ema_decay=0.999"]).ema_decay, 0.999)
        for value in ("1", "-0.1"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "--ema_decay"):
                parse_arguments(BASE + [f"--ema_decay={value}"])


if __name__ == "__main__":
    unittest.main()
