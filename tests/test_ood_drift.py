"""Drift constraint on unlabeled images: multiplier rule, loss term, folder batches, flags."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.cli import parse_arguments  # noqa: E402
from src.drift_images import DriftBatches, list_images  # noqa: E402
from src.train_loop import build_checkpoint_state, drift_penalty, update_drift_multiplier  # noqa: E402

BASE = ["--eval_datasets_folder", "/tmp", "--device", "cpu", "--model=supervlad",
        "--attack", "RankLinfAttack(model, epsilon=0.0685, steps=5)"]
INITIAL = BASE + ["--align_target", "initial", "--resume", "x.pth"]


class MultiplierTest(unittest.TestCase):
    def test_stays_zero_below_the_allowed_drift(self):
        multiplier = 0.0
        for drift in (0.0, 0.05, 0.099):
            multiplier = update_drift_multiplier(multiplier, drift, rho=0.1, lr=0.5)
            self.assertEqual(multiplier, 0.0)

    def test_grows_by_lr_times_excess_and_shrinks_back_to_zero(self):
        multiplier = update_drift_multiplier(0.0, 0.3, rho=0.1, lr=0.5)
        self.assertAlmostEqual(multiplier, 0.1)
        multiplier = update_drift_multiplier(multiplier, 0.5, rho=0.1, lr=0.5)
        self.assertAlmostEqual(multiplier, 0.3)
        self.assertEqual(update_drift_multiplier(multiplier, 0.0, rho=0.1, lr=10.0), 0.0)

    def test_penalty_uses_the_multiplier_before_the_update(self):
        current = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        anchor = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        drift, penalty, next_multiplier = drift_penalty(current, anchor, multiplier=0.2, rho=0.5, lr=1.0)
        self.assertAlmostEqual(drift.item(), 1.0)  # mean over images of the squared distance
        self.assertAlmostEqual(penalty.item(), 0.2 * 1.0)
        self.assertAlmostEqual(next_multiplier, 0.2 + 1.0 * (1.0 - 0.5))
        penalty.backward()
        self.assertTrue(torch.allclose(current.grad, 0.2 * 2 * current.detach() / 2))


class RuntimeStateTest(unittest.TestCase):
    def checkpoint(self, ood_drift_state):
        from argparse import Namespace

        model = torch.nn.Linear(2, 2)
        return build_checkpoint_state(
            Namespace(tensorboard_dir="t"), model, torch.optim.SGD(model.parameters(), lr=0.1), 0, None, None, 0.0, 0,
            ood_drift_state=ood_drift_state,
        )

    def test_multiplier_and_position_survive_a_resume(self):
        runtime = self.checkpoint({"multiplier": 0.25, "position": {"pass": 2, "offset": 5}})["runtime_state"]
        self.assertEqual(runtime["ood_drift_multiplier"], 0.25)
        self.assertEqual(runtime["ood_drift_position"], {"pass": 2, "offset": 5})

    def test_absent_without_the_constraint(self):
        runtime = self.checkpoint(None)["runtime_state"]
        self.assertNotIn("ood_drift_multiplier", runtime)


class FolderBatchesTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        (self.folder / "sub").mkdir()
        for index, path in enumerate((self.folder / "a.png", self.folder / "sub" / "b.PNG", self.folder / "c.jpg")):
            Image.fromarray(np.full((20, 30, 3), index * 60, dtype=np.uint8)).save(path)
        (self.folder / "notes.txt").write_text("not an image")

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_image_files_are_listed(self):
        self.assertEqual([path.name for path in list_images(self.folder)], ["a.png", "c.jpg", "b.PNG"])

    def test_batches_have_the_configured_size_and_continue_after_a_resume(self):
        batches = DriftBatches(self.folder, (8, 12), batch_size=2, seed=0, num_workers=0)
        first = batches.next()
        self.assertEqual(tuple(first.shape), (2, 3, 8, 12))
        second = batches.next()  # wraps into the second pass over the three images
        self.assertEqual(tuple(second.shape), (2, 3, 8, 12))
        state = batches.state_dict()
        self.assertEqual(state, {"pass": 1, "offset": 1})
        expected = batches.next()
        resumed = DriftBatches(self.folder, (8, 12), batch_size=2, seed=0, state=state, num_workers=0)
        self.assertTrue(torch.equal(resumed.next(), expected))

    def test_empty_folder_is_refused(self):
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(ValueError):
            DriftBatches(empty, (8, 8), batch_size=2, seed=0, num_workers=0)


class FlagTest(unittest.TestCase):
    def test_off_by_default(self):
        args = parse_arguments(INITIAL)
        self.assertIsNone(args.ood_drift_folder)
        self.assertEqual(args.ood_drift_batch, 16)

    def test_folder_needs_rho_lr_and_the_initial_anchor(self):
        folder = ["--ood_drift_folder", "images"]
        for extra in ([], ["--ood_drift_rho", "0.1"], ["--ood_drift_lr", "0.1"]):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "--ood_drift"):
                parse_arguments(INITIAL + folder + extra)
        with self.assertRaisesRegex(ValueError, "--align_target initial"):
            parse_arguments(BASE + folder + ["--ood_drift_rho", "0.1", "--ood_drift_lr", "0.1"])
        args = parse_arguments(INITIAL + folder + ["--ood_drift_rho", "0.1", "--ood_drift_lr", "0.01"])
        self.assertEqual((args.ood_drift_rho, args.ood_drift_lr), (0.1, 0.01))

    def test_rho_and_lr_need_the_folder(self):
        with self.assertRaisesRegex(ValueError, "--ood_drift_folder"):
            parse_arguments(INITIAL + ["--ood_drift_rho", "0.1"])


if __name__ == "__main__":
    unittest.main()
