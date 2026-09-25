import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src import components
from src.cli import parse_arguments
from src.targets import RetrievalAttackBatch
from src.train_loop import compute_attack_losses

BASE = [
    "--eval_datasets_folder", "/tmp", "--device", "cpu",
    "--model=supervlad", "--attack", "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)",
]


class Descriptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)

    def forward(self, x, queryflag=0):
        return self.linear(x.flatten(1))


class IdentityAttack(nn.Module):
    def forward(self, inputs, targets):
        return inputs


def batch():
    torch.manual_seed(0)
    return RetrievalAttackBatch(
        query_indices=torch.tensor([0, 1]),
        clean_query_descriptors=torch.zeros(2, 4),
        positive_descriptors=torch.randn(2, 4),
        negative_descriptors=torch.randn(2, 3, 4),
    )


ARGS = SimpleNamespace(
    defense_loss="hinge", adv_margin=0.1, adv_loss_weight=0.0, adv_align_weight=1.0,
    maximize_attack=False, device="cpu",
)


class AlignReferenceTests(unittest.TestCase):
    def test_default_aligns_to_batch_clean_descriptors(self):
        model, inputs = Descriptor(), torch.randn(2, 4)
        outputs = compute_attack_losses(model, inputs, batch(), [IdentityAttack()], ARGS)
        expected = model(inputs).pow(2).sum(dim=1).mean()
        torch.testing.assert_close(outputs["align_loss"], expected)

    def test_align_reference_replaces_the_align_target_only(self):
        model, inputs = Descriptor(), torch.randn(2, 4)
        reference = model(inputs).detach()
        outputs = compute_attack_losses(model, inputs, batch(), [IdentityAttack()], ARGS, align_reference=reference)
        self.assertEqual(float(outputs["align_loss"]), 0.0)


class AlignTargetCliTests(unittest.TestCase):
    def test_default_is_current(self):
        self.assertEqual(parse_arguments(BASE).align_target, "current")

    def test_initial_requires_resume(self):
        with self.assertRaisesRegex(ValueError, "requires --resume"):
            parse_arguments(BASE + ["--align_target", "initial"])
        args = parse_arguments(BASE + ["--align_target", "initial", "--resume", "w.pth"])
        self.assertEqual(args.align_target, "initial")


class BuildAnchorModelTests(unittest.TestCase):
    def test_anchor_loads_initial_model_from_save_dir(self):
        model = Descriptor()
        loaded = []

        def load_weights(target, path, args):
            loaded.append(path)
            nn.init.zeros_(target.linear.weight)

        with tempfile.TemporaryDirectory() as save_dir, mock.patch.object(
            components, "get_model_adapter", return_value=SimpleNamespace(load_weights=load_weights)
        ):
            anchor = components.build_anchor_model(SimpleNamespace(model="supervlad", save_dir=save_dir), model)

        self.assertEqual(loaded, [str(Path(save_dir) / "initial_model.pth")])
        self.assertIsNot(anchor, model)
        self.assertFalse(anchor.training)
        self.assertTrue(all(not p.requires_grad for p in anchor.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(float(anchor.linear.weight.abs().sum()), 0.0)
        self.assertNotEqual(float(model.linear.weight.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
