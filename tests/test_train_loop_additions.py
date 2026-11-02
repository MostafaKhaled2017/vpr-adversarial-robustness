import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.losses import compute_align_loss, compute_rank_loss, compute_relational_loss
from src.negative_bank import merge_bank_negatives
from src.targets import RetrievalAttackBatch
from src.train_loop import compute_attack_losses, compute_drift_loss


class Descriptor(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
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


def make_args(**overrides):
    values = dict(
        defense_loss="hinge", adv_margin=0.1, adv_loss_weight=0.5, adv_align_weight=2.0,
        maximize_attack=False, device="cpu", mixed_precision=False, relational_temperature=0.1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class AnchorKindTests(unittest.TestCase):
    def setUp(self):
        self.model, self.inputs = Descriptor(), torch.randn(2, 4)
        self.reference = torch.randn(2, 4)
        self.neighbours = torch.randn(2, 6, 4)
        with torch.no_grad():
            adversarial = self.model(self.inputs)
        self.rank = compute_rank_loss(adversarial, batch().positive_descriptors, batch().negative_descriptors, 0.1)
        self.iso = compute_align_loss(self.reference, adversarial)
        self.rel = compute_relational_loss(self.reference, adversarial, self.neighbours, 0.1)

    def losses(self, args, neighbours=None):
        return compute_attack_losses(
            self.model, self.inputs, batch(), [IdentityAttack()], args,
            align_reference=self.reference, relational_neighbours=neighbours,
        )

    def test_default_outputs_are_unchanged(self):
        outputs = self.losses(make_args())
        torch.testing.assert_close(outputs["combined_adv_loss"], 0.5 * self.rank + 2.0 * self.iso)
        torch.testing.assert_close(outputs["align_loss"], self.iso)
        self.assertEqual(float(outputs["relational_loss"]), 0.0)
        self.assertEqual(float(outputs["attack_logs"][0]["relational_loss"]), 0.0)

    def test_isotropic_kind_matches_default(self):
        default = self.losses(make_args())
        isotropic = self.losses(make_args(anchor_kind="isotropic"), self.neighbours)
        torch.testing.assert_close(isotropic["combined_adv_loss"], default["combined_adv_loss"])
        self.assertEqual(float(isotropic["relational_loss"]), 0.0)

    def test_relational_kind_drops_isotropic_term(self):
        outputs = self.losses(make_args(anchor_kind="relational"), self.neighbours)
        torch.testing.assert_close(outputs["combined_adv_loss"], 0.5 * self.rank + 2.0 * self.rel)
        torch.testing.assert_close(outputs["align_loss"], self.iso)
        torch.testing.assert_close(outputs["relational_loss"], self.rel)
        torch.testing.assert_close(outputs["attack_logs"][0]["relational_loss"], self.rel)

    def test_both_kind_sums_the_two(self):
        outputs = self.losses(make_args(anchor_kind="both"), self.neighbours)
        expected = 0.5 * self.rank + 2.0 * (self.iso + self.rel)
        torch.testing.assert_close(outputs["combined_adv_loss"], expected)
        torch.testing.assert_close(outputs["relational_loss"], self.rel)

    def test_relational_without_neighbours_raises(self):
        for kind in ("relational", "both"):
            with self.assertRaises(ValueError):
                self.losses(make_args(anchor_kind=kind))


class DriftLossTests(unittest.TestCase):
    def test_drift_is_zero_without_anchor_or_weight(self):
        model, images = Descriptor(), torch.randn(3, 4)
        clean = model(images)
        no_anchor = compute_drift_loss(None, images, clean, make_args(clean_drift_weight=1.0))
        no_weight = compute_drift_loss(Descriptor(seed=1), images, clean, make_args())
        self.assertEqual(float(no_anchor), 0.0)
        self.assertEqual(float(no_weight), 0.0)

    def test_drift_equals_mean_squared_distance_to_anchor(self):
        model, anchor, images = Descriptor(seed=0), Descriptor(seed=1), torch.randn(3, 4)
        clean = model(images)
        drift = compute_drift_loss(anchor, images, clean, make_args(clean_drift_weight=1.0))
        expected = compute_align_loss(anchor(images), clean)
        torch.testing.assert_close(drift, expected)
        self.assertGreater(float(drift.detach()), 0.0)
        drift.backward()
        self.assertIsNotNone(model.linear.weight.grad)
        self.assertIsNone(anchor.linear.weight.grad)


class BankNegativeTests(unittest.TestCase):
    def test_bank_negatives_merge_into_targets_before_the_loss(self):
        model, inputs = Descriptor(), torch.randn(2, 4)
        extra = torch.randn(2, 5, 4, dtype=torch.float64)
        merged = merge_bank_negatives(batch(), extra)
        self.assertEqual(tuple(merged.negative_descriptors.shape), (2, 8, 4))
        args = make_args(adv_align_weight=0.0, adv_loss_weight=1.0)
        outputs = compute_attack_losses(model, inputs, merged, [IdentityAttack()], args)
        with torch.no_grad():
            adversarial = model(inputs)
        expected = compute_rank_loss(adversarial, merged.positive_descriptors, merged.negative_descriptors, 0.1)
        torch.testing.assert_close(outputs["adv_rank_loss"], expected)


if __name__ == "__main__":
    unittest.main()
