import sys
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.negative_pool import NegativePool
from src.targets import select_rank_targets
from src.train_loop import compute_attack_losses, compute_defense_loss, refresh_negative_pool


class TinyEncoder(nn.Module):
    """A differentiable stand-in for a descriptor model."""

    def __init__(self, descriptor_dim=4):
        super().__init__()
        self.projection = nn.Linear(12, descriptor_dim)

    def forward(self, inputs, queryflag=0):
        return self.projection(inputs.flatten(1))


class IdentityAttack(nn.Module):
    """Stands in for a perceptual attack without pulling in the backends."""

    def forward(self, inputs, targets):
        return inputs + 0.01


def make_args(**overrides):
    args = Namespace(
        device="cpu",
        mixed_precision=False,
        adv_margin=0.1,
        adv_loss_weight=1.0,
        adv_align_weight=0.05,
        adv_negatives=3,
        maximize_attack=False,
        multi_positive=True,
        defense_loss="listwise",
        listwise_tau=0.05,
        listwise_k=1,
        negative_pool_size=32,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class Phase2CompositionTests(unittest.TestCase):
    """The Phase 2 components have to compose, not just work individually.

    Multi-positive targets change the shape of the positive tensor, the negative pool
    replaces the negative tensor, and the listwise objective consumes both. A shape error
    anywhere in that chain would only surface hours into a training run.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.batch_size = 4
        self.images_per_place = 3
        self.model = TinyEncoder()
        self.images = torch.randn(self.batch_size, self.images_per_place, 3, 2, 2)
        self.place_ids = torch.arange(self.batch_size).unsqueeze(1).repeat(1, self.images_per_place)

    def descriptors(self):
        flat = self.images.reshape(self.batch_size * self.images_per_place, 3, 2, 2)
        with torch.no_grad():
            descriptors = self.model(flat)
        return descriptors, descriptors.reshape(self.batch_size, self.images_per_place, -1)

    def test_multi_positive_targets_flow_into_the_listwise_defense_loss(self):
        flat_descriptors, descriptor_view = self.descriptors()
        args = make_args()
        targets = select_rank_targets(descriptor_view, self.place_ids, args.adv_negatives, multi_positive=True)

        self.assertIsNotNone(targets)
        self.assertEqual(targets.positive_bank.shape[1], self.images_per_place - 1)

        adversarial = self.model(self.images[:, 0].reshape(self.batch_size, 3, 2, 2))
        loss = compute_defense_loss(adversarial, targets, args)

        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.model.projection.weight.grad)

    def test_the_negative_pool_replaces_the_mined_negatives(self):
        flat_descriptors, descriptor_view = self.descriptors()
        args = make_args()
        targets = select_rank_targets(descriptor_view, self.place_ids, args.adv_negatives, multi_positive=True)
        original_negatives = targets.negative_descriptors.clone()

        pool = NegativePool(capacity=32, descriptor_dim=flat_descriptors.shape[1])
        # Seed the pool with descriptors that are nearer than anything in the batch, so a
        # working integration must change the mined set.
        pool.push(targets.clean_query_descriptors + 1e-3, torch.full((self.batch_size,), 999))

        refreshed = refresh_negative_pool(
            pool,
            args,
            targets,
            flat_descriptors,
            self.place_ids.reshape(-1),
            self.place_ids,
        )

        self.assertEqual(refreshed.negative_descriptors.shape, original_negatives.shape)
        self.assertFalse(torch.allclose(refreshed.negative_descriptors, original_negatives))
        # The batch is pushed on every step, so the pool grows past its seed.
        self.assertGreater(len(pool), self.batch_size)

    def test_a_disabled_pool_leaves_the_targets_untouched(self):
        flat_descriptors, descriptor_view = self.descriptors()
        args = make_args(negative_pool_size=0)
        targets = select_rank_targets(descriptor_view, self.place_ids, args.adv_negatives, multi_positive=True)

        refreshed = refresh_negative_pool(
            NegativePool(capacity=0, descriptor_dim=flat_descriptors.shape[1]),
            args,
            targets,
            flat_descriptors,
            self.place_ids.reshape(-1),
            self.place_ids,
        )

        self.assertIs(refreshed, targets)

    def test_full_attack_loss_path_runs_with_every_component_enabled(self):
        flat_descriptors, descriptor_view = self.descriptors()
        args = make_args()
        targets = select_rank_targets(descriptor_view, self.place_ids, args.adv_negatives, multi_positive=True)
        pool = NegativePool(capacity=32, descriptor_dim=flat_descriptors.shape[1])
        targets = refresh_negative_pool(
            pool, args, targets, flat_descriptors, self.place_ids.reshape(-1), self.place_ids
        )

        query_inputs = self.images[targets.query_indices, 0]
        outputs = compute_attack_losses(self.model, query_inputs, targets, [IdentityAttack()], args)

        self.assertTrue(torch.isfinite(outputs["combined_adv_loss"]))
        self.assertTrue(torch.isfinite(outputs["adv_rank_loss"]))
        self.assertTrue(torch.isfinite(outputs["align_loss"]))
        outputs["combined_adv_loss"].backward()
        self.assertIsNotNone(self.model.projection.weight.grad)

    def test_hinge_defense_still_works_with_multi_positive_targets(self):
        flat_descriptors, descriptor_view = self.descriptors()
        args = make_args(defense_loss="hinge")
        targets = select_rank_targets(descriptor_view, self.place_ids, args.adv_negatives, multi_positive=True)

        adversarial = self.model(self.images[:, 0].reshape(self.batch_size, 3, 2, 2))
        loss = compute_defense_loss(adversarial, targets, args)

        self.assertTrue(torch.isfinite(loss))

    def test_single_positive_targets_still_work_with_the_listwise_defense(self):
        flat_descriptors, descriptor_view = self.descriptors()
        args = make_args(multi_positive=False)
        targets = select_rank_targets(descriptor_view, self.place_ids, args.adv_negatives)

        self.assertEqual(targets.positive_descriptors.dim(), 2)

        adversarial = self.model(self.images[:, 0].reshape(self.batch_size, 3, 2, 2))
        loss = compute_defense_loss(adversarial, targets, args)

        self.assertTrue(torch.isfinite(loss))

    def test_only_attack_correct_subsetting_keeps_the_mask_aligned(self):
        flat_descriptors, descriptor_view = self.descriptors()
        args = make_args()
        targets = select_rank_targets(descriptor_view, self.place_ids, args.adv_negatives, multi_positive=True)

        keep = torch.tensor([True, False, True, False][: len(targets)])
        subset = targets.subset(keep)

        self.assertEqual(subset.positive_bank.shape[0], int(keep.sum()))
        self.assertEqual(subset.positive_bank_mask.shape[0], int(keep.sum()))
        adversarial = self.model(self.images[subset.query_indices, 0])
        self.assertTrue(torch.isfinite(compute_defense_loss(adversarial, subset, args)))


if __name__ == "__main__":
    unittest.main()
