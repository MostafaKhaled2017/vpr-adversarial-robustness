import unittest

import torch
from torch import nn

from perceptual_adv_training.config import get_normalized_bounds
from perceptual_adv_training.rank_attacks import RankAPGDLinfAttack, RankAttackConfig, RankPGDAttack
from perceptual_adv_training.targets import RetrievalAttackBatch


class TinyDescriptorModel(nn.Module):
    def forward(self, inputs, queryflag=0):
        flat = inputs.flatten(1)
        return torch.stack([flat.mean(dim=1), flat[:, 0]], dim=1)


def make_targets(batch_size):
    return RetrievalAttackBatch(
        query_indices=torch.arange(batch_size),
        clean_query_descriptors=torch.zeros(batch_size, 2),
        positive_descriptors=torch.tensor([[10.0, 0.0]]).repeat(batch_size, 1),
        negative_descriptors=torch.tensor([[[0.0, 0.0]]]).repeat(batch_size, 1, 1),
    )


class RankAttackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = TinyDescriptorModel()

    def test_linf_projection_respects_epsilon(self):
        clean = torch.zeros(2, 3, 4, 4)
        attack = RankPGDAttack(
            self.model,
            RankAttackConfig(epsilon=0.05, steps=3, restarts=2, norm="linf", device="cpu"),
        )

        result = attack(clean, make_targets(clean.shape[0]))
        max_delta = (result.adversarial - clean).flatten(1).abs().max(dim=1).values

        self.assertTrue(torch.all(max_delta <= 0.050001))

    def test_l2_projection_respects_epsilon(self):
        clean = torch.zeros(2, 3, 4, 4)
        attack = RankPGDAttack(
            self.model,
            RankAttackConfig(epsilon=0.25, steps=4, restarts=2, norm="l2", device="cpu"),
        )

        result = attack(clean, make_targets(clean.shape[0]))
        l2_delta = (result.adversarial - clean).flatten(1).norm(p=2, dim=1)

        self.assertTrue(torch.all(l2_delta <= 0.250001))

    def test_adversarial_inputs_are_clamped_to_normalized_bounds(self):
        min_value, max_value = get_normalized_bounds("cpu")
        clean = min_value.expand(2, 3, 4, 4).clone() + 0.001
        attack = RankPGDAttack(
            self.model,
            RankAttackConfig(epsilon=5.0, steps=3, restarts=1, norm="linf", device="cpu"),
        )

        result = attack(clean, make_targets(clean.shape[0]))

        self.assertTrue(torch.all(result.adversarial >= min_value - 1e-6))
        self.assertTrue(torch.all(result.adversarial <= max_value + 1e-6))

    def test_best_loss_tracks_strongest_restart(self):
        clean = torch.zeros(3, 3, 4, 4)
        attack = RankAPGDLinfAttack(
            self.model,
            RankAttackConfig(epsilon=0.5, steps=5, restarts=3, norm="linf", device="cpu"),
        )

        result = attack(clean, make_targets(clean.shape[0]))
        recomputed_best = attack._rank_loss_per_sample(result.adversarial, make_targets(clean.shape[0])).detach()

        self.assertTrue(torch.all(result.metadata["best_loss"] >= result.metadata["final_loss"]))
        self.assertTrue(torch.allclose(result.metadata["best_loss"], recomputed_best.cpu(), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
