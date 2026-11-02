import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.attacks import instantiate_attacks
from src.cli import parse_attack_names
from src.rank_attacks import CombinedShiftRankPGDAttack, EmbeddingShiftPGDAttack, RankAttackConfig, RankPGDAttack
from src.targets import RetrievalAttackBatch


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, inputs, queryflag=0):
        flat = inputs.flatten(1)
        return torch.stack([flat.mean(dim=1), flat[:, 0]], dim=1) * self.weight


def args():
    return SimpleNamespace(device="cpu", mixed_precision=False, adv_margin=0.1, defense_loss="listwise",
                           listwise_tau=0.05, listwise_k=1, lpips_model=None)


def targets(batch):
    return RetrievalAttackBatch(
        query_indices=torch.arange(batch), clean_query_descriptors=torch.ones(batch, 2),
        positive_descriptors=torch.zeros(batch, 2), negative_descriptors=torch.full((batch, 1, 2), 5.0),
    )


class CombinedShiftRankAttackTests(unittest.TestCase):
    def test_objective_is_shift_plus_weighted_rank_score(self):
        model, config = Tiny(), RankAttackConfig(epsilon=0.1, steps=1, device="cpu")
        inputs = torch.rand(2, 3, 4, 4)
        shift = EmbeddingShiftPGDAttack(model, config)._rank_components(inputs, targets(2))[0]
        score = RankPGDAttack(model, config)._rank_components(inputs, targets(2))[0]
        combined = CombinedShiftRankPGDAttack(model, config, rank_weight=0.5)._rank_components(inputs, targets(2))[0]
        torch.testing.assert_close(combined, shift + 0.5 * score)

    def test_first_restart_starts_from_the_clean_input(self):
        config = RankAttackConfig(epsilon=0.1, steps=1, device="cpu")
        clean = torch.rand(2, 3, 4, 4)
        start = CombinedShiftRankPGDAttack(Tiny(), config)._initial_inputs(clean, 0)
        torch.testing.assert_close(start, clean)

    def test_wrapper_is_in_the_namespace_with_its_weight(self):
        attack = instantiate_attacks(
            Tiny(), ["CombinedShiftRankLinfAttack(model, epsilon=0.1, steps=2, rank_weight=0.5)"], args()
        )[0]
        self.assertEqual(attack.rank_weight, 0.5)
        self.assertEqual(parse_attack_names(["CombinedShiftRankLinfAttack(model, epsilon=0.1, steps=2)"]),
                         ["CombinedShiftRankLinfAttack"])

    def test_wrapper_respects_epsilon(self):
        attack = instantiate_attacks(Tiny(), ["CombinedShiftRankLinfAttack(model, epsilon=0.1, steps=3)"], args())[0]
        clean = torch.zeros(2, 3, 4, 4)
        attack.set_strength_scale(0.1)
        delta = (attack(clean, targets(2)) - clean).abs().max()
        self.assertLessEqual(float(delta), 0.01 + 1e-6)
        self.assertGreater(float(delta), 0.0)
        attack.set_strength_scale(1.0)
        self.assertLessEqual(float((attack(clean, targets(2)) - clean).abs().max()), 0.1 + 1e-6)
