import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.attacks import instantiate_attacks
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
        query_indices=torch.arange(batch), clean_query_descriptors=torch.zeros(batch, 2),
        positive_descriptors=torch.zeros(batch, 2), negative_descriptors=torch.full((batch, 1, 2), 5.0),
    )


class RankLinfTrainingAttackTests(unittest.TestCase):
    def test_respects_epsilon_and_ramp(self):
        model = Tiny()
        attack = instantiate_attacks(model, ["RankLinfAttack(model, epsilon=0.1, steps=3)"], args())[0]
        clean = torch.zeros(2, 3, 4, 4)
        attack.set_strength_scale(0.1)
        delta = (attack(clean, targets(2)) - clean).abs().max()
        self.assertLessEqual(float(delta), 0.01 + 1e-6)
        self.assertGreater(float(delta), 0.0)
        attack.set_strength_scale(1.0)
        self.assertLessEqual(float((attack(clean, targets(2)) - clean).abs().max()), 0.1 + 1e-6)

    def test_restores_parameter_grad_and_mode(self):
        model = Tiny().train()
        attack = instantiate_attacks(model, ["RankLinfAttack(model, epsilon=0.1, steps=2)"], args())[0]
        attack(torch.zeros(1, 3, 4, 4), targets(1))
        self.assertTrue(model.training)
        self.assertTrue(model.weight.requires_grad)
        self.assertIsNone(model.weight.grad)

    def test_name_is_supported_by_cli(self):
        from src.cli import parse_attack_names

        self.assertEqual(parse_attack_names(["RankLinfAttack(model, epsilon=0.1, steps=5)"]), ["RankLinfAttack"])

    def test_embedding_shift_attack_respects_epsilon_and_ramp(self):
        model = Tiny()
        attack = instantiate_attacks(model, ["EmbeddingShiftLinfAttack(model, epsilon=0.1, steps=3)"], args())[0]
        clean = torch.zeros(2, 3, 4, 4)
        attack.set_strength_scale(0.1)
        adversarial = attack(clean, targets(2))
        self.assertLessEqual(float((adversarial - clean).abs().max()), 0.01 + 1e-6)
        self.assertGreater(float((adversarial - clean).abs().max()), 0.0)

    def test_embedding_shift_attack_is_a_supported_training_attack(self):
        from src.cli import parse_attack_names

        self.assertEqual(parse_attack_names(["EmbeddingShiftLinfAttack(model, epsilon=0.1, steps=3)"]),
                         ["EmbeddingShiftLinfAttack"])
