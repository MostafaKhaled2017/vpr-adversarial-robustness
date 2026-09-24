import unittest

import numpy as np
import torch
from torch import nn

from src.rank_attacks import RankAttackConfig, RankPGDAttack
from src.targets import RetrievalAttackBatch


class LinearDescriptor(nn.Module):
    def forward(self, inputs, queryflag=0):
        flat = inputs.flatten(1)
        return torch.stack([flat.mean(dim=1), flat[:, 0]], dim=1)


def confident_targets(batch):
    # query descriptor starts at (0, 0); positive at distance 0, negative far away: hinge is 0
    return RetrievalAttackBatch(
        query_indices=torch.arange(batch),
        clean_query_descriptors=torch.zeros(batch, 2),
        positive_descriptors=torch.zeros(batch, 2),
        negative_descriptors=torch.tensor([[[5.0, 5.0]]]).repeat(batch, 1, 1),
    )


class RawScoreObjectiveTests(unittest.TestCase):
    def test_confidently_correct_queries_are_still_perturbed(self):
        attack = RankPGDAttack(LinearDescriptor(), RankAttackConfig(epsilon=0.05, steps=3, device="cpu"))
        clean = torch.zeros(2, 3, 4, 4)
        result = attack(clean, confident_targets(2))
        self.assertTrue(torch.all((result.adversarial - clean).flatten(1).abs().max(dim=1).values > 0))

    def test_loss_metadata_can_be_negative(self):
        attack = RankPGDAttack(LinearDescriptor(), RankAttackConfig(epsilon=0.01, steps=1, device="cpu"))
        result = attack(torch.zeros(1, 3, 4, 4), confident_targets(1))
        self.assertLess(float(result.metadata["best_loss"][0]), 0.0)


class AllPositiveBatchTests(unittest.TestCase):
    def test_make_attack_batch_carries_every_positive(self):
        from src import rank_eval
        from types import SimpleNamespace

        class Dataset:
            database_num = 4

            def __getitem__(self, index):
                return torch.zeros(3, 2, 2), index

        database = torch.arange(8, dtype=torch.float32).view(4, 2)
        targets = [
            {"query_index": 0, "positive_index": 1, "positive_indexes": np.array([1, 2]), "negative_indexes": np.array([0, 3])},
            {"query_index": 1, "positive_index": 3, "positive_indexes": np.array([3]), "negative_indexes": np.array([0, 1])},
        ]
        _, batch = rank_eval.make_attack_batch(SimpleNamespace(device="cpu"), Dataset(), targets, database, np.zeros((2, 2), np.float32))
        self.assertEqual(tuple(batch.positive_descriptors.shape), (2, 2, 2))
        self.assertEqual(batch.positive_mask.tolist(), [[True, True], [True, False]])
        self.assertTrue(torch.equal(batch.positive_descriptors[0, 1], database[2]))

    def test_legacy_targets_without_positive_indexes_still_work(self):
        from src import rank_eval
        from types import SimpleNamespace

        class Dataset:
            database_num = 4

            def __getitem__(self, index):
                return torch.zeros(3, 2, 2), index

        database = torch.arange(8, dtype=torch.float32).view(4, 2)
        targets = [{"query_index": 0, "positive_index": 1, "negative_indexes": np.array([0, 3])}]
        _, batch = rank_eval.make_attack_batch(SimpleNamespace(device="cpu"), Dataset(), targets, database, np.zeros((1, 2), np.float32))
        self.assertEqual(batch.positive_mask.tolist(), [[True]])


if __name__ == "__main__":
    unittest.main()
