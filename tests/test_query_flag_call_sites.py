"""Query-side call sites must encode images independently (queryflag=1).

Covers Task 2 of the mplc-fixes plan: RankPGDAttack's internal descriptor
extraction, RetrievalAttackProxy.forward, train_loop.compute_attack_losses,
and the rank_eval clean/database feature extraction helpers all have to pass
the right queryflag through to the model.
"""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from src.rank_attacks import RankAttackConfig, RankPGDAttack
from src.targets import RetrievalAttackBatch


class RecordingModel(nn.Module):
    """A tiny differentiable model that records every queryflag it is called with."""

    def __init__(self):
        super().__init__()
        self.flags = []
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, inputs, queryflag=0):
        self.flags.append(queryflag)
        flat = inputs.flatten(1)
        return torch.stack([flat.mean(dim=1), flat.mean(dim=1) * 0.0], dim=1) * self.weight


def make_targets(batch):
    return RetrievalAttackBatch(
        query_indices=torch.arange(batch),
        clean_query_descriptors=torch.zeros(batch, 2),
        positive_descriptors=torch.tensor([[10.0, 0.0]]).repeat(batch, 1),
        negative_descriptors=torch.tensor([[[0.0, 0.0]]]).repeat(batch, 1, 1),
    )


class QueryFlagCallSiteTests(unittest.TestCase):
    def test_rank_pgd_attack_uses_query_flag(self):
        model = RecordingModel()
        attack = RankPGDAttack(model, RankAttackConfig(epsilon=0.1, steps=2, device="cpu"))
        attack(torch.zeros(2, 3, 4, 4), make_targets(2))

        self.assertTrue(model.flags)
        self.assertTrue(all(flag == 1 for flag in model.flags))

    def test_retrieval_attack_proxy_uses_query_flag(self):
        from src.attacks import RetrievalAttackProxy

        model = RecordingModel()
        proxy = RetrievalAttackProxy(model, 0.1, False, "cpu")
        proxy.set_targets(make_targets(2))
        proxy(torch.rand(2, 3, 4, 4))

        self.assertEqual(model.flags, [1])

    def test_compute_attack_losses_uses_query_flag(self):
        from src.train_loop import compute_attack_losses

        class Identity(nn.Module):
            def forward(self, inputs, attack_targets):
                return inputs

        model = RecordingModel()
        args = SimpleNamespace(
            device="cpu",
            defense_loss="hinge",
            adv_margin=0.1,
            adv_loss_weight=1.0,
            adv_align_weight=0.05,
            maximize_attack=False,
        )
        compute_attack_losses(model, torch.zeros(2, 3, 4, 4), make_targets(2), [Identity()], args)

        self.assertEqual(model.flags, [1])


class RankEvalQueryFlagTests(unittest.TestCase):
    def test_extract_clean_query_features_passes_queryflag_one(self):
        import src.rank_eval as rank_eval

        recorded = {}

        def fake_extract_features_for_models(args, eval_ds, models, dataset_indices, desc, test_method, batch_size, queryflag=0):
            recorded["queryflag"] = queryflag
            return {"model": np.zeros((len(list(dataset_indices)), 2), dtype=np.float32)}, {}, 0.0

        original = rank_eval.extract_features_for_models
        rank_eval.extract_features_for_models = fake_extract_features_for_models
        try:
            eval_ds = SimpleNamespace(database_num=3, queries_num=2)
            args = SimpleNamespace(infer_batch_size=2, test_method="hard_resize")
            rank_eval.extract_clean_query_features(args, eval_ds, model=None)
        finally:
            rank_eval.extract_features_for_models = original

        self.assertEqual(recorded["queryflag"], 1)

    def test_extract_database_features_passes_queryflag_one(self):
        import src.rank_eval as rank_eval

        recorded = {}

        def fake_extract_features_for_models(args, eval_ds, models, dataset_indices, desc, test_method, batch_size, queryflag=0):
            recorded["queryflag"] = queryflag
            return {"model": np.zeros((len(list(dataset_indices)), 2), dtype=np.float32)}, {}, 0.0

        original = rank_eval.extract_features_for_models
        rank_eval.extract_features_for_models = fake_extract_features_for_models
        try:
            eval_ds = SimpleNamespace(database_num=3, queries_num=2)
            args = SimpleNamespace(infer_batch_size=2, test_method="hard_resize")
            rank_eval.extract_database_features(args, eval_ds, model=None)
        finally:
            rank_eval.extract_features_for_models = original

        self.assertEqual(recorded["queryflag"], 1)


if __name__ == "__main__":
    unittest.main()
