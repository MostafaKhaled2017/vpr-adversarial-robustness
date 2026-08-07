import unittest

import torch
from torch import nn

from src.config import (
    denormalize_imagenet,
    get_normalized_bounds,
    normalized_epsilon_to_raw_pixels,
)
from src.losses import compute_attack_score, query_is_correct
from src.rank_attacks import RankAPGDLinfAttack, RankAttackConfig, RankPGDAttack
from src.targets import RetrievalAttackBatch, select_rank_targets


class TinyDescriptorModel(nn.Module):
    def forward(self, inputs, queryflag=0):
        flat = inputs.flatten(1)
        return torch.stack([flat.mean(dim=1), flat[:, 0]], dim=1)


class OnePixelDescriptorModel(nn.Module):
    def forward(self, inputs, queryflag=0):
        return inputs[:, 0, 0, 0].view(-1, 1)


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

        raw = denormalize_imagenet(result.adversarial)
        self.assertTrue(torch.all(raw >= -1e-6))
        self.assertTrue(torch.all(raw <= 1.0 + 1e-6))

    def test_normalized_epsilon_conversion_reports_raw_pixel_scale(self):
        converted = normalized_epsilon_to_raw_pixels(0.1)

        self.assertAlmostEqual(converted["per_channel_raw_01"]["R"], 0.0229)
        self.assertAlmostEqual(converted["max_raw_01"], 0.0229)
        self.assertAlmostEqual(converted["max_raw_255"], 5.8395)

    def test_audit_metadata_records_loss_sign_effect(self):
        clean = torch.full((1, 3, 4, 4), 0.5)
        targets = RetrievalAttackBatch(
            query_indices=torch.tensor([0]),
            clean_query_descriptors=torch.tensor([[0.5]]),
            positive_descriptors=torch.tensor([[1.0]]),
            negative_descriptors=torch.tensor([[[0.0]]]),
        )
        attack = RankPGDAttack(
            OnePixelDescriptorModel(),
            RankAttackConfig(epsilon=0.1, steps=1, restarts=1, step_size=0.1, norm="linf", device="cpu", audit=True),
        )

        result = attack(clean, targets)

        self.assertGreater(result.metadata["positive_distance_after"].item(), result.metadata["positive_distance_before"].item())
        self.assertLess(
            result.metadata["hard_negative_distance_after"].item(),
            result.metadata["hard_negative_distance_before"].item(),
        )
        self.assertGreater(result.metadata["gradient_norm"].item(), 0.0)

    def test_trace_rows_are_emitted_for_selected_queries(self):
        clean = torch.full((1, 3, 4, 4), 0.5)
        targets = RetrievalAttackBatch(
            query_indices=torch.tensor([7]),
            clean_query_descriptors=torch.tensor([[0.5]]),
            positive_descriptors=torch.tensor([[1.0]]),
            negative_descriptors=torch.tensor([[[0.0]]]),
        )
        attack = RankPGDAttack(
            OnePixelDescriptorModel(),
            RankAttackConfig(
                epsilon=0.1,
                steps=2,
                restarts=1,
                step_size=0.05,
                norm="linf",
                device="cpu",
                trace_query_indices=(7,),
            ),
        )

        result = attack(clean, targets)

        self.assertIsNotNone(result.traces)
        self.assertEqual([row["step"] for row in result.traces], [0, 1, 2])
        self.assertTrue(all(row["query_index"] == 7 for row in result.traces))
        self.assertTrue(all("descriptor" in row for row in result.traces))
        self.assertLessEqual(max(row["perturbation_linf_normalized"] for row in result.traces), 0.100001)
        self.assertGreater(result.traces[-1]["perturbation_linf_raw"], 0.0)

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


class MultiPositiveTargetTests(unittest.TestCase):
    """Task 2.1 — retrieval succeeds if *any* positive outranks the negatives.

    The single hardest positive is the wrong target: it may be visually weak, and beating
    it says nothing about whether the query still retrieves the place. The attack must beat
    the *closest* valid positive instead.
    """

    def make_batch(self):
        # One place, four images: index 0 is the query, 1..3 are positives at increasing
        # distance from it.
        descriptors = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                    [5.0, 0.0],
                ]
            ]
        )
        place_ids = torch.tensor([[7, 7, 7, 7]])
        return descriptors, place_ids

    def make_two_place_batch(self):
        descriptors = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]],
                [[0.0, 9.0], [1.0, 9.0], [3.0, 9.0]],
            ]
        )
        place_ids = torch.tensor([[7, 7, 7], [8, 8, 8]])
        return descriptors, place_ids

    def test_multi_positive_mode_keeps_every_positive(self):
        descriptors, place_ids = self.make_batch()

        targets = select_rank_targets(descriptors, place_ids, adv_negatives=1, multi_positive=True)

        self.assertIsNone(targets, "a single-place batch has no cross-place negatives")

    def test_multi_positive_mode_keeps_every_positive_for_each_place(self):
        descriptors, place_ids = self.make_two_place_batch()

        targets = select_rank_targets(descriptors, place_ids, adv_negatives=1, multi_positive=True)

        self.assertIsNotNone(targets)
        self.assertEqual(targets.positive_bank.shape, (2, 2, 2))
        self.assertTrue(torch.equal(targets.positive_bank[0], torch.tensor([[1.0, 0.0], [4.0, 0.0]])))
        self.assertTrue(torch.all(targets.positive_bank_mask))

    def test_single_positive_mode_still_keeps_only_the_hardest(self):
        descriptors, place_ids = self.make_two_place_batch()

        targets = select_rank_targets(descriptors, place_ids, adv_negatives=1)

        self.assertEqual(targets.positive_descriptors.shape, (2, 2))
        # The hardest (farthest) positive, which is the pre-Phase-2 behaviour.
        self.assertTrue(torch.equal(targets.positive_descriptors[0], torch.tensor([4.0, 0.0])))
        self.assertIsNone(targets.positive_mask)

    def test_masking_marks_places_with_fewer_positives(self):
        descriptors, place_ids = self.make_two_place_batch()
        # The second place contributes only one usable positive.
        image_mask = torch.tensor([[True, True, True], [True, True, False]])

        targets = select_rank_targets(
            descriptors,
            place_ids,
            adv_negatives=1,
            multi_positive=True,
            image_mask=image_mask,
        )

        self.assertEqual(targets.positive_bank.shape, (2, 2, 2))
        self.assertTrue(torch.equal(targets.positive_bank_mask, torch.tensor([[True, True], [True, False]])))

    def test_subset_carries_the_positive_mask(self):
        descriptors, place_ids = self.make_two_place_batch()
        image_mask = torch.tensor([[True, True, True], [True, True, False]])
        targets = select_rank_targets(
            descriptors, place_ids, adv_negatives=1, multi_positive=True, image_mask=image_mask
        )

        kept = targets.subset(torch.tensor([False, True]))

        self.assertEqual(len(kept), 1)
        self.assertTrue(torch.equal(kept.positive_bank_mask, torch.tensor([[True, False]])))


class MultiPositiveAttackScoreTests(unittest.TestCase):
    def test_score_targets_the_closest_valid_positive(self):
        query = torch.tensor([[0.0, 0.0]])
        positives = torch.tensor([[[1.0, 0.0], [4.0, 0.0]]])
        negatives = torch.tensor([[[3.0, 0.0]]])

        score = compute_attack_score(query, positives, negatives, margin=0.1)

        # margin + closest positive distance (1.0) - closest negative distance (3.0)
        self.assertAlmostEqual(float(score[0]), 0.1 + 1.0 - 3.0, places=5)

    def test_masked_positives_are_ignored_even_when_closer(self):
        query = torch.tensor([[0.0, 0.0]])
        positives = torch.tensor([[[1.0, 0.0], [4.0, 0.0]]])
        negatives = torch.tensor([[[3.0, 0.0]]])
        positive_mask = torch.tensor([[False, True]])

        score = compute_attack_score(query, positives, negatives, margin=0.1, positive_mask=positive_mask)

        self.assertAlmostEqual(float(score[0]), 0.1 + 4.0 - 3.0, places=5)

    def test_two_dimensional_positives_keep_the_legacy_result(self):
        query = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        positives = torch.tensor([[4.0, 0.0], [2.0, 1.0]])
        negatives = torch.tensor([[[3.0, 0.0]], [[5.0, 1.0]]])

        legacy = torch.norm(query - positives, dim=1) + 0.1 - torch.norm(
            query.unsqueeze(1) - negatives, dim=2
        ).min(dim=1).values
        score = compute_attack_score(query, positives, negatives, margin=0.1)

        self.assertTrue(torch.allclose(score, legacy))

    def test_query_is_correct_uses_the_closest_valid_positive(self):
        query = torch.tensor([[0.0, 0.0]])
        positives = torch.tensor([[[1.0, 0.0], [4.0, 0.0]]])
        negatives = torch.tensor([[[3.0, 0.0]]])

        self.assertTrue(bool(query_is_correct(query, positives, negatives)[0]))
        # With the near positive masked out the query no longer retrieves the place.
        masked = query_is_correct(query, positives, negatives, positive_mask=torch.tensor([[False, True]]))
        self.assertFalse(bool(masked[0]))
