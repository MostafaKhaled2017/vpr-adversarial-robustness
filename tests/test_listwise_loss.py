import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.losses import compute_listwise_loss, compute_listwise_scores, compute_soft_ranks


def points(*values):
    """Descriptors laid out on a line so distances are readable by hand."""
    return torch.tensor([[value, 0.0] for value in values])


class SoftRankTests(unittest.TestCase):
    def test_soft_rank_matches_the_hard_rank_as_tau_goes_to_zero(self):
        query = points(0.0)
        positives = points(2.0).unsqueeze(0)  # (1, 1, 2)
        negatives = points(1.0, 3.0, 5.0).unsqueeze(0)  # (1, 3, 2)

        soft_ranks = compute_soft_ranks(query, positives, negatives, tau=1e-4)

        # Exactly one negative (at distance 1) is closer than the positive (at distance 2),
        # so the positive's true rank is 2.
        self.assertAlmostEqual(float(soft_ranks[0, 0]), 2.0, places=4)

    def test_soft_rank_is_one_when_the_positive_beats_every_negative(self):
        query = points(0.0)
        positives = points(1.0).unsqueeze(0)
        negatives = points(4.0, 6.0).unsqueeze(0)

        soft_ranks = compute_soft_ranks(query, positives, negatives, tau=1e-4)

        self.assertAlmostEqual(float(soft_ranks[0, 0]), 1.0, places=4)

    def test_soft_rank_counts_every_closer_negative(self):
        query = points(0.0)
        positives = points(9.0).unsqueeze(0)
        negatives = points(1.0, 2.0, 3.0).unsqueeze(0)

        soft_ranks = compute_soft_ranks(query, positives, negatives, tau=1e-4)

        self.assertAlmostEqual(float(soft_ranks[0, 0]), 4.0, places=4)

    def test_soft_rank_is_a_half_integer_at_a_tie(self):
        query = points(0.0)
        positives = points(2.0).unsqueeze(0)
        negatives = points(2.0).unsqueeze(0)

        soft_ranks = compute_soft_ranks(query, positives, negatives, tau=0.05)

        # sigmoid(0) = 0.5, so a tied negative contributes half a rank.
        self.assertAlmostEqual(float(soft_ranks[0, 0]), 1.5, places=5)


class ListwiseLossTests(unittest.TestCase):
    def test_loss_is_near_zero_when_the_best_positive_beats_all_negatives(self):
        query = points(0.0)
        positives = points(1.0).unsqueeze(0)
        negatives = points(5.0, 7.0).unsqueeze(0)

        loss = compute_listwise_loss(query, positives, negatives, tau=0.05, k=1)

        self.assertLess(float(loss), 1e-3)

    def test_loss_is_large_when_a_negative_outranks_every_positive(self):
        query = points(0.0)
        positives = points(5.0).unsqueeze(0)
        negatives = points(1.0).unsqueeze(0)

        loss = compute_listwise_loss(query, positives, negatives, tau=0.05, k=1)

        self.assertGreater(float(loss), 1.0)

    def test_larger_k_tolerates_a_positive_ranked_below_a_few_negatives(self):
        query = points(0.0)
        positives = points(4.0).unsqueeze(0)
        negatives = points(1.0, 2.0, 3.0).unsqueeze(0)

        # The positive sits at rank 4: a violation at k=1, acceptable at k=5.
        strict = compute_listwise_loss(query, positives, negatives, tau=0.05, k=1)
        lenient = compute_listwise_loss(query, positives, negatives, tau=0.05, k=5)

        self.assertGreater(float(strict), 1.0)
        self.assertLess(float(lenient), 1e-3)

    def test_only_the_best_positive_has_to_succeed(self):
        query = points(0.0)
        # One good positive and one hopeless one; retrieval succeeds on the good one.
        positives = points(1.0, 40.0).unsqueeze(0)
        negatives = points(5.0, 7.0).unsqueeze(0)

        loss = compute_listwise_loss(query, positives, negatives, tau=0.05, k=1)

        self.assertLess(float(loss), 1e-3)

    def test_masked_positives_cannot_satisfy_the_objective(self):
        query = points(0.0)
        positives = points(1.0, 40.0).unsqueeze(0)
        negatives = points(5.0, 7.0).unsqueeze(0)
        positive_mask = torch.tensor([[False, True]])

        loss = compute_listwise_loss(query, positives, negatives, tau=0.05, k=1, positive_mask=positive_mask)

        self.assertGreater(float(loss), 1.0)

    def test_gradient_flows_to_the_query(self):
        # The positive and the negative sit on different axes. Collinear points would make
        # the two distance gradients cancel exactly and say nothing about the loss.
        query = torch.tensor([[0.0, 0.0]], requires_grad=True)
        positives = torch.tensor([[[1.02, 0.0]]])
        negatives = torch.tensor([[[0.0, 1.0]]])

        compute_listwise_loss(query, positives, negatives, tau=0.05, k=1).backward()

        self.assertIsNotNone(query.grad)
        self.assertGreater(float(query.grad.abs().sum()), 0.0)

    def test_gradient_vanishes_when_the_distance_gap_dwarfs_tau(self):
        """Documents a real limit of sigmoid soft ranking, not a bug.

        Once |d_pos - d_neg| >> tau the sigmoid saturates and no gradient reaches the
        query, so the loss cannot recover a positive that has already been pushed far down
        the list. tau must be chosen against the descriptor distance scale; this is why
        --listwise_tau is in the Task 2.6 pilot grid and why Task 2.4 monitors collapse.
        """
        query = torch.tensor([[0.0, 0.0]], requires_grad=True)
        positives = torch.tensor([[[5.0, 0.0]]])
        negatives = torch.tensor([[[0.0, 1.0]]])

        loss = compute_listwise_loss(query, positives, negatives, tau=0.05, k=1)
        loss.backward()

        self.assertGreater(float(loss), 1.0)
        self.assertEqual(float(query.grad.abs().sum()), 0.0)

    def test_loss_is_the_mean_of_the_per_sample_scores(self):
        query = points(0.0, 0.0)
        positives = torch.stack([points(1.0), points(5.0)], dim=0)
        negatives = torch.stack([points(5.0), points(1.0)], dim=0)

        scores = compute_listwise_scores(query, positives, negatives, tau=0.05, k=1)
        loss = compute_listwise_loss(query, positives, negatives, tau=0.05, k=1)

        self.assertEqual(scores.shape, (2,))
        self.assertAlmostEqual(float(loss), float(scores.mean()), places=6)

    def test_two_dimensional_positives_are_accepted(self):
        query = points(0.0)
        positives = points(1.0)  # (1, 2) rather than (1, 1, 2)
        negatives = points(5.0).unsqueeze(0)

        loss = compute_listwise_loss(query, positives, negatives, tau=0.05, k=1)

        self.assertLess(float(loss), 1e-3)


if __name__ == "__main__":
    unittest.main()
