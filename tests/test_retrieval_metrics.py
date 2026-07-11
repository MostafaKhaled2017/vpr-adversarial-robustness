import unittest

import numpy as np

from perceptual_adv_training.retrieval_metrics import (
    attack_success_metrics,
    compute_recalls_from_features,
    nearest_positive_ranks,
    rank_displacement_summary,
)


class RetrievalMetricsTests(unittest.TestCase):
    def setUp(self):
        self.database = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        )
        self.clean_queries = np.array(
            [
                [0.1, 0.0],
                [1.9, 0.0],
            ],
            dtype=np.float32,
        )
        self.attacked_queries = np.array(
            [
                [2.1, 0.0],
                [1.1, 0.0],
            ],
            dtype=np.float32,
        )
        self.positives = [np.array([0]), np.array([1])]

    def test_recall_at_k(self):
        recalls = compute_recalls_from_features(self.database, self.clean_queries, self.positives, [1, 2])

        self.assertEqual(recalls["recalls"]["R@1"], 50.0)
        self.assertEqual(recalls["recalls"]["R@2"], 100.0)

    def test_nearest_positive_rank(self):
        ranks = nearest_positive_ranks(self.database, self.clean_queries, self.positives)

        np.testing.assert_array_equal(ranks, np.array([1, 2]))

    def test_rank_displacement(self):
        clean_ranks = nearest_positive_ranks(self.database, self.clean_queries, self.positives)
        attacked_ranks = nearest_positive_ranks(self.database, self.attacked_queries, self.positives)
        summary = rank_displacement_summary(clean_ranks, attacked_ranks)

        self.assertEqual(summary["count"], 2)
        self.assertEqual(summary["max"], 2)
        self.assertEqual(summary["mean"], 0.5)

    def test_attack_success_rates(self):
        clean_ranks = nearest_positive_ranks(self.database, self.clean_queries, self.positives)
        attacked_ranks = nearest_positive_ranks(self.database, self.attacked_queries, self.positives)
        success = attack_success_metrics(clean_ranks, attacked_ranks)

        self.assertEqual(success["clean_correct"]["successes"], 1)
        self.assertEqual(success["clean_correct"]["total"], 1)
        self.assertEqual(success["clean_correct"]["rate"], 100.0)
        self.assertEqual(success["all_valid"]["successes"], 1)
        self.assertEqual(success["all_valid"]["total"], 2)
        self.assertEqual(success["all_valid"]["rate"], 50.0)


if __name__ == "__main__":
    unittest.main()
