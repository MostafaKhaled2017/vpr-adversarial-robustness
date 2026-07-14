import unittest

import numpy as np

from src.retrieval_metrics import (
    attack_success_metrics,
    compute_recalls_from_features,
    nearest_positive_ranks,
    nearest_positive_rank_from_distances,
    prepare_distance_database,
    rank_displacement_summary,
    squared_l2_distance_chunk,
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

    def test_nearest_positive_ranks_match_legacy_sort_across_chunks(self):
        rng = np.random.default_rng(5)
        database = rng.normal(size=(19, 7)).astype(np.float32)
        queries = rng.normal(size=(5, 7)).astype(np.float32)
        positives = [np.array([index, index + 5], dtype=np.int64) for index in range(5)]

        ranks = nearest_positive_ranks(database, queries, positives, chunk_size=2)
        prepared_database, database_norms = prepare_distance_database(database)
        all_distances = squared_l2_distance_chunk(prepared_database, database_norms, queries)
        legacy_ranks = []
        for distances, positive_indexes in zip(all_distances, positives):
            order = np.argsort(distances)
            inverse_ranks = np.empty_like(order)
            inverse_ranks[order] = np.arange(1, len(order) + 1)
            legacy_ranks.append(int(inverse_ranks[positive_indexes].min()))

        np.testing.assert_array_equal(ranks, np.asarray(legacy_ranks))

    def test_rank_helper_preserves_non_positive_tie_order(self):
        distances = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        positives = np.array([2], dtype=np.int64)
        order = np.argsort(distances)
        inverse_ranks = np.empty_like(order)
        inverse_ranks[order] = np.arange(1, len(order) + 1)

        rank = nearest_positive_rank_from_distances(distances, positives)

        self.assertEqual(rank, int(inverse_ranks[positives].min()))

    def test_rank_helper_returns_minus_one_without_positives(self):
        self.assertEqual(nearest_positive_rank_from_distances(np.array([0.0, 1.0]), []), -1)

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
