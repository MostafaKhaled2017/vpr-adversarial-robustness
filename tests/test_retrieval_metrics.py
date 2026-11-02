import unittest

import numpy as np

from src.retrieval_metrics import (
    attack_success_metrics,
    compute_recalls_from_features,
    nearest_positive_ranks,
    nearest_positive_rank_from_distances,
    per_query_rank_records,
    prepare_distance_database,
    rank_displacement_summary,
    rank_metric_bundle,
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

    def test_rank_displacement_percentiles_resist_outliers(self):
        clean_ranks = np.array([1, 1, 3, 8, 50, 2], dtype=np.int64)
        attacked_ranks = np.array([1, 4, 900, 12, 150, 2], dtype=np.int64)

        summary = rank_displacement_summary(clean_ranks, attacked_ranks, database_size=1000)

        # Displacements are [0, 3, 897, 4, 100, 0]: the mean is dominated by one outlier.
        self.assertEqual(summary["count"], 6)
        self.assertAlmostEqual(summary["mean"], 1004.0 / 6.0)
        self.assertAlmostEqual(summary["median"], 3.5)
        self.assertEqual(summary["max"], 897)
        self.assertAlmostEqual(summary["p90"], 498.5)
        self.assertAlmostEqual(summary["p95"], 697.75)
        self.assertAlmostEqual(summary["p99"], 857.15)
        self.assertAlmostEqual(summary["mean_normalized"], 1004.0 / 6.0 / 1000.0)
        self.assertEqual(summary["database_size"], 1000)

    def test_rank_displacement_boundary_crossing_probabilities(self):
        clean_ranks = np.array([1, 1, 3, 8, 50, 2], dtype=np.int64)
        attacked_ranks = np.array([1, 4, 900, 12, 150, 2], dtype=np.int64)

        summary = rank_displacement_summary(clean_ranks, attacked_ranks, database_size=1000)

        self.assertAlmostEqual(summary["p_cross_1"], 1.0 / 6.0)  # query 1: rank 1 -> 4
        self.assertAlmostEqual(summary["p_cross_5"], 1.0 / 6.0)  # query 2: rank 3 -> 900
        self.assertAlmostEqual(summary["p_cross_10"], 2.0 / 6.0)  # queries 2 and 3
        self.assertAlmostEqual(summary["p_cross_100"], 2.0 / 6.0)  # queries 2 and 4

    def test_rank_displacement_ignores_invalid_ranks(self):
        clean_ranks = np.array([1, -1, 3], dtype=np.int64)
        attacked_ranks = np.array([4, 9, -1], dtype=np.int64)

        summary = rank_displacement_summary(clean_ranks, attacked_ranks, database_size=10)

        self.assertEqual(summary["count"], 1)
        self.assertAlmostEqual(summary["mean"], 3.0)
        self.assertAlmostEqual(summary["p_cross_1"], 1.0)

    def test_rank_displacement_without_database_size_leaves_normalization_unset(self):
        summary = rank_displacement_summary(
            np.array([1, 2], dtype=np.int64),
            np.array([3, 4], dtype=np.int64),
        )

        self.assertIsNone(summary["mean_normalized"])
        self.assertIsNone(summary["database_size"])

    def test_empty_rank_displacement_reports_every_field(self):
        summary = rank_displacement_summary(
            np.array([-1], dtype=np.int64),
            np.array([-1], dtype=np.int64),
            database_size=10,
        )

        self.assertEqual(summary["count"], 0)
        for field in ("mean", "median", "max", "p90", "p95", "p99", "p_cross_1", "p_cross_5", "p_cross_10", "p_cross_100"):
            self.assertEqual(summary[field], 0, field)
        self.assertIsNone(summary["mean_normalized"])

    def test_per_query_rank_records_expose_summary_inputs(self):
        clean_ranks = nearest_positive_ranks(self.database, self.clean_queries, self.positives)
        attacked_ranks = nearest_positive_ranks(self.database, self.attacked_queries, self.positives)

        records = per_query_rank_records([11, 12], clean_ranks, attacked_ranks)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["query_id"], 11)
        self.assertEqual(records[0]["clean_rank"], int(clean_ranks[0]))
        self.assertEqual(records[0]["attacked_rank"], int(attacked_ranks[0]))
        self.assertTrue(records[0]["clean_correct_at_1"])
        self.assertFalse(records[0]["attacked_correct_at_1"])
        self.assertFalse(records[1]["clean_correct_at_1"])

    def test_per_query_rank_records_reject_length_mismatch(self):
        with self.assertRaises(ValueError):
            per_query_rank_records([1], np.array([1, 2]), np.array([1, 2]))

    def test_rank_metric_bundle_matches_individual_summaries(self):
        clean_ranks = nearest_positive_ranks(self.database, self.clean_queries, self.positives)
        attacked_ranks = nearest_positive_ranks(self.database, self.attacked_queries, self.positives)

        bundle = rank_metric_bundle(clean_ranks, attacked_ranks, database_size=len(self.database))

        self.assertEqual(
            bundle["rank_displacement"],
            rank_displacement_summary(clean_ranks, attacked_ranks, database_size=len(self.database)),
        )
        self.assertEqual(bundle["attack_success"], attack_success_metrics(clean_ranks, attacked_ranks))

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
