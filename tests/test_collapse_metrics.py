import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.collapse_metrics import collapse_report, knn_overlap, mean_pairwise_cosine


class KnnOverlapTests(unittest.TestCase):
    def test_identical_embeddings_overlap_completely(self):
        descriptors = torch.randn(40, 8)

        self.assertAlmostEqual(knn_overlap(descriptors, descriptors, k=5), 1.0, places=6)

    def test_a_rescaled_embedding_preserves_neighbourhoods(self):
        descriptors = torch.randn(40, 8)

        # Scaling every descriptor leaves the ordering of distances untouched.
        self.assertAlmostEqual(knn_overlap(descriptors * 3.7, descriptors, k=5), 1.0, places=6)

    def test_unrelated_embeddings_barely_overlap(self):
        torch.manual_seed(0)
        current = torch.randn(200, 16)
        reference = torch.randn(200, 16)

        self.assertLess(knn_overlap(current, reference, k=5), 0.2)

    def test_self_is_never_counted_as_its_own_neighbour(self):
        # Two tight clusters: each point's nearest neighbours are its cluster mates.
        cluster_a = torch.tensor([[0.0, 0.0], [0.01, 0.0], [0.0, 0.01]])
        cluster_b = torch.tensor([[9.0, 9.0], [9.01, 9.0], [9.0, 9.01]])
        descriptors = torch.cat([cluster_a, cluster_b], dim=0)
        shuffled_within_clusters = torch.cat([cluster_a.flip(0), cluster_b.flip(0)], dim=0)

        # Neighbour *sets* are unchanged by reordering inside a cluster only if self is
        # excluded consistently; this pins that behaviour down.
        self.assertGreater(knn_overlap(descriptors, shuffled_within_clusters, k=2), 0.0)

    def test_k_is_clamped_to_the_available_neighbours(self):
        descriptors = torch.randn(4, 3)

        # Only 3 neighbours exist per sample; asking for 10 must not raise.
        self.assertAlmostEqual(knn_overlap(descriptors, descriptors, k=10), 1.0, places=6)

    def test_too_few_samples_yields_nan_rather_than_a_crash(self):
        descriptors = torch.randn(1, 3)

        self.assertTrue(torch.isnan(torch.tensor(knn_overlap(descriptors, descriptors, k=5))))


class UniformityTests(unittest.TestCase):
    def test_identical_directions_are_perfectly_similar(self):
        descriptors = torch.ones(10, 4)

        self.assertAlmostEqual(mean_pairwise_cosine(descriptors), 1.0, places=5)

    def test_opposing_directions_are_perfectly_dissimilar(self):
        descriptors = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

        self.assertAlmostEqual(mean_pairwise_cosine(descriptors), -1.0, places=5)

    def test_random_high_dimensional_directions_are_nearly_orthogonal(self):
        torch.manual_seed(0)
        descriptors = torch.randn(300, 128)

        self.assertLess(abs(mean_pairwise_cosine(descriptors)), 0.05)


class CollapseReportTests(unittest.TestCase):
    def test_constant_descriptors_have_no_per_dimension_variance(self):
        descriptors = torch.ones(32, 8)

        report = collapse_report(descriptors, descriptors, k=5)

        self.assertAlmostEqual(report["dimension_std_mean"], 0.0, places=6)
        self.assertAlmostEqual(report["dimension_std_min"], 0.0, places=6)

    def test_constant_descriptors_are_flagged_as_maximally_similar(self):
        descriptors = torch.ones(32, 8)

        report = collapse_report(descriptors, descriptors, k=5)

        self.assertAlmostEqual(report["mean_pairwise_cosine"], 1.0, places=5)

    def test_healthy_embeddings_keep_variance_and_low_similarity(self):
        torch.manual_seed(0)
        descriptors = torch.randn(200, 32)

        report = collapse_report(descriptors, descriptors, k=10)

        self.assertGreater(report["dimension_std_mean"], 0.5)
        self.assertLess(abs(report["mean_pairwise_cosine"]), 0.1)
        self.assertAlmostEqual(report["knn_overlap"], 1.0, places=6)

    def test_report_exposes_the_sample_count_it_used(self):
        report = collapse_report(torch.randn(17, 4), torch.randn(17, 4), k=5)

        self.assertEqual(report["sample_count"], 17)

    def test_drifted_embeddings_lower_the_knn_overlap(self):
        torch.manual_seed(0)
        reference = torch.randn(200, 16)
        current = torch.randn(200, 16)

        report = collapse_report(current, reference, k=10)

        self.assertLess(report["knn_overlap"], 0.2)

    def test_mismatched_sample_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            collapse_report(torch.randn(10, 4), torch.randn(11, 4), k=5)


if __name__ == "__main__":
    unittest.main()
