import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src import rank_eval
from src.retrieval_metrics import targeted_success_rate
from src.targets import retarget_attack_targets

DATABASE = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)


def untargeted():
    return [{"query_index": 0, "positive_index": 1, "positive_indexes": np.array([1]), "negative_indexes": np.array([0, 2])}]


class RetargetTests(unittest.TestCase):
    def test_target_is_kth_nearest_non_positive_and_competitors_exclude_it(self):
        queries = np.array([[0.1]], dtype=np.float32)
        (result,) = retarget_attack_targets(untargeted(), DATABASE, queries, target_rank=2, adv_negatives=2)
        # Non-positives by distance from 0.1: 0, 2, 3, 4 -> the 2nd is database row 2.
        self.assertEqual(result["target_index"], 2)
        np.testing.assert_array_equal(result["competitor_indexes"], [0, 3])
        np.testing.assert_array_equal(result["negative_indexes"], [0, 2])

    def test_too_few_non_positives_raises(self):
        with self.assertRaisesRegex(RuntimeError, "target_rank"):
            retarget_attack_targets(untargeted(), DATABASE, np.array([[0.1]], dtype=np.float32), target_rank=5, adv_negatives=1)


class TargetedSuccessRateTests(unittest.TestCase):
    def test_counts_queries_whose_top1_is_the_target(self):
        queries = np.array([[1.1], [0.1]], dtype=np.float32)
        self.assertEqual(targeted_success_rate(DATABASE, queries, [1, 2]), 50.0)


class TargetedAttackBatchTests(unittest.TestCase):
    def test_target_is_the_only_negative_and_competitors_join_positives(self):
        class Dataset:
            database_num = 5

            def __getitem__(self, index):
                return torch.zeros(3, 2, 2), index

        target = {**untargeted()[0], "target_index": 3, "competitor_indexes": np.array([0, 2])}
        _, batch = rank_eval.make_attack_batch(
            Namespace(device="cpu"), Dataset(), [target], torch.from_numpy(DATABASE), np.zeros((1, 1), dtype=np.float32)
        )
        torch.testing.assert_close(batch.negative_descriptors, torch.tensor([[[3.0]]]))
        torch.testing.assert_close(batch.positive_bank[0, :, 0], torch.tensor([1.0, 0.0, 2.0]))


class TargetedContextTests(unittest.TestCase):
    def test_targeted_goal_retargets_and_caches_separately(self):
        context = {
            "sampled_gallery": False,
            "eval_ds": object(),
            "clean_features": {"base": {"database": DATABASE, "queries": np.array([[0.1]], dtype=np.float32)}},
            "valid_query_indices": np.array([0], dtype=np.int64),
            "target_cache": {},
        }
        original = rank_eval.build_attack_targets
        rank_eval.build_attack_targets = lambda *args, **kwargs: (untargeted(), np.array([0], dtype=np.int64))
        try:
            plain, _ = rank_eval.get_context_targets(Namespace(adv_negatives=1, max_queries=None, model_tags=["base"]), context, "base")
            targeted_args = Namespace(
                adv_negatives=1, max_queries=None, model_tags=["base"], rank_attack_goal="targeted", target_rank=1
            )
            targeted, _ = rank_eval.get_context_targets(targeted_args, context, "base")
        finally:
            rank_eval.build_attack_targets = original
        self.assertNotIn("target_index", plain[0])
        self.assertEqual(targeted[0]["target_index"], 0)


class TargetedCsvTests(unittest.TestCase):
    def test_flatten_rows_reports_targeted_success(self):
        rows = rank_eval.flatten_rows(
            {"msls": {"m": {"rank_pgd_linf_eps_0.0685_targeted": {
                "recalls": {"R@1": 5.0}, "recalls_str": "R@1: 5.0", "targeted_success_rate": 40.0,
            }}}},
            [1],
        )
        self.assertEqual(rows[0]["targeted_success_rate"], 40.0)


class TargetedValidationTests(unittest.TestCase):
    def _base_args(self):
        return rank_eval.build_parser().parse_args(
            [
                "--datasets",
                "msls",
                "--model_type",
                "supervlad",
                "--model_paths",
                "base.pth",
                "--epsilons",
                "0.01",
                "--rank_attack_goal",
                "targeted",
            ]
        )

    def test_targeted_rejects_max_dataset_samples(self):
        args = self._base_args()
        args.max_dataset_samples = 100

        with self.assertRaisesRegex(ValueError, "--rank_attack_goal targeted"):
            rank_eval.validate_arguments(args)

    def test_targeted_rejects_audit_sample_database_size(self):
        args = self._base_args()
        args.audit_sample_database_size = 10

        with self.assertRaisesRegex(ValueError, "--rank_attack_goal targeted"):
            rank_eval.validate_arguments(args)

    def test_embshift_rejects_targeted_goal(self):
        args = self._base_args()
        args.rank_attack = "embshift_linf"

        with self.assertRaisesRegex(ValueError, "embshift_linf is untargeted"):
            rank_eval.validate_arguments(args)


if __name__ == "__main__":
    unittest.main()
