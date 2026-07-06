import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rank_eval


class RankEvalInterfaceTests(unittest.TestCase):
    def test_default_model_tag_for_one_model(self):
        tags = rank_eval.resolve_model_tags(["base.pth"], None)

        self.assertEqual(tags, ["base"])

    def test_default_model_tags_for_two_models(self):
        tags = rank_eval.resolve_model_tags(["base.pth", "checkpoint.pth"], None)

        self.assertEqual(tags, ["base", "checkpoint"])

    def test_more_than_two_models_requires_tags(self):
        with self.assertRaises(ValueError):
            rank_eval.resolve_model_tags(["a.pth", "b.pth", "c.pth"], None)

    def test_mismatched_model_tag_count_fails(self):
        with self.assertRaises(ValueError):
            rank_eval.resolve_model_tags(["a.pth", "b.pth"], ["base"])

    def test_duplicate_model_tags_fail(self):
        with self.assertRaises(ValueError):
            rank_eval.resolve_model_tags(["a.pth", "b.pth"], ["base", "base"])

    def test_parser_uses_unified_dataset_and_model_args(self):
        parser = rank_eval.build_parser()

        self.assertIn("--datasets", parser._option_string_actions)
        self.assertIn("--models", parser._option_string_actions)
        self.assertIn("--model_tags", parser._option_string_actions)
        self.assertIn("--audit_attack_implementation", parser._option_string_actions)
        self.assertIn("--audit_output_json", parser._option_string_actions)
        self.assertIn("--audit_sample_database_size", parser._option_string_actions)
        self.assertIn("--trace_query_indices", parser._option_string_actions)
        self.assertIn("--trace_output_dir", parser._option_string_actions)
        self.assertIn("--diagnostics_output_dir", parser._option_string_actions)
        self.assertIn("--save_attack_images", parser._option_string_actions)
        self.assertIn("--save_attack_image_count", parser._option_string_actions)
        self.assertIn("--attack_image_output_dir", parser._option_string_actions)
        self.assertIn("--attack_image_amplification", parser._option_string_actions)
        self.assertNotIn("--resume", parser._option_string_actions)
        self.assertNotIn("--eval_dataset_name", parser._option_string_actions)

    def test_descriptor_norm_summary(self):
        summary = rank_eval.summarize_descriptor_norms(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32))

        self.assertEqual(summary["count"], 2)
        self.assertAlmostEqual(summary["mean"], 3.5)
        self.assertAlmostEqual(summary["min"], 2.0)
        self.assertAlmostEqual(summary["max"], 5.0)

    def test_flatten_rows_includes_dataset_and_model(self):
        rows = rank_eval.flatten_rows(
            {
                "msls": {
                    "base": {
                        "clean_all_queries": {
                            "recalls": {"R@1": 10.0},
                            "recalls_str": "R@1: 10.0",
                        }
                    }
                }
            },
            [1],
        )

        self.assertEqual(rows[0]["dataset"], "msls")
        self.assertEqual(rows[0]["model"], "base")
        self.assertEqual(rows[0]["condition"], "clean_all_queries")
        self.assertEqual(rows[0]["R@1"], 10.0)

    def test_default_outputs_use_timestamped_rank_eval_run_dir(self):
        args = Namespace(output_json=None, output_csv=None)

        output_json, output_csv, run_dir = rank_eval.build_output_paths(args, "2026-06-08_18-45-00")

        self.assertEqual(run_dir, Path("test/rank_eval/2026-06-08_18-45-00"))
        self.assertEqual(output_json, run_dir / "rank_eval_results.json")
        self.assertEqual(output_csv, run_dir / "rank_eval_results.csv")

    def test_explicit_outputs_are_written_inside_timestamped_run_dir(self):
        args = Namespace(
            output_json="test/rank_eval/msls_sped_nordland_rank_comparison.json",
            output_csv="test/rank_eval/msls_sped_nordland_rank_comparison.csv",
        )

        output_json, output_csv, run_dir = rank_eval.build_output_paths(args, "2026-06-08_18-45-00")

        self.assertEqual(run_dir, Path("test/rank_eval/2026-06-08_18-45-00"))
        self.assertEqual(output_json, run_dir / "msls_sped_nordland_rank_comparison.json")
        self.assertEqual(output_csv, run_dir / "msls_sped_nordland_rank_comparison.csv")

    def test_audit_output_path_uses_run_dir(self):
        args = Namespace(audit_attack_implementation=True, audit_output_json="custom_audit.json")

        output_path = rank_eval.build_audit_output_path(args, Path("test/rank_eval/run"))

        self.assertEqual(output_path, Path("test/rank_eval/run/custom_audit.json"))

    def test_trace_and_diagnostics_default_to_run_dir(self):
        args = Namespace(trace_output_dir=None, diagnostics_output_dir=None)
        run_dir = Path("test/rank_eval/run")

        self.assertEqual(rank_eval.build_trace_output_dir(args, run_dir), run_dir / "traces")
        self.assertEqual(rank_eval.build_diagnostics_output_dir(args, run_dir), run_dir / "diagnostics")

    def test_explicit_trace_and_diagnostics_dirs_are_used(self):
        args = Namespace(trace_output_dir="test/custom_traces", diagnostics_output_dir="test/custom_diagnostics")
        run_dir = Path("test/rank_eval/run")

        self.assertEqual(rank_eval.build_trace_output_dir(args, run_dir), Path("test/custom_traces"))
        self.assertEqual(rank_eval.build_diagnostics_output_dir(args, run_dir), Path("test/custom_diagnostics"))

    def test_attack_image_output_dir_defaults_to_run_dir(self):
        args = Namespace(attack_image_output_dir=None)
        run_dir = Path("test/rank_eval/run")

        self.assertEqual(rank_eval.build_attack_image_output_dir(args, run_dir), run_dir / "attack_images")

    def test_explicit_attack_image_output_dir_is_used(self):
        args = Namespace(attack_image_output_dir="test/custom_attack_images")
        run_dir = Path("test/rank_eval/run")

        self.assertEqual(rank_eval.build_attack_image_output_dir(args, run_dir), Path("test/custom_attack_images"))

    def test_attack_image_paths_are_deterministic(self):
        paths = rank_eval.attack_image_output_paths(
            Path("test/images"),
            "msls",
            "rank_pgd_linf",
            0.05,
            12,
            20.0,
        )

        self.assertEqual(paths["clean"], Path("test/images/msls/rank_pgd_linf/eps_0.05/12_clean.png"))
        self.assertEqual(paths["attacked"], Path("test/images/msls/rank_pgd_linf/eps_0.05/12_attacked.png"))
        self.assertEqual(
            paths["perturbation"],
            Path("test/images/msls/rank_pgd_linf/eps_0.05/12_perturbation_x20.png"),
        )
        self.assertEqual(paths["abs_heatmap"], Path("test/images/msls/rank_pgd_linf/eps_0.05/12_abs_heatmap.png"))

    def test_attack_image_query_selection_prioritizes_trace_indices(self):
        targets = [
            {"query_index": 3},
            {"query_index": 5},
            {"query_index": 7},
        ]

        selected = rank_eval.select_attack_image_query_indices(targets, [7, 4, 3], 2)

        self.assertEqual(selected, {3, 7})

    def test_audit_database_sample_includes_query_positives(self):
        positives = [
            np.array([4, 6]),
            np.array([], dtype=np.int64),
            np.array([2]),
        ]

        selected = rank_eval.select_audit_database_indices(
            positives,
            np.array([0, 2]),
            database_size=10,
            requested_size=6,
        )

        self.assertIn(4, selected)
        self.assertIn(6, selected)
        self.assertIn(2, selected)
        self.assertEqual(len(selected), 6)

    def test_sampled_attack_targets_use_sampled_database_rows(self):
        args = Namespace(adv_negatives=2)
        sampled_database_indices = np.array([4, 6, 2, 0, 1], dtype=np.int64)
        database_features = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ],
            dtype=np.float32,
        )
        query_features = np.array([[0.1, 0.0]], dtype=np.float32)
        sampled_positives = [np.array([0, 1], dtype=np.int64)]

        targets = rank_eval.build_sampled_attack_targets(
            args,
            sampled_database_indices,
            database_features,
            query_features,
            np.array([12], dtype=np.int64),
            sampled_positives,
        )

        self.assertEqual(targets[0]["query_index"], 12)
        self.assertEqual(targets[0]["query_feature_index"], 0)
        self.assertEqual(targets[0]["positive_index"], 0)
        np.testing.assert_array_equal(targets[0]["negative_indexes"], np.array([2, 3]))

    def test_query_diagnostics_compute_margins_and_cwr_estimate(self):
        database = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
            ],
            dtype=np.float32,
        )
        clean_queries = np.array([[0.1, 0.0]], dtype=np.float32)
        attacked_queries = np.array([[1.8, 0.0]], dtype=np.float32)
        targets = [
            {
                "query_index": 5,
                "positive_index": 0,
                "negative_indexes": np.array([1, 2], dtype=np.int64),
            }
        ]

        rows = rank_eval.compute_query_diagnostic_rows(
            database,
            clean_queries,
            attacked_queries,
            [np.array([0], dtype=np.int64)],
            np.array([5], dtype=np.int64),
            targets,
            np.array([0.2], dtype=np.float32),
            clean_query_feature_indices=np.array([0], dtype=np.int64),
        )

        self.assertEqual(rows[0]["query_index"], 5)
        self.assertEqual(rows[0]["clean_nearest_positive_rank"], 1)
        self.assertEqual(rows[0]["attacked_nearest_positive_rank"], 3)
        self.assertTrue(rows[0]["attack_success"])
        self.assertAlmostEqual(rows[0]["clean_positive_distance"], 0.1, places=6)
        self.assertAlmostEqual(rows[0]["clean_nearest_negative_distance"], 0.9, places=6)
        self.assertAlmostEqual(rows[0]["clean_margin"], 0.8, places=6)
        self.assertEqual(rows[0]["rank_displacement"], 2)
        self.assertAlmostEqual(rows[0]["rho_q"], 1.7, places=6)
        self.assertEqual(rows[0]["cwr_estimate"], 3)
        self.assertEqual(rows[0]["selected_hard_negative_indexes"], "1 2")


if __name__ == "__main__":
    unittest.main()
