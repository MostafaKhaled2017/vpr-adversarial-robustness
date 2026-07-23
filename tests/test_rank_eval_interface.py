import contextlib
import io
import sys
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch


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
        self.assertIn("--model_type", parser._option_string_actions)
        self.assertIn("--model_paths", parser._option_string_actions)
        self.assertIn("--model_tags", parser._option_string_actions)
        self.assertTrue(parser._option_string_actions["--model_type"].required)
        self.assertTrue(parser._option_string_actions["--model_paths"].required)
        self.assertNotIn("--models", parser._option_string_actions)
        self.assertNotIn("--model", parser._option_string_actions)
        self.assertIn("--max_dataset_samples", parser._option_string_actions)
        self.assertIn("--audit_attack_implementation", parser._option_string_actions)
        self.assertIn("--audit_output_json", parser._option_string_actions)
        self.assertIn("--audit_sample_database_size", parser._option_string_actions)
        self.assertIn("--trace_query_indices", parser._option_string_actions)
        self.assertIn("--trace_output_dir", parser._option_string_actions)
        self.assertIn("--diagnostics_output_dir", parser._option_string_actions)
        self.assertIn("--compute_diagnostics", parser._option_string_actions)
        self.assertFalse(parser._option_string_actions["--compute_diagnostics"].default)
        self.assertIn("--save_attack_images", parser._option_string_actions)
        self.assertIn("--save_attack_image_count", parser._option_string_actions)
        self.assertIn("--attack_image_output_dir", parser._option_string_actions)
        self.assertIn("--attack_image_amplification", parser._option_string_actions)
        self.assertNotIn("--resume", parser._option_string_actions)
        self.assertNotIn("--eval_dataset_name", parser._option_string_actions)

    def test_model_type_and_model_paths_are_required(self):
        parser = rank_eval.build_parser()

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--datasets", "msls", "--epsilons", "0.01"])

    def test_legacy_model_options_are_rejected(self):
        parser = rank_eval.build_parser()
        common = ["--datasets", "msls", "--epsilons", "0.01"]

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([*common, "--model_type", "supervlad", "--models", "base.pth"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([*common, "--model", "supervlad", "--model_paths", "base.pth"])

    def test_shared_attacks_flag_defaults_to_per_model(self):
        parser = rank_eval.build_parser()

        self.assertIn("--shared_attacks", parser._option_string_actions)
        self.assertFalse(parser._option_string_actions["--shared_attacks"].default)

    def test_attack_groups_default_to_one_group_per_model(self):
        args = Namespace(shared_attacks=False, model_tags=["base", "adv"])
        models = {"base": ("model_base", "args_base"), "adv": ("model_adv", "args_adv")}

        groups = list(rank_eval.attack_groups(args, models))

        self.assertEqual(
            groups,
            [
                ("base", {"base": ("model_base", "args_base")}),
                ("adv", {"adv": ("model_adv", "args_adv")}),
            ],
        )

    def test_attack_groups_shared_mode_uses_first_model_for_all(self):
        args = Namespace(shared_attacks=True, model_tags=["base", "adv"])
        models = {"base": ("model_base", "args_base"), "adv": ("model_adv", "args_adv")}

        groups = list(rank_eval.attack_groups(args, models))

        self.assertEqual(groups, [("base", models)])

    def test_boq_and_mixvpr_resolve_reference_input_sizes(self):
        boq_args = Namespace(
            boq_backbone="Dinov2",
            boq_descriptors_dimension=None,
            resize=None,
            test_method="hard_resize",
        )
        mixvpr_args = Namespace(
            mixvpr_descriptors_dimension=None,
            resize=None,
            test_method="hard_resize",
        )

        rank_eval.get_model_adapter("boq").configure_evaluation(boq_args)
        rank_eval.get_model_adapter("mixvpr").configure_evaluation(mixvpr_args)

        self.assertEqual(boq_args.boq_descriptors_dimension, 12288)
        self.assertEqual(boq_args.resize, [322, 322])
        self.assertEqual(mixvpr_args.mixvpr_descriptors_dimension, 4096)
        self.assertEqual(mixvpr_args.resize, [320, 320])

    def test_boq_and_mixvpr_reject_non_reference_test_methods(self):
        boq_args = Namespace(
            boq_backbone="Dinov2",
            boq_descriptors_dimension=None,
            resize=None,
            test_method="central_crop",
        )
        mixvpr_args = Namespace(
            mixvpr_descriptors_dimension=None,
            resize=None,
            test_method="single_query",
        )

        with self.assertRaisesRegex(ValueError, "hard_resize"):
            rank_eval.get_model_adapter("boq").configure_evaluation(boq_args)
        with self.assertRaisesRegex(ValueError, "hard_resize"):
            rank_eval.get_model_adapter("mixvpr").configure_evaluation(mixvpr_args)

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

    def test_diagnostics_are_explicitly_opt_in(self):
        parser = rank_eval.build_parser()

        args = parser.parse_args(
            [
                "--datasets",
                "msls",
                "--model_type",
                "supervlad",
                "--model_paths",
                "base.pth",
                "--epsilons",
                "0.01",
                "--compute_diagnostics",
            ]
        )

        self.assertTrue(args.compute_diagnostics)

    def test_diagnostics_output_dir_requires_opt_in(self):
        args = rank_eval.build_parser().parse_args(
            [
                "--datasets",
                "msls",
                "--model_type",
                "supervlad",
                "--model_paths",
                "base.pth",
                "--epsilons",
                "0.01",
            ]
        )
        args.diagnostics_output_dir = "test/diagnostics"

        with self.assertRaisesRegex(ValueError, "requires --compute_diagnostics"):
            rank_eval.validate_arguments(args)

    def test_shared_extraction_loads_each_image_once_for_all_models(self):
        class CountingDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.calls = 0
                self.test_method = ""

            def __len__(self):
                return 5

            def __getitem__(self, index):
                self.calls += 1
                return torch.tensor([float(index), float(index + 1)]), index

        class TwoDimensionalModel(torch.nn.Module):
            def forward(self, inputs, queryflag=0):
                del queryflag
                return inputs * 2.0

        class ThreeDimensionalModel(torch.nn.Module):
            def forward(self, inputs, queryflag=0):
                del queryflag
                return torch.cat([inputs, inputs[:, :1] + inputs[:, 1:]], dim=1)

        args = Namespace(device="cpu", num_workers=0)
        dataset = CountingDataset()
        models = {
            "two": (TwoDimensionalModel(), Namespace(features_dim=2)),
            "three": (ThreeDimensionalModel(), Namespace(features_dim=3)),
        }

        features, model_seconds, shared_seconds = rank_eval.extract_features_for_models(
            args,
            dataset,
            models,
            [3, 1, 4],
            desc="test",
            test_method="hard_resize",
            batch_size=2,
        )

        self.assertEqual(dataset.calls, 3)
        self.assertEqual(dataset.test_method, "hard_resize")
        np.testing.assert_array_equal(features["two"], np.array([[6, 8], [2, 4], [8, 10]], dtype=np.float32))
        np.testing.assert_array_equal(
            features["three"],
            np.array([[3, 4, 7], [1, 2, 3], [4, 5, 9]], dtype=np.float32),
        )
        self.assertEqual(set(model_seconds), {"two", "three"})
        self.assertGreaterEqual(shared_seconds, 0.0)

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

    def test_context_targets_are_cached_by_negative_count(self):
        args = Namespace(adv_negatives=2, max_queries=None, model_tags=["base"])
        context = {
            "sampled_gallery": False,
            "eval_ds": object(),
            "clean_features": {
                "base": {
                    "database": np.zeros((4, 2), dtype=np.float32),
                    "queries": np.zeros((2, 2), dtype=np.float32),
                }
            },
            "valid_query_indices": np.array([0, 1], dtype=np.int64),
            "target_cache": {},
        }
        calls = []
        original_build_attack_targets = rank_eval.build_attack_targets

        def fake_build_attack_targets(*_args, **_kwargs):
            calls.append(args.adv_negatives)
            return [{"query_index": 0}], np.array([0, 1], dtype=np.int64)

        try:
            rank_eval.build_attack_targets = fake_build_attack_targets
            rank_eval.get_context_targets(args, context, "base")
            rank_eval.get_context_targets(args, context, "base")
            args.adv_negatives = 3
            rank_eval.get_context_targets(args, context, "base")
        finally:
            rank_eval.build_attack_targets = original_build_attack_targets

        self.assertEqual(calls, [2, 3])

    def test_context_targets_are_cached_per_reference_model(self):
        args = Namespace(adv_negatives=2, max_queries=None, model_tags=["base", "adv"])
        context = {
            "sampled_gallery": False,
            "eval_ds": object(),
            "clean_features": {
                "base": {
                    "database": np.zeros((4, 2), dtype=np.float32),
                    "queries": np.zeros((2, 2), dtype=np.float32),
                },
                "adv": {
                    "database": np.ones((4, 2), dtype=np.float32),
                    "queries": np.ones((2, 2), dtype=np.float32),
                },
            },
            "valid_query_indices": np.array([0, 1], dtype=np.int64),
            "target_cache": {},
        }
        seen_databases = []
        original_build_attack_targets = rank_eval.build_attack_targets

        def fake_build_attack_targets(_args, _eval_ds, database_features, _queries, limit_queries=None):
            del limit_queries
            seen_databases.append(float(database_features[0, 0]))
            return [{"query_index": 0}], np.array([0, 1], dtype=np.int64)

        try:
            rank_eval.build_attack_targets = fake_build_attack_targets
            rank_eval.get_context_targets(args, context, "base")
            rank_eval.get_context_targets(args, context, "adv")
            rank_eval.get_context_targets(args, context, "base")
        finally:
            rank_eval.build_attack_targets = original_build_attack_targets

        self.assertEqual(seen_databases, [0.0, 1.0])
        self.assertEqual(set(context["target_cache"]), {("base", 2), ("adv", 2)})

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

    def test_batched_query_diagnostics_match_legacy_calculation(self):
        rng = np.random.default_rng(11)
        database = rng.normal(size=(23, 5)).astype(np.float32)
        clean_queries = rng.normal(size=(4, 5)).astype(np.float32)
        attacked_queries = (clean_queries + rng.normal(scale=0.02, size=(4, 5))).astype(np.float32)
        positives = [np.array([index, index + 6], dtype=np.int64) for index in range(4)]
        valid_indices = np.arange(4, dtype=np.int64)
        targets = [
            {
                "query_index": index,
                "positive_index": int(positive_indexes[0]),
                "negative_indexes": np.array([20, 21], dtype=np.int64),
            }
            for index, positive_indexes in enumerate(positives)
        ]

        rows = rank_eval.compute_query_diagnostic_rows(
            database,
            clean_queries,
            attacked_queries,
            positives,
            valid_indices,
            targets,
            np.full(4, 0.02, dtype=np.float32),
            chunk_size=2,
        )

        float_fields = {
            "clean_positive_distance",
            "clean_nearest_negative_distance",
            "clean_margin",
            "attacked_positive_distance",
            "attacked_nearest_negative_distance",
            "attacked_margin",
            "rho_q",
        }
        for index, row in enumerate(rows):
            positive_indexes = positives[index]
            negative_indexes = np.setdiff1d(np.arange(len(database)), positive_indexes)
            clean_distances = np.linalg.norm(database - clean_queries[index][None, :], axis=1)
            attacked_distances = np.linalg.norm(database - attacked_queries[index][None, :], axis=1)
            clean_positive = float(clean_distances[targets[index]["positive_index"]])
            attacked_positive = float(attacked_distances[targets[index]["positive_index"]])
            clean_negative = float(np.min(clean_distances[negative_indexes]))
            attacked_negative = float(np.min(attacked_distances[negative_indexes]))
            rho_q = float(np.linalg.norm(attacked_queries[index] - clean_queries[index]))
            expected = {
                "clean_positive_distance": clean_positive,
                "clean_nearest_negative_distance": clean_negative,
                "clean_margin": clean_negative - clean_positive,
                "attacked_positive_distance": attacked_positive,
                "attacked_nearest_negative_distance": attacked_negative,
                "attacked_margin": attacked_negative - attacked_positive,
                "rho_q": rho_q,
            }
            for field in float_fields:
                self.assertTrue(np.isclose(row[field], expected[field], rtol=1e-5, atol=1e-6), field)

            clean_rank = rank_eval._distance_to_rank(database, clean_queries[index], positive_indexes)
            attacked_rank = rank_eval._distance_to_rank(database, attacked_queries[index], positive_indexes)
            cwr = 1 + np.count_nonzero(
                np.maximum(clean_distances[negative_indexes] - rho_q, 0.0) <= clean_positive + rho_q
            )
            self.assertEqual(row["clean_nearest_positive_rank"], clean_rank)
            self.assertEqual(row["attacked_nearest_positive_rank"], attacked_rank)
            self.assertEqual(row["attack_success"], clean_rank == 1 and attacked_rank > 1)
            self.assertEqual(row["cwr_estimate"], cwr)

    def test_attack_generation_seed_is_stable_and_model_dependent(self):
        first = rank_eval.attack_generation_seed(0, "msls", "base", "rank_pgd_linf_eps_0.01")
        second = rank_eval.attack_generation_seed(0, "msls", "base", "rank_pgd_linf_eps_0.01")
        other_model = rank_eval.attack_generation_seed(0, "msls", "adv", "rank_pgd_linf_eps_0.01")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other_model)
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 2**63)

    def test_reseed_attack_rng_makes_draws_reproducible(self):
        args = Namespace(shared_attacks=False, seed=0)

        rank_eval.reseed_attack_rng(args, "msls", "base", "rank_pgd_linf_eps_0.01")
        first_draw = torch.rand(4)
        rank_eval.reseed_attack_rng(args, "msls", "base", "rank_pgd_linf_eps_0.01")
        second_draw = torch.rand(4)

        self.assertTrue(torch.equal(first_draw, second_draw))

    def test_reseed_attack_rng_is_noop_in_shared_mode_and_for_seed_minus_one(self):
        torch.manual_seed(123)
        expected = torch.rand(4)

        torch.manual_seed(123)
        rank_eval.reseed_attack_rng(Namespace(shared_attacks=True, seed=0), "msls", "base", "c")
        shared_draw = torch.rand(4)

        torch.manual_seed(123)
        rank_eval.reseed_attack_rng(Namespace(shared_attacks=False, seed=-1), "msls", "base", "c")
        unseeded_draw = torch.rand(4)

        self.assertTrue(torch.equal(expected, shared_draw))
        self.assertTrue(torch.equal(expected, unseeded_draw))


if __name__ == "__main__":
    unittest.main()
