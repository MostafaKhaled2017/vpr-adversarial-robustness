import sys
import unittest
from argparse import Namespace
from pathlib import Path


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
        self.assertNotIn("--resume", parser._option_string_actions)
        self.assertNotIn("--eval_dataset_name", parser._option_string_actions)

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


if __name__ == "__main__":
    unittest.main()
