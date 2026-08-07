import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVLAD_ROOT = REPO_ROOT / "third_party" / "SuperVLAD"
if str(SUPERVLAD_ROOT) not in sys.path:
    sys.path.insert(0, str(SUPERVLAD_ROOT))

from src.phase1_summary import (
    build_three_way_table,
    format_table_markdown,
    parse_model_tag,
    summarize_seeds,
)


def make_condition(clean_r1, attacked_r1, ccasr):
    return {
        "clean_all_queries": {"recalls": {"R@1": clean_r1, "R@5": 0.0, "R@10": 0.0, "R@100": 0.0}},
        "rank_pgd_linf_eps_0.01": {
            "recalls": {"R@1": attacked_r1, "R@5": 0.0, "R@10": 0.0, "R@100": 0.0},
            "epsilon": 0.01,
            "attack_success": {"clean_correct": {"rate": ccasr}},
            "rank_displacement": {"median": 3.0, "mean": 10.0},
        },
    }


def make_results(tag_to_values):
    return {
        "results": {
            "sped": {tag: make_condition(*values) for tag, values in tag_to_values.items()},
        }
    }


class ModelTagParsingTests(unittest.TestCase):
    def test_seeded_tag_splits_into_arm_and_seed(self):
        self.assertEqual(parse_model_tag("clean_ft_s1"), ("clean_ft", 1))

    def test_unseeded_tag_has_no_seed(self):
        self.assertEqual(parse_model_tag("pretrained"), ("pretrained", None))

    def test_arm_name_containing_digits_is_preserved(self):
        self.assertEqual(parse_model_tag("pat_v2_s0"), ("pat_v2", 0))


class SeedAggregationTests(unittest.TestCase):
    def test_mean_and_range_over_two_seeds(self):
        summary = summarize_seeds([70.0, 74.0])
        self.assertAlmostEqual(summary["mean"], 72.0)
        self.assertAlmostEqual(summary["min"], 70.0)
        self.assertAlmostEqual(summary["max"], 74.0)
        self.assertAlmostEqual(summary["half_range"], 2.0)
        self.assertEqual(summary["n"], 2)

    def test_single_seed_has_zero_range(self):
        summary = summarize_seeds([70.0])
        self.assertAlmostEqual(summary["mean"], 70.0)
        self.assertAlmostEqual(summary["half_range"], 0.0)
        self.assertEqual(summary["n"], 1)


class ThreeWayTableTests(unittest.TestCase):
    def setUp(self):
        # pretrained is the reference; clean-FT reproduces part of the robustness gain,
        # which is exactly the confound Phase 1 exists to expose.
        self.results = make_results(
            {
                "pretrained": (90.0, 50.0, 40.0),
                "clean_ft_s0": (88.0, 58.0, 32.0),
                "clean_ft_s1": (86.0, 62.0, 30.0),
                "pat_s0": (82.0, 66.0, 26.0),
                "pat_s1": (80.0, 70.0, 24.0),
            }
        )

    def write_results(self, directory):
        path = Path(directory) / "rank_eval_results.json"
        path.write_text(json.dumps(self.results))
        return path

    def test_table_has_one_row_per_arm_and_epsilon(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = build_three_way_table([self.write_results(directory)])

        self.assertEqual({row["arm"] for row in rows}, {"pretrained", "clean_ft", "pat"})
        self.assertEqual({row["dataset"] for row in rows}, {"sped"})
        self.assertEqual({row["epsilon"] for row in rows}, {0.01})

    def test_seeded_arms_report_mean_and_range(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = build_three_way_table([self.write_results(directory)])

        clean_ft = next(row for row in rows if row["arm"] == "clean_ft")
        self.assertEqual(clean_ft["n_seeds"], 2)
        self.assertAlmostEqual(clean_ft["clean_r1"]["mean"], 87.0)
        self.assertAlmostEqual(clean_ft["clean_r1"]["half_range"], 1.0)
        self.assertAlmostEqual(clean_ft["attacked_r1"]["mean"], 60.0)
        self.assertAlmostEqual(clean_ft["ccasr_own"]["mean"], 31.0)

    def test_deltas_are_measured_against_the_pretrained_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = build_three_way_table([self.write_results(directory)])

        pat = next(row for row in rows if row["arm"] == "pat")
        # PAT mean clean R@1 is 81.0 against a pretrained reference of 90.0.
        self.assertAlmostEqual(pat["clean_r1_delta_vs_pretrained"], -9.0)
        self.assertAlmostEqual(pat["attacked_r1_delta_vs_pretrained"], 18.0)

        clean_ft = next(row for row in rows if row["arm"] == "clean_ft")
        # The confound: clean fine-tuning alone already recovers 10 of PAT's 18 points.
        self.assertAlmostEqual(clean_ft["attacked_r1_delta_vs_pretrained"], 10.0)

    def test_pretrained_row_has_no_seed_spread(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = build_three_way_table([self.write_results(directory)])

        pretrained = next(row for row in rows if row["arm"] == "pretrained")
        self.assertEqual(pretrained["n_seeds"], 1)
        self.assertAlmostEqual(pretrained["clean_r1"]["half_range"], 0.0)
        self.assertAlmostEqual(pretrained["clean_r1_delta_vs_pretrained"], 0.0)

    def test_markdown_rendering_lists_every_arm(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = build_three_way_table([self.write_results(directory)])

        markdown = format_table_markdown(rows)
        for arm in ("pretrained", "clean_ft", "pat"):
            self.assertIn(arm, markdown)
        self.assertIn("| dataset |", markdown)

    def test_missing_pretrained_arm_does_not_crash(self):
        self.results["results"]["sped"].pop("pretrained")
        with tempfile.TemporaryDirectory() as directory:
            rows = build_three_way_table([self.write_results(directory)])

        self.assertEqual({row["arm"] for row in rows}, {"clean_ft", "pat"})
        self.assertTrue(all(row["clean_r1_delta_vs_pretrained"] is None for row in rows))


if __name__ == "__main__":
    unittest.main()
