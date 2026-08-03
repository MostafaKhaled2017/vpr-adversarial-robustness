import csv
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.paired_analysis import compute_paired_ccasr, load_per_query_rows
from src.rank_eval import PER_QUERY_RANK_FIELDNAMES


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PER_QUERY_RANK_FIELDNAMES))
        writer.writeheader()
        writer.writerows(rows)


def make_row(query_id, model_tag, clean_rank, attacked_rank, dataset="msls", condition="rank_pgd_linf_eps_0.01"):
    return {
        "query_id": query_id,
        "dataset": dataset,
        "model_tag": model_tag,
        "checkpoint_tag": f"{model_tag}.pth",
        "condition": condition,
        "epsilon": 0.01,
        "clean_rank": clean_rank,
        "attacked_rank": attacked_rank,
        "clean_correct_at_1": clean_rank == 1,
        "attacked_correct_at_1": attacked_rank == 1,
    }


# The trained model is clean-correct on {1, 3}, a strict subset of the base model's {1, 2, 3}.
# Base breaks on both intersection queries; trained breaks on one. Per-model and intersection
# CC-ASR therefore differ for the base model by construction.
BASE_ROWS = [
    make_row(1, "base", clean_rank=1, attacked_rank=5),
    make_row(2, "base", clean_rank=1, attacked_rank=1),
    make_row(3, "base", clean_rank=1, attacked_rank=7),
    make_row(4, "base", clean_rank=4, attacked_rank=9),
]
TRAINED_ROWS = [
    make_row(1, "trained", clean_rank=1, attacked_rank=1),
    make_row(2, "trained", clean_rank=3, attacked_rank=6),
    make_row(3, "trained", clean_rank=1, attacked_rank=4),
    make_row(4, "trained", clean_rank=5, attacked_rank=8),
]


class PairedCcasrTests(unittest.TestCase):
    def test_per_model_and_intersection_ccasr_differ(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_csv = Path(temp_dir) / "base.csv"
            trained_csv = Path(temp_dir) / "trained.csv"
            write_rows(base_csv, BASE_ROWS)
            write_rows(trained_csv, TRAINED_ROWS)

            summary = compute_paired_ccasr(base_csv, trained_csv)

        self.assertAlmostEqual(summary["ccasr_own_base"], 2.0 / 3.0 * 100.0)
        self.assertAlmostEqual(summary["ccasr_own_trained"], 1.0 / 2.0 * 100.0)
        self.assertAlmostEqual(summary["ccasr_intersection_base"], 100.0)
        self.assertAlmostEqual(summary["ccasr_intersection_trained"], 50.0)
        self.assertEqual(summary["n_intersection"], 2)
        self.assertNotAlmostEqual(summary["ccasr_own_base"], summary["ccasr_intersection_base"])

    def test_paired_summary_reports_own_clean_correct_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_csv = Path(temp_dir) / "base.csv"
            trained_csv = Path(temp_dir) / "trained.csv"
            write_rows(base_csv, BASE_ROWS)
            write_rows(trained_csv, TRAINED_ROWS)

            summary = compute_paired_ccasr(base_csv, trained_csv)

        self.assertEqual(summary["n_clean_correct_base"], 3)
        self.assertEqual(summary["n_clean_correct_trained"], 2)
        self.assertEqual(summary["n_paired"], 4)
        self.assertEqual(summary["model_tag_base"], "base")
        self.assertEqual(summary["model_tag_trained"], "trained")

    def test_single_file_with_both_models_requires_explicit_tags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            combined_csv = Path(temp_dir) / "per_query_ranks.csv"
            write_rows(combined_csv, [*BASE_ROWS, *TRAINED_ROWS])

            with self.assertRaisesRegex(ValueError, "model_tag"):
                compute_paired_ccasr(combined_csv, combined_csv)

            summary = compute_paired_ccasr(
                combined_csv,
                combined_csv,
                base_model_tag="base",
                trained_model_tag="trained",
            )

        self.assertAlmostEqual(summary["ccasr_intersection_base"], 100.0)
        self.assertAlmostEqual(summary["ccasr_intersection_trained"], 50.0)

    def test_conditions_are_paired_independently(self):
        other_condition_rows = [
            make_row(1, "base", clean_rank=1, attacked_rank=1, condition="rank_pgd_linf_eps_0.1"),
            make_row(3, "base", clean_rank=1, attacked_rank=1, condition="rank_pgd_linf_eps_0.1"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            base_csv = Path(temp_dir) / "base.csv"
            trained_csv = Path(temp_dir) / "trained.csv"
            write_rows(base_csv, [*BASE_ROWS, *other_condition_rows])
            write_rows(trained_csv, TRAINED_ROWS)

            summary = compute_paired_ccasr(base_csv, trained_csv, condition="rank_pgd_linf_eps_0.01")

        self.assertEqual(summary["n_paired"], 4)
        self.assertAlmostEqual(summary["ccasr_intersection_base"], 100.0)

    def test_unpaired_queries_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_csv = Path(temp_dir) / "base.csv"
            trained_csv = Path(temp_dir) / "trained.csv"
            write_rows(base_csv, BASE_ROWS)
            write_rows(trained_csv, TRAINED_ROWS[:-1])

            with self.assertRaisesRegex(ValueError, "same queries"):
                compute_paired_ccasr(base_csv, trained_csv)

    def test_empty_intersection_reports_zero_rates(self):
        base_rows = [make_row(1, "base", clean_rank=1, attacked_rank=4)]
        trained_rows = [make_row(1, "trained", clean_rank=3, attacked_rank=6)]
        with tempfile.TemporaryDirectory() as temp_dir:
            base_csv = Path(temp_dir) / "base.csv"
            trained_csv = Path(temp_dir) / "trained.csv"
            write_rows(base_csv, base_rows)
            write_rows(trained_csv, trained_rows)

            summary = compute_paired_ccasr(base_csv, trained_csv)

        self.assertEqual(summary["n_intersection"], 0)
        self.assertEqual(summary["ccasr_intersection_base"], 0.0)
        self.assertEqual(summary["ccasr_intersection_trained"], 0.0)
        self.assertEqual(summary["ccasr_own_trained"], 0.0)

    def test_rows_round_trip_with_typed_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_csv = Path(temp_dir) / "base.csv"
            write_rows(base_csv, BASE_ROWS)

            rows = load_per_query_rows(base_csv)

        self.assertEqual(rows[0]["query_id"], 1)
        self.assertEqual(rows[0]["clean_rank"], 1)
        self.assertIs(rows[0]["clean_correct_at_1"], True)
        self.assertIs(rows[1]["attacked_correct_at_1"], True)
        self.assertIs(rows[3]["clean_correct_at_1"], False)


if __name__ == "__main__":
    unittest.main()
